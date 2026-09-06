from transformers import pipeline
from sentence_transformers import util
from app.database import shared_embedding_model, get_cached_doc_embedding
import re

# ---------------------------------------------------------------------------
# NLI model
# ---------------------------------------------------------------------------
nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,
    top_k=None
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
QUERY_DOC_RELEVANCE     = 0.30
CLAIM_DOC_RELEVANCE     = 0.25
CONTRADICTION_THRESHOLD = 0.90
ENTAILMENT_THRESHOLD    = 0.50

# How close the claim's best sentence must be to the truth doc before we'll
# accept "neutral NLI + low contradiction" as a confident paraphrase rather
# than "topically adjacent but actually a different fact". CLAIM_DOC_RELEVANCE
# (0.25) is deliberately low so a real contradiction on a barely-related
# sentence still gets caught upstream by the numeric/NLI checks — but that
# same low bar is NOT enough evidence to auto-VERIFY. Two different facts
# about the same entity (e.g. "GDPR fines are X" vs "GDPR erasure rights")
# routinely score neutral+low-contradiction on NLI while sitting well below
# this similarity floor. A true paraphrase of the same fact scores much
# higher on embedding similarity.
PARAPHRASE_MIN_SIMILARITY = 0.55

_FILLERS = {
    'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were',
    'has', 'have', 'been', 'its', 'their', 'about', 'states', 'says', 'claims',
    'according', 'under', 'which', 'what', 'how', 'when', 'where', 'who', 'is',
    'not', 'can', 'will', 'must', 'should', 'may', 'also', 'only', 'any', 'all'
}

# Words that mean "the number after me is an identifier/label, not a quantity"
# e.g. "Article 17", "Section 302", "Rule 10b-5" — comparing these as if they
# were dollar amounts or day-counts is how you get false CONTRADICTIONs on
# claims that are actually fine. This was a real bug in the previous version:
# any bare number fell into a generic 'number' bucket and got compared
# against any other bare number in the doc, regardless of what it labeled.
_IDENTIFIER_PREFIXES = (
    'article', 'section', 'rule', 'clause', 'chapter', 'part', 'paragraph',
    'act', 'title', 'subsection', 'annex', 'schedule', '§'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _meaningful_word_count(text: str) -> int:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return len([w for w in words if w not in _FILLERS])


def _split_sentences(text: str) -> list:
    """Split into sentences so a multi-sentence claim isn't collapsed into a
    single diluted embedding when checking relevance / feeding NLI."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _batch_embed(texts: list):
    """Encode every text this request needs in a SINGLE model call instead
    of one call per string. Batching is the single biggest, safest speed
    win available here — it doesn't change the math, just how many times
    the model is invoked."""
    return shared_embedding_model.encode(texts, convert_to_tensor=True)


def _nli_scores(premise: str, hypothesis: str) -> dict:
    results = nli(f"{premise} [SEP] {hypothesis}")[0]
    return {r["label"].upper(): r["score"] for r in results}


def _extract_numbers(text: str) -> list:
    """Extract (value, unit_type) pairs from text, skipping numbers that are
    clearly identifiers (Article 17, Section 302, Rule 10b-5) rather than
    quantities. Unit types group numbers so we only compare apples to apples.
    """
    results = []
    # Track consumed character SPANS of the numeric digits themselves (not
    # just match-start positions). A generic pattern's match can start at a
    # different offset than a specific pattern's match (e.g. currency_large
    # starts at "€", the bare-number pattern starts at the digit right after
    # it) even though they cover the same digits. Position-only dedup let
    # the same digits get typed twice ("€20 million" -> both currency=20e6
    # AND bare number=20), which could later collide with an unrelated bare
    # number elsewhere and trigger a false CONTRADICTION. Span-overlap dedup
    # closes that.
    covered = []

    def _overlaps(a_start, a_end):
        return any(not (a_end <= s or a_start >= e) for s, e in covered)

    patterns = [
        (r'[\$€£¥]?\s*(\d+(?:\.\d+)?)\s*(million|billion)', 'currency_large'),
        (r'\$\s*(\d+(?:\.\d+)?)',                            'dollar'),
        (r'€\s*(\d+(?:\.\d+)?)',                             'euro'),
        (r'(\d+(?:\.\d+)?)\s*%',                             'percent'),
        (r'(\d+(?:\.\d+)?)\s*°?\s*[CcFf]\b',                'temperature'),
        (r'(\d+(?:\.\d+)?)\s*(days?|hours?|years?|months?|weeks?)', 'time'),
        (r'(\d+(?:\.\d+)?)\s*x\b',                          'multiplier'),
        (r'\b(\d+(?:\.\d+)?)\b',                             'number'),
    ]

    for pattern, unit in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            num_start, num_end = m.start(1), m.end(1)
            if _overlaps(num_start, num_end):
                continue

            # Skip if this number is a labeled identifier, not a quantity.
            # Look at up to ~15 chars before the match for words like
            # "Article", "Section", "Rule", etc.
            preceding = text[max(0, num_start - 15):num_start].strip().lower()
            if unit == 'number' and any(
                preceding.endswith(p) or p in preceding.split()[-1:]
                for p in _IDENTIFIER_PREFIXES
            ):
                covered.append((num_start, num_end))
                continue

            covered.append((num_start, num_end))
            try:
                val = float(m.group(1).replace(',', ''))
            except (ValueError, IndexError):
                continue
            if unit == 'currency_large':
                suffix = m.group(2).lower()
                val *= (1_000_000 if 'million' in suffix else 1_000_000_000)
                unit = 'currency'
            results.append((val, unit))

    return results


def _numerical_contradiction(truth: str, claim: str) -> bool:
    """Returns True if the claim contains a number that contradicts truth in
    the same unit category. NLI is blind to numbers ("25 days" vs "20 days"
    scores neutral) — this catches those mismatches directly via regex, not
    embeddings, so it's immune to the whole-claim-similarity dilution issue
    described below in audit_response.

    Uses a relative + absolute tolerance (instead of a strict 0.001 exact
    match) so "approximately $20 million" vs "$20 million" doesn't misfire
    on floating point noise, while still catching real mismatches like
    "25 days" vs "20 days".
    """
    truth_nums = _extract_numbers(truth)
    claim_nums = _extract_numbers(claim)

    if not truth_nums or not claim_nums:
        return False

    truth_by_unit = {}
    for val, unit in truth_nums:
        truth_by_unit.setdefault(unit, []).append(val)

    for c_val, c_unit in claim_nums:
        if c_unit not in truth_by_unit:
            continue
        truth_vals = truth_by_unit[c_unit]
        tolerance = max(0.001, 0.01 * max(abs(c_val), max(abs(t) for t in truth_vals)))
        if not any(abs(c_val - t_val) <= tolerance for t_val in truth_vals):
            print(f"DEBUG - Numerical mismatch: claim={c_val} ({c_unit}), "
                  f"truth has {truth_vals}")
            return True

    return False


# ---------------------------------------------------------------------------
# Main auditor
# ---------------------------------------------------------------------------
def audit_response(truth: str, ai_claim: str, query: str = "") -> str:
    """
    Returns: VERIFIED | CONTRADICTION | NEUTRAL | UNCERTAIN

    NEUTRAL       -> wrong doc, nonsense claim, or completely unrelated topic
    CONTRADICTION -> claim has a detectable factual error (numerical or NLI)
    VERIFIED      -> claim is correct or a faithful paraphrase
    UNCERTAIN     -> genuinely ambiguous; the model isn't confident either
                     way, so this is surfaced honestly instead of silently
                     labeled VERIFIED. Route these to human review.

    Note on the old behavior: previously, ambiguous cases fell through to
    "benefit of the doubt -> VERIFIED". For a hallucination detector that's
    the wrong default — an unclear claim should be flagged, not stamped as
    confirmed. That's the main behavioral change here.
    """

    if not truth or truth.strip() == "No reference found.":
        return "NEUTRAL"

    try:
        # Gate 1 - Nonsense/placeholder claim
        if _meaningful_word_count(ai_claim) < 2:
            print("DEBUG - Claim has no meaningful content -> NEUTRAL")
            return "NEUTRAL"

        # Truth doc embedding is cached (same KB doc gets reused across
        # requests) - skips one full model call per request.
        truth_emb = get_cached_doc_embedding(truth)

        # Everything else this request needs gets embedded in ONE batched
        # call instead of one call per sentence.
        sentences = _split_sentences(ai_claim) or [ai_claim]
        to_embed = sentences[:]
        query_idx = None
        if query and query.strip():
            query_idx = len(to_embed)
            to_embed.append(query)

        batch_emb = _batch_embed(to_embed)
        sentence_embs = batch_emb[:len(sentences)]

        # Gate 2 - query<->doc: is the retrieved doc actually about what was asked?
        if query_idx is not None:
            q_sim = util.cos_sim(batch_emb[query_idx], truth_emb).item()
            print(f"DEBUG - Query<->Doc similarity : {q_sim:.3f}")
            if q_sim < QUERY_DOC_RELEVANCE:
                print("DEBUG - Retrieved doc not relevant to query -> NEUTRAL")
                return "NEUTRAL"

        # Numerical check FIRST, before the claim<->doc relevance gate.
        # A claim like "The rate is 7.45%. All customers must adhere to this
        # exact rate for new accounts." has a second sentence unrelated to
        # the doc; embedding the WHOLE claim as one vector dilutes similarity
        # below the relevance gate before the numerical check ever runs.
        # Numbers are pulled straight from raw text via regex, so this check
        # is immune to that dilution and always runs.
        if _numerical_contradiction(truth, ai_claim):
            print("DEBUG - Numerical contradiction -> CONTRADICTION")
            return "CONTRADICTION"

        # Gate 3 - claim<->doc: pick whichever sentence is most relevant to
        # the truth doc, so an unrelated trailing sentence can't drag the
        # whole-claim similarity down and mask a real contradiction elsewhere.
        sims = util.cos_sim(sentence_embs, truth_emb).squeeze(-1).tolist()
        if isinstance(sims, float):
            sims = [sims]
        c_sim, hypothesis = max(zip(sims, sentences), key=lambda pair: pair[0])

        print(f"DEBUG - Claim<->Doc best-sentence similarity : {c_sim:.3f}  (\"{hypothesis}\")")
        if c_sim < CLAIM_DOC_RELEVANCE:
            print("DEBUG - Claim not relevant to retrieved doc -> NEUTRAL")
            return "NEUTRAL"

        # NLI check on the most relevant sentence only
        scores              = _nli_scores(truth, hypothesis)
        entailment_score    = scores.get("ENTAILMENT",    0)
        contradiction_score = scores.get("CONTRADICTION", 0)
        neutral_score       = scores.get("NEUTRAL",       0)

        print(f"DEBUG - NLI -> entail={entailment_score:.3f}  "
              f"contra={contradiction_score:.3f}  "
              f"neutral={neutral_score:.3f}")

        if contradiction_score >= CONTRADICTION_THRESHOLD:
            print("DEBUG - High NLI contradiction -> CONTRADICTION")
            return "CONTRADICTION"

        if entailment_score >= ENTAILMENT_THRESHOLD:
            print("DEBUG - High entailment -> VERIFIED")
            return "VERIFIED"

        # Neutral-dominant + near-zero contradiction = confident paraphrase
        # (NLI returns neutral, not entailment, for summaries/restatements;
        # contra≈0 confirms it isn't wrong, just worded differently) — but
        # ONLY if the claim is actually close to the truth doc. Passing the
        # 0.25 relevance gate is not enough evidence on its own: a different
        # fact about the same topic can look identical to NLI (neutral,
        # near-zero contradiction) while being flatly wrong. Require real
        # similarity too, or this degrades into "if in doubt, verify it".
        if neutral_score > 0.50 and contradiction_score < 0.15 and c_sim >= PARAPHRASE_MIN_SIMILARITY:
            print(f"DEBUG - Neutral-dominant + low contradiction + high sim ({c_sim:.3f}) -> VERIFIED (paraphrase)")
            return "VERIFIED"

        # Genuinely ambiguous: neither entailment nor contradiction is
        # confident. Surface this honestly rather than guessing VERIFIED.
        print("DEBUG - Ambiguous -> UNCERTAIN (flag for review)")
        return "UNCERTAIN"

    except Exception as e:
        print(f"Auditor Error: {e}")
        return "UNCERTAIN"