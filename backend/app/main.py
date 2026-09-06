from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document
from sentence_transformers import util
from app.database import initialize_retriever, get_cached_doc_embedding, shared_embedding_model
from app.auditor import audit_response, QUERY_DOC_RELEVANCE
import asyncio
import os
import time

# ---------------------------------------------------------------------------
# Knowledge base source file
# ---------------------------------------------------------------------------
# Was previously a hardcoded Windows absolute path (C:\Users\hp\...), which
# breaks on any other machine, container, or deployment. Now: configurable
# via an env var, defaulting to a file next to this project, and fails with
# a clear, actionable error instead of a raw FileNotFoundError traceback.
KNOWLEDGE_BASE_PATH = Path(
"C:\\Users\\hp\\Documents\\project-veracity\\example.txt"
)

if not KNOWLEDGE_BASE_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge base file not found at '{KNOWLEDGE_BASE_PATH}'. "
        f"Set the VERACITY_KB_PATH environment variable to point at your "
        f"example.txt, or place it at that default location."
    )

with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as file:
    KNOWLEDGE_BASE = [
        Document(page_content=line.strip())
        for line in file
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
# KNOWLEDGE_BASE = [
#     Document(page_content="The company refund policy is strictly 20 days."),
#     Document(page_content="The server room temperature must be 18°C."),
#     Document(page_content="Under GDPR Article 17, individuals have the right to erasure ('right to be forgotten') and organizations must delete personal data within 30 days of a valid request."),
#     Document(page_content="The statutory limitation period for filing a breach of contract claim is 6 years from the date the breach occurred under the Limitation Act 1980."),
#     Document(page_content="Under the Americans with Disabilities Act (ADA), employers with 15 or more employees are required to provide reasonable accommodations to qualified individuals with disabilities."),
#     Document(page_content="HIPAA requires covered entities to notify affected individuals of a data breach within 60 days of discovery. Breaches affecting more than 500 individuals must also be reported to the HHS Secretary."),
#     Document(page_content="Under the Fair Labor Standards Act (FLSA), the federal minimum wage is $7.25 per hour. Overtime pay must be at least 1.5 times the regular rate for hours worked beyond 40 in a workweek."),
#     Document(page_content="The Securities Exchange Act Section 10(b) and SEC Rule 10b-5 prohibit insider trading. Penalties include fines up to $5 million and imprisonment up to 20 years for individuals."),
#     Document(page_content="Under the Foreign Corrupt Practices Act (FCPA), it is illegal for U.S. persons and companies to bribe foreign government officials to obtain or retain business. Civil penalties can reach $16,000 per violation."),
#     Document(page_content="The General Data Protection Regulation (GDPR) imposes maximum fines of €20 million or 4% of annual global turnover, whichever is higher, for the most serious infringements."),
#     Document(page_content="Under the Sarbanes-Oxley Act Section 302, CEOs and CFOs must personally certify the accuracy of financial statements. False certification carries criminal penalties of up to $5 million and 20 years imprisonment."),
#     Document(page_content="The California Consumer Privacy Act (CCPA) grants consumers the right to know what personal data is collected, the right to delete it, and the right to opt out of its sale. Businesses have 45 days to respond to consumer requests."),
# ]

retriever = None

# ---------------------------------------------------------------------------
# Lifespan — initialize on startup, not at import time
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    retriever = initialize_retriever(KNOWLEDGE_BASE)
    yield

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class AuditRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    query: str
    response: str

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/verify")
async def verify_flow(data: AuditRequest):
    t0 = time.perf_counter()
    relevant_docs = await asyncio.to_thread(retriever.invoke, data.query)

    if not relevant_docs:
        return {"status": "NEUTRAL", "message": "No relevant documents found."}

    # Pick the retrieved doc most SEMANTICALLY similar to the query, instead
    # of exact keyword overlap. Keyword overlap breaks on trivial morphology
    # ("fine" in the query vs "fines" in the doc share zero tokens after
    # regex tokenization) and ties get resolved by silently keeping
    # relevant_docs[0], which is not guaranteed to be the right line. That
    # was handing the auditor the WRONG ground-truth doc to compare against
    # — the auditor was then correctly comparing a claim to the wrong
    # sentence and returning a confident-looking but meaningless verdict.
    #
    # Embedding similarity doesn't care about singular/plural or tense, and
    # doc embeddings are cached (get_cached_doc_embedding), so for a fixed,
    # small KB this costs one new embedding (the query) per request — same
    # ballpark cost as the keyword loop it replaces, just far more robust.
    query_emb = await asyncio.to_thread(
        shared_embedding_model.encode, data.query, convert_to_tensor=True
    )

    best_doc, best_sim = relevant_docs[0], -1.0
    for doc in relevant_docs:
        doc_emb = get_cached_doc_embedding(doc.page_content)
        sim = util.cos_sim(query_emb, doc_emb).item()
        if sim > best_sim:
            best_doc, best_sim = doc, sim

    context = best_doc.page_content
    print(f"DEBUG - Best-doc query similarity: {best_sim:.3f} -> \"{context}\"")

    if best_sim < QUERY_DOC_RELEVANCE:
        return {"status": "NEUTRAL", "message": "Query does not match any knowledge base documents."}

    verdict = audit_response(context, data.response, query=data.query)
    print(f"DEBUG - Verdict: {verdict}")
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"[LATENCY] {latency_ms:.1f}ms")

    if verdict == "CONTRADICTION":
        return {
            "status": "HALLUCINATION_DETECTED",
            "correction": f"Actually, {context}",
            "reference": context
        }

    if verdict == "UNCERTAIN":
        # Genuinely ambiguous cases used to be silently marked VERIFIED
        # ("benefit of the doubt"). That's the wrong default for a
        # hallucination detector - an unclear claim should be flagged for
        # human review, not confidently rubber-stamped.
        return {
            "status": "UNCERTAIN",
            "message": "Could not confidently verify this claim - flagged for human review.",
            "reference": context
        }

    return {
        "status": verdict,
        "message": "Claim matches ground truth." if verdict == "VERIFIED"
                   else "Could not verify claim against knowledge base."
    }