// import fs from 'fs';
const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const router  = express.Router();

// Initialise Gemini client once — key never leaves the server
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const fs = require('fs');
const referenceData = fs.readFileSync('C:\\Users\\hp\\Documents\\project-veracity\\example.txt', 'utf-8');
// System prompt designed to produce confident answers that may hallucinate
// specific numbers/dates — perfect for demonstrating VeracityVault
const SYSTEM_PROMPT = `You are a helpful legal and compliance assistant.
Answer questions about regulations, company policies, and legal requirements confidently.
Give specific numbers, dates, and figures in your answers but make them specifically wrong slighly like change the figures etc in every answer.
Keep answers to 2-3 sentences. Be direct and authoritative.
And dont give any unnecessary information be direct.
try sticking to the words/script in the reference data
Do not hedge or say "I'm not sure" — always give a specific answer.
use the following data for reference and answer only from this:

${referenceData}
`;

// ── POST /api/chat ─────────────────────────────────────────────────────────
// Body: { query: string, history: Array<{role: string, text: string}> }
//
// history is the full prior conversation EXCLUDING the current query.
// Each entry: { role: 'user' | 'model', text: string }
//
// Gemini contents format:
//   [ { role: 'user', parts: [{text}] }, { role: 'model', parts: [{text}] }, ... ]
// ──────────────────────────────────────────────────────────────────────────
router.post('/', async (req, res) => {
  const { query, history = [] } = req.body;

  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'query is required' });
  }

  // Validate history — must be array of {role, text}
  const safeHistory = Array.isArray(history)
    ? history.filter(m => m && typeof m.text === 'string' && ['user','model'].includes(m.role))
    : [];

  // ── SSE headers ────────────────────────────────────────────────────────
  res.setHeader('Content-Type',      'text/event-stream');
  res.setHeader('Cache-Control',     'no-cache');
  res.setHeader('Connection',        'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const model = genAI.getGenerativeModel({
      model: 'gemini-3.5-flash-lite',
      systemInstruction: SYSTEM_PROMPT,
      generationConfig: {
        temperature:     0.9,
        maxOutputTokens: 200,
        topP:            0.95,
      }
    });

    // ── Build Gemini contents array ────────────────────────────────────
    // Format: alternating user/model turns + current query at the end.
    // Gemini requires the array to start with 'user' and alternate strictly.
    const contents = [
      // Prior turns from history
      ...safeHistory.map(m => ({
        role:  m.role,
        parts: [{ text: m.text }]
      })),
      // Current user message
      {
        role:  'user',
        parts: [{ text: query }]
      }
    ];

    console.log(`[CHAT] Sending ${contents.length} turns to Gemini (${safeHistory.length} history + 1 new)`);

    const streamResult = await model.generateContentStream({ contents });

    let fullText = '';

    for await (const chunk of streamResult.stream) {
      const token = chunk.text();
      if (token) {
        fullText += token;
        send('token', { text: token });
      }
    }

    send('done', { fullText });

    // ── Call FastAPI auditor ───────────────────────────────────────────
    console.log(`[CHAT] Calling FastAPI at ${process.env.FASTAPI_URL}/verify`);
    const verdict = await callFastAPIVerify(query, fullText);
    console.log(`[CHAT] FastAPI verdict:`, JSON.stringify(verdict));
    send('verdict', verdict);

  } catch (err) {
    console.error('[chat route error]', err.message);
    send('error', { message: err.message || 'Gemini request failed' });
  } finally {
    res.end();
  }
});

// ── Internal: call FastAPI /verify ────────────────────────────────────────
async function callFastAPIVerify(query, response) {
  try {
    const res = await fetch(`${process.env.FASTAPI_URL}/verify`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, response }),
      signal:  AbortSignal.timeout(15000)
    });

    if (!res.ok) throw new Error(`FastAPI returned ${res.status}`);
    return await res.json();

  } catch (err) {
    console.error('[FastAPI verify error]', err.message);
    return {
      status:  'NEUTRAL',
      message: `Auditor unavailable: ${err.message}`
    };
  }
}

module.exports = router;