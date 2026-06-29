const express = require('express');
const cors = require('cors');
require('dotenv').config();

const OpenAI = require('openai');

const app = express();

app.use(cors());
app.use(express.json({ limit: '3mb' }));

app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
  next();
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

app.get('/', (req, res) => {
  res.send('ReviewRequest Tax Guidance backend is running.');
});

app.post('/tax-search', async (req, res) => {
  try {
    const {
      stateCode,
      stateName,
      taxYear,
      clientType,
      depth,
      query
    } = req.body;

    console.log('Tax search body:', {
      stateCode,
      stateName,
      taxYear,
      clientType,
      depth,
      query
    });

    if (!query) {
      return res.status(400).json({ error: 'Missing query.' });
    }

    const detailLevel = depth === 'deep'
      ? 'Use more detail, but stay practical.'
      : depth === 'quick'
        ? 'Be brief and focus only on the likely answer and key action.'
        : 'Use a balanced CPA-ready summary.';

    const prompt = `
You are a CPA tax research assistant. The user is researching a tax rule.

Return ONLY valid JSON. No markdown. No commentary outside JSON.

Research target:
- State: ${stateName || 'Federal only'} (${stateCode || 'US'})
- Tax year: ${taxYear || 'current'}
- Client type: ${clientType || 'General'}
- Depth: ${depth || 'standard'}
- Question: ${query}

Instructions:
- Use your tax knowledge to give practical CPA guidance.
- Prioritize official sources that the CPA should verify, such as IRS, Treasury, state department of revenue, state tax agency, official forms, official instructions, notices, and FAQs.
- If the topic is state-specific, include federal context only when useful.
- Distinguish confirmed rules from items that need verification.
- Do not invent exact source URLs if you are not sure.
- Keep it useful for a CPA working inside a client file.
- ${detailLevel}

JSON shape:
{
  "state": "state or Federal",
  "taxYear": "tax year",
  "confidence": "High | Medium | Low",
  "summary": "plain-English rule summary",
  "cpaImpact": "what this means for the CPA's review work",
  "affectedClients": ["client type or fact pattern", "..."],
  "actionItems": ["specific next step", "..."],
  "risks": ["risk or uncertainty to verify", "..."],
  "sources": [
    { "title": "source title to verify", "url": "https://..." }
  ]
}
`;

    const response = await withTimeout(
      client.responses.create({
        model: process.env.OPENAI_MODEL || 'gpt-4.1-mini',
        input: prompt
      }),
      45000
    );

    const raw = response.output_text || '';
    const result = safeParseJson(raw);

    if (!result) {
      return res.json({
        result: {
          state: stateName || 'Federal',
          taxYear: taxYear || '',
          confidence: 'Low',
          summary: raw || 'No result returned.',
          cpaImpact: 'Review the underlying official guidance before relying on this result.',
          affectedClients: [],
          actionItems: ['Verify the rule against IRS and applicable state guidance.'],
          risks: ['The model did not return structured source data.'],
          sources: []
        }
      });
    }

    res.json({ result });
  } catch (error) {
    console.error('Tax search failed:', error);

    res.status(500).json({
      error: error.message || 'Tax search failed.'
    });
  }
});

app.post('/ai-insights', async (req, res) => {
  try {
    const { hiddenSummary } = req.body;

    if (!hiddenSummary) {
      return res.status(400).json({ error: 'Missing hiddenSummary.' });
    }

    const response = await withTimeout(
      client.responses.create({
        model: process.env.OPENAI_MODEL || 'gpt-4.1-mini',
        input: `
You are a CPA financial analyst writing a short internal review memo.

Rewrite the analysis below into a polished CPA-style narrative.

Rules:
- Do not include disclaimers, limitations, notes, or signatures.
- Do not mention that the analysis is based on provided data.
- Do not use bullet points.
- Prioritize the most important review issues first.
- Focus on financial meaning, not just number changes.
- Keep it concise: 1 to 2 short paragraphs.

Analysis:
${hiddenSummary}
`
      }),
      45000
    );

    res.json({ insights: response.output_text });
  } catch (error) {
    console.error('AI insights failed:', error);

    res.status(500).json({
      error: error.message || 'AI insights failed.'
    });
  }
});

function safeParseJson(text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');

    if (start >= 0 && end > start) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch (_) {
        return null;
      }
    }

    return null;
  }
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Request timed out after ${ms / 1000} seconds.`));
      }, ms);
    })
  ]);
}

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
