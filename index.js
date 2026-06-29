const express = require('express');
const cors = require('cors');
require('dotenv').config();

const OpenAI = require('openai');

const app = express();

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
  next();
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const TAX_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
const INSIGHTS_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';

const taxCache = new Map();
const CACHE_TTL_MS = 1000 * 60 * 30;
const MAX_CACHE_ITEMS = 100;

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

    if (!query || !String(query).trim()) {
      return res.status(400).json({ error: 'Missing query.' });
    }

    const normalizedDepth = ['quick', 'standard', 'deep'].includes(depth)
      ? depth
      : 'standard';

    const cleanQuery = String(query).trim();

    const cacheKey = JSON.stringify({
      stateCode,
      stateName,
      taxYear,
      clientType,
      depth: normalizedDepth,
      query: cleanQuery.toLowerCase()
    });

    const cached = getCached(cacheKey);
    if (cached) {
      return res.json({ result: cached, cached: true });
    }

    const prompt = buildTaxPrompt({
      stateCode,
      stateName,
      taxYear,
      clientType,
      depth: normalizedDepth,
      query: cleanQuery
    });

    const response = await withTimeout(
      client.responses.create({
        model: TAX_MODEL,
        input: prompt,
        temperature: 0.1,
        max_output_tokens: getMaxOutputTokens(normalizedDepth)
      }),
      getTimeoutMs(normalizedDepth)
    );

    const raw = response.output_text || '';
    const parsed = safeParseJson(raw);

    const result = normalizeTaxResult(parsed, raw, {
      stateName,
      taxYear,
      depth: normalizedDepth
    });

    setCached(cacheKey, result);

    res.json({ result });
  } catch (error) {
    console.error('Tax search failed:', error);

    res.status(500).json({
      error: cleanErrorMessage(error)
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
        model: INSIGHTS_MODEL,
        temperature: 0.1,
        max_output_tokens: 700,
        input: `
You are a CPA financial analyst. Rewrite the analysis below into a polished internal review memo.

Rules:
- No disclaimers, notes, signatures, or bullet points.
- Do not mention that the analysis is based on provided data.
- Prioritize the biggest review issues.
- Focus on financial meaning, not just number changes.
- Keep it to 1-2 short paragraphs.

Analysis:
${hiddenSummary}
`
      }),
      25000
    );

    res.json({
      insights: response.output_text || ''
    });
  } catch (error) {
    console.error('AI insights failed:', error);

    res.status(500).json({
      error: cleanErrorMessage(error)
    });
  }
});

function buildTaxPrompt({
  stateCode,
  stateName,
  taxYear,
  clientType,
  depth,
  query
}) {
  return `
You are ReviewRequest Tax Guidance, a CPA-focused tax assistant.

Return ONLY valid JSON. No markdown. No commentary. No trailing commas.

Target:
State: ${stateName || 'Federal only'} (${stateCode || 'US'})
Tax year: ${taxYear || 'current'}
Client type: ${clientType || 'General'}
Depth: ${depth}
Question: ${query}

Accuracy rules:
- Give the likely answer first.
- Check if the fact pattern fits the selected client type.
- If facts are unusual, set factPatternWarning and explain it in summary.
- Future tax years: do not use "High" confidence unless the rule is highly stable. Prefer "Medium" and say future guidance should be verified.
- Separate federal and state treatment when both matter.
- Do not invent exact Code sections, publication numbers, form numbers, or URLs unless confident.
- If source URL is uncertain, leave url as "".
- Source titles must be specific official documents, forms, instructions, topics, publications, statutes, regulations, or agency pages. Avoid vague source titles.
- Be practical for CPA return prep, review, documentation, and client follow-up.

Depth rules:
${getDepthRules(depth)}

JSON shape:
{
  "state": "",
  "taxYear": "",
  "confidence": "High | Medium | Low",
  "factPatternWarning": "",
  "summary": "",
  "federalTreatment": "",
  "stateTreatment": "",
  "cpaImpact": "",
  "affectedClients": [],
  "actionItems": [],
  "clientQuestions": [],
  "documentation": [],
  "risks": [],
  "commonMistakes": [],
  "sources": [
    { "title": "", "url": "" }
  ]
}
`;
}

function getDepthRules(depth) {
  if (depth === 'quick') {
    return `
Quick:
- Summary: 3-5 sentences.
- CPA impact: 1-2 sentences.
- Action items: 2-3.
- Client questions: 1-2.
- Documentation: 1-2.
- Risks: 1-2.
- Common mistakes: 1.
- Sources: 2-3.
`;
  }

  if (depth === 'deep') {
    return `
Detailed:
- Summary: detailed but readable.
- Include issue framing, federal treatment, state treatment, exceptions, documentation, audit risks, planning notes, and client follow-up where relevant.
- Action items: 7-10.
- Client questions: 4-7.
- Documentation: 4-7.
- Risks: 5-8.
- Common mistakes: 4-6.
- Sources: 5-8.
- If facts are unrealistic, make that the first point.
`;
  }

  return `
Standard:
- Summary: 1-2 useful paragraphs.
- Include key rule, exceptions, filing impact, review impact.
- Action items: 4-5.
- Client questions: 2-4.
- Documentation: 2-4.
- Risks: 2-4.
- Common mistakes: 2-3.
- Sources: 3-5.
`;
}

function getMaxOutputTokens(depth) {
  if (depth === 'quick') return 900;
  if (depth === 'deep') return 2200;
  return 1400;
}

function getTimeoutMs(depth) {
  if (depth === 'quick') return 18000;
  if (depth === 'deep') return 35000;
  return 25000;
}

function normalizeTaxResult(parsed, raw, context) {
  const fallback = {
    state: context.stateName || 'Federal',
    taxYear: context.taxYear || '',
    confidence: 'Low',
    factPatternWarning: '',
    summary: raw || 'No result returned.',
    federalTreatment: '',
    stateTreatment: '',
    cpaImpact: 'Review official guidance before relying on this result.',
    affectedClients: [],
    actionItems: ['Verify the rule against IRS and applicable state guidance.'],
    clientQuestions: [],
    documentation: [],
    risks: ['The model did not return structured source data.'],
    commonMistakes: [],
    sources: []
  };

  if (!parsed || typeof parsed !== 'object') return fallback;

  return {
    state: asText(parsed.state) || context.stateName || 'Federal',
    taxYear: asText(parsed.taxYear) || context.taxYear || '',
    confidence: normalizeConfidence(parsed.confidence, context.taxYear),
    factPatternWarning: asText(parsed.factPatternWarning),
    summary: asText(parsed.summary) || fallback.summary,
    federalTreatment: asText(parsed.federalTreatment),
    stateTreatment: asText(parsed.stateTreatment),
    cpaImpact: asText(parsed.cpaImpact) || fallback.cpaImpact,
    affectedClients: asArray(parsed.affectedClients, 12),
    actionItems: asArray(parsed.actionItems, 12),
    clientQuestions: asArray(parsed.clientQuestions, 10),
    documentation: asArray(parsed.documentation, 10),
    risks: asArray(parsed.risks, 10),
    commonMistakes: asArray(parsed.commonMistakes, 8),
    sources: normalizeSources(parsed.sources)
  };
}

function normalizeConfidence(value, taxYear) {
  const text = asText(value);
  const year = Number(taxYear);
  const currentYear = new Date().getFullYear();

  if (year && year > currentYear && text === 'High') {
    return 'Medium';
  }

  if (['High', 'Medium', 'Low'].includes(text)) return text;
  return 'Medium';
}

function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];

  return sources
    .map(src => {
      if (typeof src === 'string') {
        return { title: src, url: '' };
      }

      if (!src || typeof src !== 'object') return null;

      return {
        title: asText(src.title) || asText(src.name) || 'Official source to verify',
        url: asText(src.url)
      };
    })
    .filter(Boolean)
    .filter(src => src.title.length > 3)
    .slice(0, 10);
}

function asArray(value, limit = 12) {
  if (!Array.isArray(value)) return [];

  return value
    .map(item => asText(item))
    .filter(Boolean)
    .slice(0, limit);
}

function asText(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim();
}

function safeParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');

    if (start >= 0 && end > start) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch {
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

function getCached(key) {
  const entry = taxCache.get(key);

  if (!entry) return null;

  if (Date.now() - entry.createdAt > CACHE_TTL_MS) {
    taxCache.delete(key);
    return null;
  }

  return entry.value;
}

function setCached(key, value) {
  if (taxCache.size >= MAX_CACHE_ITEMS) {
    const firstKey = taxCache.keys().next().value;
    taxCache.delete(firstKey);
  }

  taxCache.set(key, {
    value,
    createdAt: Date.now()
  });
}

function cleanErrorMessage(error) {
  if (!error) return 'Request failed.';
  return error.message || 'Request failed.';
}

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
