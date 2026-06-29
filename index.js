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

const TAX_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
const INSIGHTS_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';

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

    if (!query || !String(query).trim()) {
      return res.status(400).json({
        error: 'Missing query.'
      });
    }

    const normalizedDepth = ['quick', 'standard', 'deep'].includes(depth)
      ? depth
      : 'standard';

    const depthRules = getDepthRules(normalizedDepth);

    const prompt = `
You are ReviewRequest Tax Guidance, a CPA-focused tax research assistant.

Your job:
Give practical, accurate, CPA-useful tax guidance with low fluff.

Return ONLY valid JSON.
No markdown.
No commentary outside JSON.
No trailing commas.

Research target:
- State: ${stateName || 'Federal only'} (${stateCode || 'US'})
- Tax year: ${taxYear || 'current'}
- Client type: ${clientType || 'General'}
- Depth: ${normalizedDepth}
- Question: ${query}

Critical accuracy rules:
- First, check whether the fact pattern makes sense for the selected client type.
- If the selected client type and question are inconsistent, flag it clearly.
- Do not pretend an unrealistic fact pattern is normal.
- Example: unemployment compensation is generally paid to individuals, not C corporations. If client type is C corporation, explain that the selected client type is likely wrong or that the relevant recipient may be an employee, shareholder, or owner.
- Separate federal treatment from state treatment when both matter.
- Distinguish rules that are likely known from items that must be verified.
- Do not invent exact Code sections, publication numbers, forms, or URLs unless you are confident.
- If you are not confident about a source URL, leave url as an empty string and give the source title to verify.
- Do not provide legal certainty where verification is required.
- Do not give generic disclaimers. Be specific about what must be checked.

CPA usefulness rules:
- Give the likely answer first.
- Explain the return-preparation impact.
- Explain documentation or workpaper impact.
- Include practical review steps.
- Include common mistakes.
- Include client follow-up questions where useful.
- Keep wording direct and professional.

Depth rules:
${depthRules}

Return this exact JSON shape:
{
  "state": "state or Federal",
  "taxYear": "tax year",
  "confidence": "High | Medium | Low",
  "factPatternWarning": "clear warning if the facts/client type are unusual, otherwise empty string",
  "summary": "plain-English rule summary",
  "federalTreatment": "federal treatment if relevant, otherwise empty string",
  "stateTreatment": "state-specific treatment if relevant, otherwise empty string",
  "cpaImpact": "what this means for the CPA's review work",
  "affectedClients": ["client type or fact pattern"],
  "actionItems": ["specific next step"],
  "clientQuestions": ["question to ask the client"],
  "documentation": ["documents or records to review"],
  "risks": ["risk or uncertainty to verify"],
  "commonMistakes": ["common mistake to avoid"],
  "sources": [
    { "title": "official source title to verify", "url": "" }
  ]
}
`;

    const response = await withTimeout(
      client.responses.create({
        model: TAX_MODEL,
        input: prompt
      }),
      40000
    );

    const raw = response.output_text || '';
    const parsed = safeParseJson(raw);

    const result = normalizeTaxResult(parsed, raw, {
      stateName,
      taxYear
    });

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
      return res.status(400).json({
        error: 'Missing hiddenSummary.'
      });
    }

    const response = await withTimeout(
      client.responses.create({
        model: INSIGHTS_MODEL,
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
      40000
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

function getDepthRules(depth) {
  if (depth === 'quick') {
    return `
Quick mode:
- Summary must be 3-5 sentences.
- Federal treatment: 1-2 sentences max.
- State treatment: 1-2 sentences max.
- CPA impact: 2-3 sentences max.
- Action items: 2-3 items.
- Client questions: 1-3 items.
- Documentation: 1-3 items.
- Risks: 1-2 items.
- Common mistakes: 1-2 items.
- Sources: 2-4 official sources to verify.
`;
  }

  if (depth === 'deep') {
    return `
Detailed review mode:
- Write like an internal CPA research memo.
- Summary must be detailed but still readable.
- Include issue framing, federal treatment, state treatment, exceptions, documentation, audit risks, planning notes, and client follow-up questions where relevant.
- Action items: 8-12 items.
- Client questions: 5-10 items.
- Documentation: 6-10 items.
- Risks: 6-10 items.
- Common mistakes: 5-8 items.
- Sources: 6-10 official sources to verify.
- If the fact pattern is unrealistic, make that the first point in the summary.
`;
  }

  return `
Standard mode:
- Give a practical CPA-ready explanation.
- Summary should be 1-2 solid paragraphs.
- Include key rule, exceptions, and filing/review impact.
- Action items: 4-6 items.
- Client questions: 3-5 items.
- Documentation: 3-6 items.
- Risks: 3-5 items.
- Common mistakes: 2-4 items.
- Sources: 4-6 official sources to verify.
`;
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
    cpaImpact: 'Review the underlying official guidance before relying on this result.',
    affectedClients: [],
    actionItems: ['Verify the rule against IRS and applicable state guidance.'],
    clientQuestions: [],
    documentation: [],
    risks: ['The model did not return structured source data.'],
    commonMistakes: [],
    sources: []
  };

  if (!parsed || typeof parsed !== 'object') {
    return fallback;
  }

  return {
    state: asText(parsed.state) || context.stateName || 'Federal',
    taxYear: asText(parsed.taxYear) || context.taxYear || '',
    confidence: normalizeConfidence(parsed.confidence),
    factPatternWarning: asText(parsed.factPatternWarning),
    summary: asText(parsed.summary) || fallback.summary,
    federalTreatment: asText(parsed.federalTreatment),
    stateTreatment: asText(parsed.stateTreatment),
    cpaImpact: asText(parsed.cpaImpact) || fallback.cpaImpact,
    affectedClients: asArray(parsed.affectedClients),
    actionItems: asArray(parsed.actionItems),
    clientQuestions: asArray(parsed.clientQuestions),
    documentation: asArray(parsed.documentation),
    risks: asArray(parsed.risks),
    commonMistakes: asArray(parsed.commonMistakes),
    sources: normalizeSources(parsed.sources)
  };
}

function normalizeConfidence(value) {
  const text = asText(value);
  if (['High', 'Medium', 'Low'].includes(text)) return text;
  return 'Medium';
}

function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];

  return sources
    .map(src => {
      if (typeof src === 'string') {
        return {
          title: src,
          url: ''
        };
      }

      if (!src || typeof src !== 'object') {
        return null;
      }

      return {
        title: asText(src.title) || asText(src.name) || 'Source to verify',
        url: asText(src.url)
      };
    })
    .filter(Boolean)
    .slice(0, 12);
}

function asArray(value) {
  if (!Array.isArray(value)) return [];

  return value
    .map(item => asText(item))
    .filter(Boolean);
}

function asText(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim();
}

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

function cleanErrorMessage(error) {
  if (!error) return 'Request failed.';
  return error.message || 'Request failed.';
}

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
