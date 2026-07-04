
'use strict';

const https = require('https');
const http = require('http');
const { URL } = require('url');

const SIMILARITY_SEARCH_API_BASE_URL = process.env.SIMILARITY_SEARCH_API_BASE_URL || 'https://api.similaritysearch.io/v1';
const SIMILARITY_SEARCH_API_KEY = process.env.SIMILARITY_SEARCH_API_KEY || null;
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_TOP_K = 10;
const MAX_CORPUS_ITEMS = 100000;
const MAX_QUERY_LENGTH = 8192;
const MIN_TOP_K = 1;
const MAX_TOP_K = 1000;

class SimilaritySearchError extends Error {
  constructor(message, statusCode = null, responseBody = null) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

class SimilaritySearchAuthError extends SimilaritySearchError {
  constructor(message) {
    super(message, 401, null);
    this.name = 'SimilaritySearchAuthError';
  }
}

class SimilaritySearchValidationError extends SimilaritySearchError {
  constructor(message) {
    super(message, 422, null);
    this.name = 'SimilaritySearchValidationError';
  }
}

class SimilaritySearchRateLimitError extends SimilaritySearchError {
  constructor(message, retryAfterSeconds = null) {
    super(message, 429, null);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

function resolveApiKey(explicitKey) {
  const key = explicitKey || SIMILARITY_SEARCH_API_KEY;
  if (!key || typeof key !== 'string' || key.trim().length === 0) {
    throw new SimilaritySearchAuthError(
      'API key is required. Set SIMILARITY_SEARCH_API_KEY environment variable or pass apiKey in options.'
    );
  }
  return key.trim();
}

function httpRequest(method, urlString, body, apiKey, timeoutMs) {
  return new Promise((resolve, reject) => {
    let parsedUrl;
    try {
      parsedUrl = new URL(urlString);
    } catch (err) {
      return reject(new SimilaritySearchError(`Invalid URL: ${urlString}`));
    }

    const isHttps = parsedUrl.protocol === 'https:';
    const transport = isHttps ? https : http;
    const bodyBuffer = body ? Buffer.from(JSON.stringify(body), 'utf8') : null;

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'similarity-search-sdk-js/1.0.0',
      },
      timeout: timeoutMs,
    };

    if (bodyBuffer) {
      options.headers['Content-Length'] = bodyBuffer.length;
    }

    const req = transport.request(options, (res) => {
      let rawData = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { rawData += chunk; });
      res.on('end', () => {
        let parsed = null;
        try {
          parsed = rawData.length > 0 ? JSON.parse(rawData) : null;
        } catch (_) {
          parsed = { raw: rawData };
        }

        if (res.statusCode === 401 || res.statusCode === 403) {
          return reject(new SimilaritySearchAuthError(
            parsed?.detail || parsed?.message || 'Authentication failed.'
          ));
        }
        if (res.statusCode === 422) {
          return reject(new SimilaritySearchValidationError(
            parsed?.detail || parsed?.message || 'Validation error.'
          ));
        }
        if (res.statusCode === 429) {
          const retryAfter = res.headers['retry-after'] ? parseInt(res.headers['retry-after'], 10) : null;
          return reject(new SimilaritySearchRateLimitError(
            parsed?.detail || 'Rate limit exceeded.',
            retryAfter
          ));
        }
        if (res.statusCode >= 400) {
          return reject(new SimilaritySearchError(
            parsed?.detail || parsed?.message || `HTTP ${res.statusCode}`,
            res.statusCode,
            parsed
          ));
        }
        resolve({ statusCode: res.statusCode, body: parsed, headers: res.headers });
      });
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new SimilaritySearchError(`Request timed out after ${timeoutMs}ms`));
    });

    req.on('error', (err) => {
      reject(new SimilaritySearchError(`Network error: ${err.message}`));
    });

    if (bodyBuffer) {
      req.write(bodyBuffer);
    }
    req.end();
  });
}

function validateCorpusItems(items) {
  if (!Array.isArray(items)) {
    throw new SimilaritySearchValidationError('corpus.items must be an array.');
  }
  if (items.length === 0) {
    throw new SimilaritySearchValidationError('corpus.items must contain at least one item.');
  }
  if (items.length > MAX_CORPUS_ITEMS) {
    throw new SimilaritySearchValidationError(
      `corpus.items exceeds maximum of ${MAX_CORPUS_ITEMS} items (received ${items.length}).`
    );
  }
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item === null || item === undefined) {
      throw new SimilaritySearchValidationError(`corpus.items[${i}] is null or undefined.`);
    }
    if (typeof item !== 'object') {
      throw new SimilaritySearchValidationError(
        `corpus.items[${i}] must be an object with 'id' and 'content' fields.`
      );
    }
    if (!item.id || typeof item.id !== 'string' || item.id.trim().length === 0) {
      throw new SimilaritySearchValidationError(
        `corpus.items[${i}].id must be a non-empty string.`
      );
    }
    if (item.content === null || item.content === undefined) {
      throw new SimilaritySearchValidationError(
        `corpus.items[${i}].content is required (string, string[], or number[]).`
      );
    }
    const validContent =
      typeof item.content === 'string' ||
      (Array.isArray(item.content) &&
        item.content.every((v) => typeof v === 'string' || typeof v === 'number'));
    if (!validContent) {
      throw new SimilaritySearchValidationError(
        `corpus.items[${i}].content must be a string, string[], or number[].`
      );
    }
  }
}

function validateQuery(query) {
  if (query === null || query === undefined) {
    throw new SimilaritySearchValidationError('query is required and cannot be null or undefined.');
  }
  const validQuery =
    typeof query === 'string' ||
    (Array.isArray(query) &&
      query.every((v) => typeof v === 'string' || typeof v === 'number'));
  if (!validQuery) {
    throw new SimilaritySearchValidationError(
      'query must be a string, string[], or number[] matching the content type of corpus items.'
    );
  }
  if (typeof query === 'string' && query.trim().length === 0) {
    throw new SimilaritySearchValidationError('query string cannot be empty.');
  }
  if (typeof query === 'string' && query.length > MAX_QUERY_LENGTH) {
    throw new SimilaritySearchValidationError(
      `query string exceeds maximum length of ${MAX_QUERY_LENGTH} characters.`
    );
  }
  if (Array.isArray(query) && query.length === 0) {
    throw new SimilaritySearchValidationError('query array cannot be empty.');
  }
}

function validateTopK(topK) {
  if (!Number.isInteger(topK) || topK < MIN_TOP_K || topK > MAX_TOP_K) {
    throw new SimilaritySearchValidationError(
      `topK must be an integer between ${MIN_TOP_K} and ${MAX_TOP_K} (received ${topK}).`
    );
  }
}

function buildClient(options = {}) {
  if (options !== null && typeof options !== 'object') {
    throw new SimilaritySearchValidationError('options must be a plain object.');
  }

  const apiKey = resolveApiKey(options.apiKey);
  const baseUrl = (options.baseUrl || SIMILARITY_SEARCH_API_BASE_URL).replace(/\/$/, '');
  const timeoutMs = options.timeoutMs !== undefined ? options.timeoutMs : DEFAULT_TIMEOUT_MS;

  if (typeof timeoutMs !== 'number' || timeoutMs <= 0) {
    throw new SimilaritySearchValidationError('options.timeoutMs must be a positive number.');
  }

  async function ingestCorpus(corpusId, items, ingestOptions = {}) {
    if (!corpusId || typeof corpusId !== 'string' || corpusId.trim().length === 0) {
      throw new SimilaritySearchValidationError('corpusId must be a non-empty string.');
    }
    validateCorpusItems(items);

    const payload = {
      corpus_id: corpusId.trim(),
      items: items.map((item) => ({
        id: item.id.trim(),
        content: item.content,
        metadata: item.metadata || null,
      })),
      content_type: ingestOptions.contentType || 'text',
    };

    const response = await httpRequest(
      'POST',
      `${baseUrl}/corpus/ingest`,
      payload,
      apiKey,
      timeoutMs
    );

    return {
      corpusId: response.body.corpus_id,
      itemCount: response.body.item_count,
      entropyMarginal: response.body.entropy_marginal,
      alphaCoefficient: response.body.alpha_coefficient,
      vocabularySize: response.body.vocabulary_size,
      ingestedAt: response.body.ingested_at,
    };
  }

  async function searchByNmiHybridScore(corpusId, query, searchOptions = {}) {
    if (!corpusId || typeof corpusId !== 'string' || corpusId.trim().length === 0) {
      throw new SimilaritySearchValidationError('corpusId must be a non-empty string.');
    }
    validateQuery(query);

    const topK = searchOptions.topK !== undefined ? searchOptions.topK : DEFAULT_TOP_K;
    validateTopK(topK);

    const payload = {
      corpus_id: corpusId.trim(),
      query,
      top_k: topK,
      include_score_breakdown: searchOptions.includeScoreBreakdown === true,
      min_hybrid_score: searchOptions.minHybridScore !== undefined ? searchOptions.minHybridScore : null,
    };

    if (payload.min_hybrid_score !== null) {
      if (
        typeof payload.min_hybrid_score !== 'number' ||
        payload.min_hybrid_score < 0 ||
        payload.min_hybrid_score > 1
      ) {
        throw new SimilaritySearchValidationError('searchOptions.minHybridScore must be a number between 0 and 1.');
      }
    }

    const response = await httpRequest(
      'POST',
      `${baseUrl}/search/hybrid-nmi`,
      payload,
      apiKey,
      timeoutMs
    );

    return {
      corpusId: response.body.corpus_id,
      query: response.body.query,
      alphaCoefficient: response.body.alpha_coefficient,
      results: (response.body.results || []).map((r) => ({
        id: r.id,
        hybridScore: r.hybrid_score,
        cosineScore: r.cosine_score,
        nmiScore: r.nmi_score,
        metadata: r.metadata || null,
        scoreBreakdown: r.score_breakdown || null,
      })),
      computedAt: response.body.computed_at,
    };
  }

  async function computeNmiPairwise(itemA, itemB, pairwiseOptions = {}) {
    if (itemA === null || itemA === undefined) {
      throw new SimilaritySearchValidationError('itemA is required and cannot be null or undefined.');
    }
    if (itemB === null || itemB === undefined) {
      throw new SimilaritySearchValidationError('itemB is required and cannot be null or undefined.');
    }

    const validItem = (v) =>
      typeof v === 'string' ||
      (Array.isArray(v) && v.every((x) => typeof x === 'string' || typeof x === 'number'));

    if (!validItem(itemA)) {
      throw new SimilaritySearchValidationError('itemA must be a string, string[], or number[].');
    }
    if (!validItem(itemB)) {
      throw new SimilaritySearchValidationError('itemB must be a string, string[], or number[].');
    }
    if (typeof itemA === 'string' && itemA.trim().length === 0) {
      throw new SimilaritySearchValidationError('itemA string cannot be empty.');
    }
    if (typeof itemB === 'string' && itemB.trim().length === 0) {
      throw new SimilaritySearchValidationError('itemB string cannot be empty.');
    }

    const payload = {
      item_a: itemA,
      item_b: itemB,
      content_type: pairwiseOptions.contentType || 'text',
    };

    const response = await httpRequest(
      'POST',
      `${baseUrl}/nmi/pairwise`,
      payload,
      apiKey,
      timeoutMs
    );

    return {
      nmiScore: response.body.nmi_score,
      mutualInformation: response.body.mutual_information,
      entropyA: response.body.entropy_a,
      entropyB: response.body.entropy_b,
      jointEntropy: response.body.joint_entropy,
      computedAt: response.body.computed_at,
    };
  }

  async function describeCorpusEntropyProfile(corpusId) {
    if (!corpusId || typeof corpusId !== 'string' || corpusId.trim().length === 0) {
      throw new SimilaritySearchValidationError('corpusId must be a non-empty string.');
    }

    const url = new URL(`${baseUrl}/corpus/${encodeURIComponent(corpusId.trim())}/entropy-profile`);
    const response = await httpRequest('GET', url.toString(), null, apiKey, timeoutMs);

    return {
      corpusId: response.body.corpus_id,
      itemCount: response.body.item_count,
      vocabularySize: response.body.vocabulary_size,
      marginalEntropy: response.body.marginal_entropy,
      alphaCoefficient: response.body.alpha_coefficient,
      log2VocabularySize: response.body.log2_vocabulary_size,
      entropyPerToken: response.body.entropy_per_token,
      topTokensByInformation: response.body.top_tokens_by_information || [],
      createdAt: response.body.created_at,
      updatedAt: response.body.updated_at,
    };
  }

  async function mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        'data is required. Pass { corpusId, query } to search, { corpusId, items } to ingest, or { itemA, itemB } for pairwise NMI.'
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        'data must be a plain object. See SDK documentation for supported shapes.'
      );
    }

    const hasItems = Array.isArray(data.items) && data.items.length > 0;
    const hasQuery = data.query !== undefined && data.query !== null;
    const hasPairwise = data.itemA !== undefined || data.itemB !== undefined;

    if (hasPairwise) {
      return computeNmiPairwise(data.itemA, data.itemB, {
        contentType: data.contentType,
      });
    }

    if (hasItems && !hasQuery) {
      if (!data.corpusId) {
        throw new SimilaritySearchValidationError(
          'data.corpusId is required when ingesting corpus items.'
        );
      }
      return ingestCorpus(data.corpusId, data.items, {
        contentType: data.contentType,
      });
    }

    if (hasQuery) {
      if (!data.corpusId) {
        throw new SimilaritySearchValidationError(
          'data.corpusId is required when searching. For pairwise NMI without a corpus, use { itemA, itemB } instead.'
        );
      }
      return searchByNmiHybridScore(data.corpusId, data.query, {
        topK: data.topK,
        includeScoreBreakdown: data.includeScoreBreakdown,
        minHybridScore: data.minHybridScore,
      });
    }

    throw new SimilaritySearchValidationError(
      'data shape not recognized. Provide { corpusId, items } to ingest, { corpusId, query } to search, or { itemA, itemB } for pairwise NMI.'
    );
  }

  return {
    mainMethod,
    ingestCorpus,
    searchByNmiHybridScore,
    computeNmiPairwise,
    describeCorpusEntropyProfile,
  };
}

const defaultClient = buildClient({
  apiKey: SIMILARITY_SEARCH_API_KEY,
});

module.exports = {
  mainMethod: defaultClient.mainMethod,
  ingestCorpus: defaultClient.ingestCorpus,
  searchByNmiHybridScore: defaultClient.searchByNmiHybridScore,
  computeNmiPairwise: defaultClient.computeNmiPairwise,
  describeCorpusEntropyProfile: defaultClient.describeCorpusEntropyProfile,
  buildClient,
  SimilaritySearchError,
  SimilaritySearchAuthError,
  SimilaritySearchValidationError,
  SimilaritySearchRateLimitError,
};