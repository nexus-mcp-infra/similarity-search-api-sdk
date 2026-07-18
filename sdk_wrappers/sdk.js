const axios = require('axios');

const BASE_URL = process.env.SIMILARITY_API_URL || 'https://api.similarity-search.nexus/v1';
const DEFAULT_TIMEOUT_MS = 30000;
const MAX_ITEMS_PER_REQUEST = 500000;
const MIN_ALPHA = 0.0;
const MAX_ALPHA = 1.0;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, details) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.details = details || null;
  }
}

class AuthenticationError extends SimilaritySearchError {
  constructor() {
    super('Missing or invalid API key. Set SIMILARITY_API_KEY or pass apiKey in options.', 401);
    this.name = 'AuthenticationError';
  }
}

class ValidationError extends SimilaritySearchError {
  constructor(message) {
    super(message, 422);
    this.name = 'ValidationError';
  }
}

class RateLimitError extends SimilaritySearchError {
  constructor(retryAfterSeconds) {
    super('Rate limit exceeded.', 429);
    this.name = 'RateLimitError';
    this.retryAfterSeconds = retryAfterSeconds || null;
  }
}

function resolveApiKey(options) {
  const key = (options && options.apiKey) || process.env.SIMILARITY_API_KEY;
  if (!key || typeof key !== 'string' || key.trim().length === 0) {
    throw new AuthenticationError();
  }
  return key.trim();
}

function buildAxiosInstance(apiKey, timeoutMs) {
  return axios.create({
    baseURL: BASE_URL,
    timeout: timeoutMs || DEFAULT_TIMEOUT_MS,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client': 'similarity-search-sdk-js/1.0.0'
    }
  });
}

function wrapAxiosError(err) {
  if (!err.response) {
    return new SimilaritySearchError(
      `Network error: ${err.message}`,
      null,
      { originalMessage: err.message }
    );
  }
  const { status, data } = err.response;
  const detail = (data && data.detail) || (data && data.message) || JSON.stringify(data);
  if (status === 401 || status === 403) return new AuthenticationError();
  if (status === 429) {
    const retryAfter = err.response.headers['retry-after']
      ? parseInt(err.response.headers['retry-after'], 10)
      : null;
    return new RateLimitError(retryAfter);
  }
  if (status === 422) return new ValidationError(detail);
  return new SimilaritySearchError(detail, status, data);
}

function validateVector(vec, label) {
  if (!Array.isArray(vec)) {
    throw new ValidationError(`${label} must be an array of numbers.`);
  }
  if (vec.length === 0) {
    throw new ValidationError(`${label} must not be empty.`);
  }
  for (let i = 0; i < vec.length; i++) {
    if (typeof vec[i] !== 'number' || !isFinite(vec[i])) {
      throw new ValidationError(`${label}[${i}] must be a finite number, got: ${vec[i]}.`);
    }
  }
}

function validateCorpusItems(items) {
  if (!Array.isArray(items)) {
    throw new ValidationError('corpus must be an array of item objects.');
  }
  if (items.length === 0) {
    throw new ValidationError('corpus must contain at least one item.');
  }
  if (items.length > MAX_ITEMS_PER_REQUEST) {
    throw new ValidationError(
      `corpus exceeds maximum of ${MAX_ITEMS_PER_REQUEST} items per request. Received: ${items.length}.`
    );
  }
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (!item || typeof item !== 'object') {
      throw new ValidationError(`corpus[${i}] must be an object with 'id' and 'vector' fields.`);
    }
    if (item.id === undefined || item.id === null) {
      throw new ValidationError(`corpus[${i}].id is required and must not be null.`);
    }
    validateVector(item.vector, `corpus[${i}].vector`);
  }
}

function validateAlphaOverride(alpha) {
  if (alpha === undefined || alpha === null) return;
  if (typeof alpha !== 'number' || !isFinite(alpha)) {
    throw new ValidationError('alphaOverride must be a finite number between 0.0 and 1.0.');
  }
  if (alpha < MIN_ALPHA || alpha > MAX_ALPHA) {
    throw new ValidationError(
      `alphaOverride must be between ${MIN_ALPHA} and ${MAX_ALPHA}. Received: ${alpha}.`
    );
  }
}

function validateTopK(topK) {
  if (topK === undefined || topK === null) return;
  if (!Number.isInteger(topK) || topK < 1 || topK > 10000) {
    throw new ValidationError('topK must be an integer between 1 and 10000.');
  }
}

class SimilaritySearchClient {
  constructor(options) {
    this._apiKey = resolveApiKey(options);
    this._http = buildAxiosInstance(
      this._apiKey,
      options && options.timeoutMs
    );
  }

  async computeNmiCosineScore(queryVector, corpus, options) {
    if (queryVector === undefined || queryVector === null) {
      throw new ValidationError('queryVector is required and must not be null or undefined.');
    }
    validateVector(queryVector, 'queryVector');
    validateCorpusItems(corpus);

    const topK = (options && options.topK) !== undefined ? options.topK : 10;
    const alphaOverride = options && options.alphaOverride;
    validateTopK(topK);
    validateAlphaOverride(alphaOverride);

    const payload = {
      query_vector: queryVector,
      corpus: corpus.map(item => ({ id: item.id, vector: item.vector })),
      top_k: topK
    };
    if (alphaOverride !== undefined && alphaOverride !== null) {
      payload.alpha_override = alphaOverride;
    }

    try {
      const response = await this._http.post('/similarity/nmi-cosine', payload);
      return response.data;
    } catch (err) {
      throw wrapAxiosError(err);
    }
  }

  async batchComputeNmiCosineScores(queries, corpus, options) {
    if (!Array.isArray(queries)) {
      throw new ValidationError('queries must be an array of query objects.');
    }
    if (queries.length === 0) {
      throw new ValidationError('queries must contain at least one query.');
    }
    if (queries.length > 100) {
      throw new ValidationError(
        `Maximum 100 queries per batch request. Received: ${queries.length}.`
      );
    }
    for (let i = 0; i < queries.length; i++) {
      const q = queries[i];
      if (!q || typeof q !== 'object') {
        throw new ValidationError(`queries[${i}] must be an object with 'queryId' and 'vector' fields.`);
      }
      if (q.queryId === undefined || q.queryId === null) {
        throw new ValidationError(`queries[${i}].queryId is required.`);
      }
      validateVector(q.vector, `queries[${i}].vector`);
    }
    validateCorpusItems(corpus);

    const topK = (options && options.topK) !== undefined ? options.topK : 10;
    const alphaOverride = options && options.alphaOverride;
    validateTopK(topK);
    validateAlphaOverride(alphaOverride);

    const payload = {
      queries: queries.map(q => ({ query_id: q.queryId, vector: q.vector })),
      corpus: corpus.map(item => ({ id: item.id, vector: item.vector })),
      top_k: topK
    };
    if (alphaOverride !== undefined && alphaOverride !== null) {
      payload.alpha_override = alphaOverride;
    }

    try {
      const response = await this._http.post('/similarity/nmi-cosine/batch', payload);
      return response.data;
    } catch (err) {
      throw wrapAxiosError(err);
    }
  }

  async inspectEntropyCalibration(corpus) {
    validateCorpusItems(corpus);

    const payload = {
      corpus: corpus.map(item => ({ id: item.id, vector: item.vector }))
    };

    try {
      const response = await this._http.post('/similarity/entropy-calibration', payload);
      return response.data;
    } catch (err) {
      throw wrapAxiosError(err);
    }
  }

  async rankPairwiseNmiCosine(itemPairs, options) {
    if (!Array.isArray(itemPairs)) {
      throw new ValidationError('itemPairs must be an array of pair objects.');
    }
    if (itemPairs.length === 0) {
      throw new ValidationError('itemPairs must contain at least one pair.');
    }
    if (itemPairs.length > 50000) {
      throw new ValidationError(
        `Maximum 50000 pairs per request. Received: ${itemPairs.length}.`
      );
    }
    for (let i = 0; i < itemPairs.length; i++) {
      const pair = itemPairs[i];
      if (!pair || typeof pair !== 'object') {
        throw new ValidationError(`itemPairs[${i}] must be an object with 'a' and 'b' vector arrays.`);
      }
      validateVector(pair.a, `itemPairs[${i}].a`);
      validateVector(pair.b, `itemPairs[${i}].b`);
      if (pair.a.length !== pair.b.length) {
        throw new ValidationError(
          `itemPairs[${i}]: vectors 'a' and 'b' must have equal dimensionality. ` +
          `Got a.length=${pair.a.length}, b.length=${pair.b.length}.`
        );
      }
    }

    const alphaOverride = options && options.alphaOverride;
    validateAlphaOverride(alphaOverride);

    const payload = {
      pairs: itemPairs.map((pair, i) => ({
        pair_id: pair.pairId !== undefined ? pair.pairId : i,
        a: pair.a,
        b: pair.b
      }))
    };
    if (alphaOverride !== undefined && alphaOverride !== null) {
      payload.alpha_override = alphaOverride;
    }

    try {
      const response = await this._http.post('/similarity/pairwise-rank', payload);
      return response.data;
    } catch (err) {
      throw wrapAxiosError(err);
    }
  }
}

function createClient(options) {
  return new SimilaritySearchClient(options);
}

const _defaultClient = {
  _instance: null,
  _getInstance() {
    if (!this._instance) {
      this._instance = new SimilaritySearchClient({});
    }
    return this._instance;
  }
};

async function mainMethod(data) {
  if (data === undefined || data === null) {
    throw new ValidationError('data must be a non-null object with queryVector and corpus fields.');
  }
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new ValidationError('data must be a plain object with queryVector and corpus fields.');
  }
  const client = _defaultClient._getInstance();
  return client.computeNmiCosineScore(data.queryVector, data.corpus, data.options);
}

module.exports = {
  createClient,
  mainMethod,
  SimilaritySearchClient,
  SimilaritySearchError,
  AuthenticationError,
  ValidationError,
  RateLimitError
};