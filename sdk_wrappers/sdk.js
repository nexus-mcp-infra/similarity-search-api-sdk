const axios = require('axios');

const SIMILARITY_SEARCH_BASE_URL = 'https://api.similaritysearch.io/v1';
const DEFAULT_TIMEOUT_MS = 30000;
const MAX_CORPUS_ITEMS = 10000;
const MAX_VECTOR_DIMENSIONS = 4096;
const MIN_CORPUS_ITEMS = 2;

class SimilaritySearchAuthError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SimilaritySearchAuthError';
  }
}

class SimilaritySearchValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SimilaritySearchValidationError';
  }
}

class SimilaritySearchApiError extends Error {
  constructor(message, statusCode, responseBody) {
    super(message);
    this.name = 'SimilaritySearchApiError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

class SimilaritySearchRateLimitError extends Error {
  constructor(retryAfterSeconds) {
    super(`Rate limit exceeded. Retry after ${retryAfterSeconds} seconds.`);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

function assertApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new SimilaritySearchAuthError(
      'API key is required. Pass it as the first argument to createClient() or set the SIMILARITY_SEARCH_API_KEY environment variable.'
    );
  }
  if (typeof apiKey !== 'string') {
    throw new SimilaritySearchAuthError(
      `API key must be a string, received ${typeof apiKey}.`
    );
  }
  if (apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError('API key must not be an empty string.');
  }
}

function assertVector(vector, label) {
  if (!Array.isArray(vector)) {
    throw new SimilaritySearchValidationError(
      `${label} must be an array of numbers, received ${typeof vector}.`
    );
  }
  if (vector.length === 0) {
    throw new SimilaritySearchValidationError(
      `${label} must not be an empty array.`
    );
  }
  if (vector.length > MAX_VECTOR_DIMENSIONS) {
    throw new SimilaritySearchValidationError(
      `${label} has ${vector.length} dimensions; maximum allowed is ${MAX_VECTOR_DIMENSIONS}.`
    );
  }
  for (let i = 0; i < vector.length; i++) {
    if (typeof vector[i] !== 'number' || !isFinite(vector[i])) {
      throw new SimilaritySearchValidationError(
        `${label}[${i}] is not a finite number (got ${vector[i]}).`
      );
    }
  }
}

function assertCorpus(corpus) {
  if (corpus === null || corpus === undefined) {
    throw new SimilaritySearchValidationError(
      'corpus is required and must be a non-empty array of items.'
    );
  }
  if (!Array.isArray(corpus)) {
    throw new SimilaritySearchValidationError(
      `corpus must be an array, received ${typeof corpus}.`
    );
  }
  if (corpus.length < MIN_CORPUS_ITEMS) {
    throw new SimilaritySearchValidationError(
      `corpus must contain at least ${MIN_CORPUS_ITEMS} items for NMI estimation to be meaningful; received ${corpus.length}.`
    );
  }
  if (corpus.length > MAX_CORPUS_ITEMS) {
    throw new SimilaritySearchValidationError(
      `corpus has ${corpus.length} items; maximum allowed per call is ${MAX_CORPUS_ITEMS}.`
    );
  }
}

function assertTopK(topK, corpusLength) {
  if (topK === undefined || topK === null) return;
  if (!Number.isInteger(topK) || topK < 1) {
    throw new SimilaritySearchValidationError(
      `top_k must be a positive integer, received ${topK}.`
    );
  }
  if (topK > corpusLength) {
    throw new SimilaritySearchValidationError(
      `top_k (${topK}) cannot exceed corpus size (${corpusLength}).`
    );
  }
}

function assertNmiWeight(nmiWeight) {
  if (nmiWeight === undefined || nmiWeight === null) return;
  if (typeof nmiWeight !== 'number' || !isFinite(nmiWeight)) {
    throw new SimilaritySearchValidationError(
      `nmi_weight_override must be a finite number, received ${nmiWeight}.`
    );
  }
  if (nmiWeight < 0 || nmiWeight > 1) {
    throw new SimilaritySearchValidationError(
      `nmi_weight_override must be between 0.0 and 1.0 inclusive, received ${nmiWeight}.`
    );
  }
}

function assertBandwidth(bandwidth) {
  if (bandwidth === undefined || bandwidth === null) return;
  if (typeof bandwidth !== 'number' || !isFinite(bandwidth) || bandwidth <= 0) {
    throw new SimilaritySearchValidationError(
      `kde_bandwidth must be a positive finite number, received ${bandwidth}.`
    );
  }
}

async function handleAxiosError(error) {
  if (error.response) {
    const { status, data, headers } = error.response;
    if (status === 401 || status === 403) {
      throw new SimilaritySearchAuthError(
        `Authentication failed (HTTP ${status}): ${data?.detail || data?.message || 'invalid or missing API key'}.`
      );
    }
    if (status === 429) {
      const retryAfter = parseInt(headers['retry-after'] || '60', 10);
      throw new SimilaritySearchRateLimitError(retryAfter);
    }
    throw new SimilaritySearchApiError(
      `API returned HTTP ${status}: ${data?.detail || data?.message || JSON.stringify(data)}`,
      status,
      data
    );
  }
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    throw new SimilaritySearchApiError(
      `Request timed out after ${DEFAULT_TIMEOUT_MS}ms. Consider reducing corpus size or vector dimensionality.`,
      null,
      null
    );
  }
  throw new SimilaritySearchApiError(
    `Network error: ${error.message}`,
    null,
    null
  );
}

function createClient(apiKey, options = {}) {
  if (apiKey === undefined || apiKey === null) {
    apiKey = process.env.SIMILARITY_SEARCH_API_KEY;
  }
  assertApiKey(apiKey);

  const baseURL = options.baseURL || SIMILARITY_SEARCH_BASE_URL;
  const timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;

  if (typeof baseURL !== 'string' || baseURL.trim().length === 0) {
    throw new SimilaritySearchValidationError('options.baseURL must be a non-empty string.');
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1) {
    throw new SimilaritySearchValidationError('options.timeoutMs must be a positive integer.');
  }

  const httpClient = axios.create({
    baseURL: baseURL.replace(/\/$/, ''),
    timeout: timeoutMs,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'similarity-search-sdk-js/1.0.0',
    },
  });

  async function rankCorpusByQuerySimilarity(payload) {
    if (payload === null || payload === undefined || typeof payload !== 'object' || Array.isArray(payload)) {
      throw new SimilaritySearchValidationError(
        'payload must be a non-null object with fields: corpus, query_vector, and optionally top_k, nmi_weight_override, kde_bandwidth.'
      );
    }

    const { corpus, query_vector, top_k, nmi_weight_override, kde_bandwidth } = payload;

    assertCorpus(corpus);

    corpus.forEach((item, idx) => {
      if (item === null || item === undefined || typeof item !== 'object' || Array.isArray(item)) {
        throw new SimilaritySearchValidationError(
          `corpus[${idx}] must be an object with fields: id (string) and vector (number[]).`
        );
      }
      if (typeof item.id !== 'string' || item.id.trim().length === 0) {
        throw new SimilaritySearchValidationError(
          `corpus[${idx}].id must be a non-empty string, received ${JSON.stringify(item.id)}.`
        );
      }
      assertVector(item.vector, `corpus[${idx}].vector`);
    });

    const corpusDim = corpus[0].vector.length;
    corpus.forEach((item, idx) => {
      if (item.vector.length !== corpusDim) {
        throw new SimilaritySearchValidationError(
          `All corpus vectors must have the same dimensionality. corpus[0].vector has ${corpusDim} dimensions but corpus[${idx}].vector has ${item.vector.length}.`
        );
      }
    });

    assertVector(query_vector, 'query_vector');

    if (query_vector.length !== corpusDim) {
      throw new SimilaritySearchValidationError(
        `query_vector has ${query_vector.length} dimensions but corpus vectors have ${corpusDim} dimensions. All vectors must share the same dimensionality.`
      );
    }

    assertTopK(top_k, corpus.length);
    assertNmiWeight(nmi_weight_override);
    assertBandwidth(kde_bandwidth);

    const body = { corpus, query_vector };
    if (top_k !== undefined && top_k !== null) body.top_k = top_k;
    if (nmi_weight_override !== undefined && nmi_weight_override !== null) body.nmi_weight_override = nmi_weight_override;
    if (kde_bandwidth !== undefined && kde_bandwidth !== null) body.kde_bandwidth = kde_bandwidth;

    try {
      const response = await httpClient.post('/search', body);
      return response.data;
    } catch (error) {
      if (
        error instanceof SimilaritySearchAuthError ||
        error instanceof SimilaritySearchValidationError ||
        error instanceof SimilaritySearchApiError ||
        error instanceof SimilaritySearchRateLimitError
      ) {
        throw error;
      }
      await handleAxiosError(error);
    }
  }

  async function computeNmiCosineScores(payload) {
    if (payload === null || payload === undefined || typeof payload !== 'object' || Array.isArray(payload)) {
      throw new SimilaritySearchValidationError(
        'payload must be a non-null object with fields: vector_pairs, and optionally kde_bandwidth.'
      );
    }

    const { vector_pairs, kde_bandwidth } = payload;

    if (vector_pairs === null || vector_pairs === undefined) {
      throw new SimilaritySearchValidationError(
        'vector_pairs is required and must be a non-empty array of {a, b} objects.'
      );
    }
    if (!Array.isArray(vector_pairs)) {
      throw new SimilaritySearchValidationError(
        `vector_pairs must be an array, received ${typeof vector_pairs}.`
      );
    }
    if (vector_pairs.length === 0) {
      throw new SimilaritySearchValidationError(
        'vector_pairs must not be an empty array.'
      );
    }
    if (vector_pairs.length > 500) {
      throw new SimilaritySearchValidationError(
        `vector_pairs has ${vector_pairs.length} pairs; maximum allowed per call is 500.`
      );
    }

    vector_pairs.forEach((pair, idx) => {
      if (pair === null || pair === undefined || typeof pair !== 'object' || Array.isArray(pair)) {
        throw new SimilaritySearchValidationError(
          `vector_pairs[${idx}] must be an object with fields: a (number[]) and b (number[]).`
        );
      }
      assertVector(pair.a, `vector_pairs[${idx}].a`);
      assertVector(pair.b, `vector_pairs[${idx}].b`);
      if (pair.a.length !== pair.b.length) {
        throw new SimilaritySearchValidationError(
          `vector_pairs[${idx}].a has ${pair.a.length} dimensions but vector_pairs[${idx}].b has ${pair.b.length}. Both vectors in a pair must have equal dimensionality.`
        );
      }
    });

    assertBandwidth(kde_bandwidth);

    const body = { vector_pairs };
    if (kde_bandwidth !== undefined && kde_bandwidth !== null) body.kde_bandwidth = kde_bandwidth;

    try {
      const response = await httpClient.post('/scores', body);
      return response.data;
    } catch (error) {
      if (
        error instanceof SimilaritySearchAuthError ||
        error instanceof SimilaritySearchValidationError ||
        error instanceof SimilaritySearchApiError ||
        error instanceof SimilaritySearchRateLimitError
      ) {
        throw error;
      }
      await handleAxiosError(error);
    }
  }

  async function resolveAdaptiveNmiWeight(payload) {
    if (payload === null || payload === undefined || typeof payload !== 'object' || Array.isArray(payload)) {
      throw new SimilaritySearchValidationError(
        'payload must be a non-null object with field: corpus (array of number[]).'
      );
    }

    const { corpus } = payload;

    if (corpus === null || corpus === undefined) {
      throw new SimilaritySearchValidationError(
        'corpus is required and must be an array of number[] vectors.'
      );
    }
    if (!Array.isArray(corpus)) {
      throw new SimilaritySearchValidationError(
        `corpus must be an array of number[] vectors, received ${typeof corpus}.`
      );
    }
    if (corpus.length < MIN_CORPUS_ITEMS) {
      throw new SimilaritySearchValidationError(
        `corpus must contain at least ${MIN_CORPUS_ITEMS} vectors for inter-item variance estimation; received ${corpus.length}.`
      );
    }
    if (corpus.length > MAX_CORPUS_ITEMS) {
      throw new SimilaritySearchValidationError(
        `corpus has ${corpus.length} vectors; maximum allowed is ${MAX_CORPUS_ITEMS}.`
      );
    }

    corpus.forEach((vec, idx) => {
      assertVector(vec, `corpus[${idx}]`);
    });

    const dim = corpus[0].length;
    corpus.forEach((vec, idx) => {
      if (vec.length !== dim) {
        throw new SimilaritySearchValidationError(
          `All corpus vectors must share the same dimensionality. corpus[0] has ${dim} dimensions but corpus[${idx}] has ${vec.length}.`
        );
      }
    });

    try {
      const response = await httpClient.post('/weight', { corpus });
      return response.data;
    } catch (error) {
      if (
        error instanceof SimilaritySearchAuthError ||
        error instanceof SimilaritySearchValidationError ||
        error instanceof SimilaritySearchApiError ||
        error instanceof SimilaritySearchRateLimitError
      ) {
        throw error;
      }
      await handleAxiosError(error);
    }
  }

  async function mainMethod(data) {
    if (data === null || data === undefined || typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        'data must be a non-null object with fields: corpus (array of {id, vector}), query_vector (number[]), and optionally top_k, nmi_weight_override, kde_bandwidth.'
      );
    }
    return rankCorpusByQuerySimilarity(data);
  }

  return {
    mainMethod,
    rankCorpusByQuerySimilarity,
    computeNmiCosineScores,
    resolveAdaptiveNmiWeight,
  };
}

module.exports = createClient;
module.exports.createClient = createClient;
module.exports.SimilaritySearchAuthError = SimilaritySearchAuthError;
module.exports.SimilaritySearchValidationError = SimilaritySearchValidationError;
module.exports.SimilaritySearchApiError = SimilaritySearchApiError;
module.exports.SimilaritySearchRateLimitError = SimilaritySearchRateLimitError;