const axios = require('axios');

const SIMILARITY_SEARCH_BASE_URL = 'https://api.similarity-search.nexus/v1';
const DEFAULT_ALPHA = 0.6;
const DEFAULT_TOP_K = 10;
const DEFAULT_TIMEOUT_MS = 30000;

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
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'SimilaritySearchApiError';
    this.statusCode = statusCode;
    this.body = body;
  }
}

class SimilaritySearchRateLimitError extends Error {
  constructor(retryAfterSeconds) {
    super(`Rate limit exceeded. Retry after ${retryAfterSeconds}s.`);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

function assertApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new SimilaritySearchAuthError(
      'API key is required. Pass it via SIMILARITY_SEARCH_API_KEY env var or the apiKey option.'
    );
  }
  if (typeof apiKey !== 'string' || apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError(
      'API key must be a non-empty string.'
    );
  }
}

function assertEmbeddingVector(vec, label) {
  if (vec === null || vec === undefined) {
    throw new SimilaritySearchValidationError(`${label} must be a non-null array of numbers.`);
  }
  if (!Array.isArray(vec)) {
    throw new SimilaritySearchValidationError(
      `${label} must be an array of numbers, received ${typeof vec}.`
    );
  }
  if (vec.length === 0) {
    throw new SimilaritySearchValidationError(`${label} must not be empty.`);
  }
  for (let i = 0; i < vec.length; i++) {
    if (typeof vec[i] !== 'number' || !isFinite(vec[i])) {
      throw new SimilaritySearchValidationError(
        `${label}[${i}] is not a finite number: ${vec[i]}.`
      );
    }
  }
}

function assertCorpusEntries(corpus) {
  if (corpus === null || corpus === undefined) {
    throw new SimilaritySearchValidationError('corpus must be a non-null array.');
  }
  if (!Array.isArray(corpus)) {
    throw new SimilaritySearchValidationError(
      `corpus must be an array, received ${typeof corpus}.`
    );
  }
  if (corpus.length === 0) {
    throw new SimilaritySearchValidationError('corpus must contain at least one entry.');
  }
  if (corpus.length > 500000) {
    throw new SimilaritySearchValidationError(
      'corpus exceeds the 500K item limit for this API. Consider batching or a dedicated vector DB.'
    );
  }
  for (let i = 0; i < corpus.length; i++) {
    const entry = corpus[i];
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      throw new SimilaritySearchValidationError(
        `corpus[${i}] must be an object with an "embedding" array and optional "id"/"metadata".`
      );
    }
    if (!entry.embedding) {
      throw new SimilaritySearchValidationError(
        `corpus[${i}].embedding is required.`
      );
    }
    assertEmbeddingVector(entry.embedding, `corpus[${i}].embedding`);
  }
}

function assertAlpha(alpha) {
  if (typeof alpha !== 'number' || !isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new SimilaritySearchValidationError(
      `alpha must be a finite number in [0, 1], received: ${alpha}.`
    );
  }
}

function assertTopK(topK, corpusLength) {
  if (!Number.isInteger(topK) || topK < 1) {
    throw new SimilaritySearchValidationError(
      `topK must be a positive integer, received: ${topK}.`
    );
  }
  if (topK > corpusLength) {
    throw new SimilaritySearchValidationError(
      `topK (${topK}) cannot exceed corpus size (${corpusLength}).`
    );
  }
}

function assertPValueThreshold(pValueThreshold) {
  if (
    typeof pValueThreshold !== 'number' ||
    !isFinite(pValueThreshold) ||
    pValueThreshold <= 0 ||
    pValueThreshold >= 1
  ) {
    throw new SimilaritySearchValidationError(
      `pValueThreshold must be a finite number in (0, 1), received: ${pValueThreshold}.`
    );
  }
}

function buildHttpClient(apiKey, timeoutMs) {
  return axios.create({
    baseURL: SIMILARITY_SEARCH_BASE_URL,
    timeout: timeoutMs,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'X-Client': 'similarity-search-sdk-js/1.0.0',
    },
  });
}

async function executeRequest(httpClient, method, path, payload) {
  try {
    const response = await httpClient[method](path, payload);
    return response.data;
  } catch (err) {
    if (err.response) {
      const status = err.response.status;
      const body = err.response.data;
      if (status === 401 || status === 403) {
        throw new SimilaritySearchAuthError(
          `Authentication failed (HTTP ${status}): ${JSON.stringify(body)}`
        );
      }
      if (status === 429) {
        const retryAfter = parseInt(
          (err.response.headers && err.response.headers['retry-after']) || '60',
          10
        );
        throw new SimilaritySearchRateLimitError(retryAfter);
      }
      throw new SimilaritySearchApiError(
        `API returned HTTP ${status}: ${JSON.stringify(body)}`,
        status,
        body
      );
    }
    if (err.code === 'ECONNABORTED') {
      throw new SimilaritySearchApiError(
        `Request timed out after ${err.config && err.config.timeout}ms.`,
        null,
        null
      );
    }
    throw new SimilaritySearchApiError(
      `Network error: ${err.message}`,
      null,
      null
    );
  }
}

class SimilaritySearchClient {
  constructor(options = {}) {
    const apiKey =
      options.apiKey !== undefined
        ? options.apiKey
        : process.env.SIMILARITY_SEARCH_API_KEY;

    assertApiKey(apiKey);

    this._apiKey = apiKey;
    this._timeoutMs =
      typeof options.timeoutMs === 'number' && options.timeoutMs > 0
        ? options.timeoutMs
        : DEFAULT_TIMEOUT_MS;

    this._http = buildHttpClient(this._apiKey, this._timeoutMs);
  }

  async compositeNmiCosineSimilarity(queryEmbedding, candidateEmbedding, options = {}) {
    assertEmbeddingVector(queryEmbedding, 'queryEmbedding');
    assertEmbeddingVector(candidateEmbedding, 'candidateEmbedding');

    if (queryEmbedding.length !== candidateEmbedding.length) {
      throw new SimilaritySearchValidationError(
        `queryEmbedding dimension (${queryEmbedding.length}) must match candidateEmbedding dimension (${candidateEmbedding.length}).`
      );
    }

    const alpha =
      options.alpha !== undefined ? options.alpha : DEFAULT_ALPHA;
    assertAlpha(alpha);

    const corpusSize =
      options.corpusSize !== undefined ? options.corpusSize : null;
    if (corpusSize !== null) {
      if (!Number.isInteger(corpusSize) || corpusSize < 1) {
        throw new SimilaritySearchValidationError(
          `corpusSize must be a positive integer when provided, received: ${corpusSize}.`
        );
      }
    }

    const payload = {
      query_embedding: queryEmbedding,
      candidate_embedding: candidateEmbedding,
      alpha,
    };
    if (corpusSize !== null) {
      payload.corpus_size = corpusSize;
    }

    return executeRequest(this._http, 'post', '/composite-similarity', payload);
  }

  async rankCorpusByNmiCosineScore(queryEmbedding, corpus, options = {}) {
    assertEmbeddingVector(queryEmbedding, 'queryEmbedding');
    assertCorpusEntries(corpus);

    const queryDim = queryEmbedding.length;
    for (let i = 0; i < corpus.length; i++) {
      if (corpus[i].embedding.length !== queryDim) {
        throw new SimilaritySearchValidationError(
          `corpus[${i}].embedding dimension (${corpus[i].embedding.length}) must match queryEmbedding dimension (${queryDim}).`
        );
      }
    }

    const topK =
      options.topK !== undefined ? options.topK : Math.min(DEFAULT_TOP_K, corpus.length);
    assertTopK(topK, corpus.length);

    const alpha =
      options.alpha !== undefined ? options.alpha : DEFAULT_ALPHA;
    assertAlpha(alpha);

    const pValueThreshold =
      options.pValueThreshold !== undefined ? options.pValueThreshold : 0.05;
    assertPValueThreshold(pValueThreshold);

    const payload = {
      query_embedding: queryEmbedding,
      corpus: corpus.map((entry, idx) => ({
        id: entry.id !== undefined ? String(entry.id) : String(idx),
        embedding: entry.embedding,
        metadata: entry.metadata !== undefined ? entry.metadata : null,
      })),
      top_k: topK,
      alpha,
      p_value_threshold: pValueThreshold,
    };

    return executeRequest(this._http, 'post', '/rank-corpus', payload);
  }

  async batchCompositeNmiCosineSimilarity(pairs, options = {}) {
    if (pairs === null || pairs === undefined) {
      throw new SimilaritySearchValidationError('pairs must be a non-null array.');
    }
    if (!Array.isArray(pairs)) {
      throw new SimilaritySearchValidationError(
        `pairs must be an array, received ${typeof pairs}.`
      );
    }
    if (pairs.length === 0) {
      throw new SimilaritySearchValidationError('pairs must contain at least one entry.');
    }
    if (pairs.length > 1000) {
      throw new SimilaritySearchValidationError(
        'pairs exceeds the batch limit of 1000. Split into multiple batch calls.'
      );
    }

    for (let i = 0; i < pairs.length; i++) {
      const pair = pairs[i];
      if (!pair || typeof pair !== 'object' || Array.isArray(pair)) {
        throw new SimilaritySearchValidationError(
          `pairs[${i}] must be an object with query_embedding and candidate_embedding.`
        );
      }
      assertEmbeddingVector(pair.query_embedding, `pairs[${i}].query_embedding`);
      assertEmbeddingVector(pair.candidate_embedding, `pairs[${i}].candidate_embedding`);
      if (pair.query_embedding.length !== pair.candidate_embedding.length) {
        throw new SimilaritySearchValidationError(
          `pairs[${i}]: query_embedding dimension (${pair.query_embedding.length}) must match candidate_embedding dimension (${pair.candidate_embedding.length}).`
        );
      }
    }

    const alpha =
      options.alpha !== undefined ? options.alpha : DEFAULT_ALPHA;
    assertAlpha(alpha);

    const corpusSize =
      options.corpusSize !== undefined ? options.corpusSize : null;
    if (corpusSize !== null) {
      if (!Number.isInteger(corpusSize) || corpusSize < 1) {
        throw new SimilaritySearchValidationError(
          `corpusSize must be a positive integer when provided, received: ${corpusSize}.`
        );
      }
    }

    const payload = {
      pairs: pairs.map((pair, idx) => ({
        id: pair.id !== undefined ? String(pair.id) : String(idx),
        query_embedding: pair.query_embedding,
        candidate_embedding: pair.candidate_embedding,
      })),
      alpha,
    };
    if (corpusSize !== null) {
      payload.corpus_size = corpusSize;
    }

    return executeRequest(this._http, 'post', '/batch-composite-similarity', payload);
  }

  async introspectEmbeddingActivationHistogram(embedding, options = {}) {
    assertEmbeddingVector(embedding, 'embedding');

    if (embedding.length < 4) {
      throw new SimilaritySearchValidationError(
        `embedding must have at least 4 dimensions to produce a meaningful activation histogram; received ${embedding.length}.`
      );
    }

    const binStrategy =
      options.binStrategy !== undefined ? options.binStrategy : 'freedman-diaconis';
    const allowedBinStrategies = ['freedman-diaconis', 'sturges', 'scott'];
    if (!allowedBinStrategies.includes(binStrategy)) {
      throw new SimilaritySearchValidationError(
        `binStrategy must be one of: ${allowedBinStrategies.join(', ')}. Received: ${binStrategy}.`
      );
    }

    const payload = {
      embedding,
      bin_strategy: binStrategy,
    };

    return executeRequest(this._http, 'post', '/embedding-histogram', payload);
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        'data must be a non-null object. Expected shape: { queryEmbedding, corpus, options }.'
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        `data must be a plain object, received ${typeof data}.`
      );
    }

    const { queryEmbedding, corpus, options = {} } = data;

    if (!queryEmbedding && !corpus) {
      throw new SimilaritySearchValidationError(
        'data must include at least queryEmbedding and corpus for ranking, or use compositeNmiCosineSimilarity directly for a single pair.'
      );
    }

    return this.rankCorpusByNmiCosineScore(queryEmbedding, corpus, options);
  }
}

function createSimilaritySearchClient(options = {}) {
  return new SimilaritySearchClient(options);
}

const _defaultClient = (() => {
  if (process.env.SIMILARITY_SEARCH_API_KEY) {
    try {
      return new SimilaritySearchClient();
    } catch (_) {
      return null;
    }
  }
  return null;
})();

function _getDefaultClientOrThrow() {
  if (!_defaultClient) {
    throw new SimilaritySearchAuthError(
      'No default client available. Set SIMILARITY_SEARCH_API_KEY or use createSimilaritySearchClient({ apiKey }).'
    );
  }
  return _defaultClient;
}

module.exports = {
  createSimilaritySearchClient,
  SimilaritySearchClient,
  SimilaritySearchAuthError,
  SimilaritySearchValidationError,
  SimilaritySearchApiError,
  SimilaritySearchRateLimitError,

  compositeNmiCosineSimilarity: (queryEmbedding, candidateEmbedding, options) =>
    _getDefaultClientOrThrow().compositeNmiCosineSimilarity(queryEmbedding, candidateEmbedding, options),

  rankCorpusByNmiCosineScore: (queryEmbedding, corpus, options) =>
    _getDefaultClientOrThrow().rankCorpusByNmiCosineScore(queryEmbedding, corpus, options),

  batchCompositeNmiCosineSimilarity: (pairs, options) =>
    _getDefaultClientOrThrow().batchCompositeNmiCosineSimilarity(pairs, options),

  introspectEmbeddingActivationHistogram: (embedding, options) =>
    _getDefaultClientOrThrow().introspectEmbeddingActivationHistogram(embedding, options),

  mainMethod: (data) =>
    _getDefaultClientOrThrow().mainMethod(data),
};