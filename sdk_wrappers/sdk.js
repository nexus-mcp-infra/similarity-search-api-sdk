const axios = require('axios');
const crypto = require('crypto');

const SIMILARITY_API_BASE = 'https://api.nexus-similarity.io/v1';
const SUPPORTED_DOMAINS = ['text', 'image', 'tabular'];
const MAX_BATCH_PAIRS = 64;
const MAX_EMBEDDING_DIM = 4096;
const MIN_EMBEDDING_DIM = 2;
const DEFAULT_TIMEOUT_MS = 15000;
const DEFAULT_RETRIES = 3;
const RETRY_BACKOFF_BASE_MS = 200;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, requestId) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.requestId = requestId || null;
  }
}

class SimilarityAuthError extends SimilaritySearchError {
  constructor(requestId) {
    super('Invalid or missing API key for Similarity Search API', 401, requestId);
    this.name = 'SimilarityAuthError';
  }
}

class SimilarityRateLimitError extends SimilaritySearchError {
  constructor(retryAfterMs, requestId) {
    super(`Rate limit exceeded. Retry after ${retryAfterMs}ms`, 429, requestId);
    this.name = 'SimilarityRateLimitError';
    this.retryAfterMs = retryAfterMs || 1000;
  }
}

class SimilarityValidationError extends SimilaritySearchError {
  constructor(message) {
    super(message, 422, null);
    this.name = 'SimilarityValidationError';
  }
}

function validateEmbeddingVector(vec, label) {
  if (!Array.isArray(vec)) {
    throw new SimilarityValidationError(
      `${label} must be an array of numbers, got ${typeof vec}`
    );
  }
  if (vec.length < MIN_EMBEDDING_DIM || vec.length > MAX_EMBEDDING_DIM) {
    throw new SimilarityValidationError(
      `${label} dimension ${vec.length} out of range [${MIN_EMBEDDING_DIM}, ${MAX_EMBEDDING_DIM}]`
    );
  }
  for (let i = 0; i < vec.length; i++) {
    if (typeof vec[i] !== 'number' || !isFinite(vec[i])) {
      throw new SimilarityValidationError(
        `${label}[${i}] is not a finite number: ${vec[i]}`
      );
    }
  }
}

function validateDomain(domain) {
  if (!SUPPORTED_DOMAINS.includes(domain)) {
    throw new SimilarityValidationError(
      `domain must be one of [${SUPPORTED_DOMAINS.join(', ')}], got '${domain}'`
    );
  }
}

function sha256HexOfPair(vecA, vecB) {
  const payload = JSON.stringify([vecA, vecB]);
  return crypto.createHash('sha256').update(payload, 'utf8').digest('hex');
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildAxiosInstance(apiKey, timeoutMs) {
  return axios.create({
    baseURL: SIMILARITY_API_BASE,
    timeout: timeoutMs,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'X-Client': 'similarity-search-sdk-js/1.0.0',
    },
  });
}

async function executeWithRetry(fn, maxRetries, label) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (err instanceof SimilarityAuthError) {
        throw err;
      }
      if (err instanceof SimilarityRateLimitError) {
        if (attempt < maxRetries) {
          await sleep(err.retryAfterMs);
          lastError = err;
          continue;
        }
        throw err;
      }
      if (err instanceof SimilarityValidationError) {
        throw err;
      }
      if (attempt < maxRetries) {
        const backoff = RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt);
        await sleep(backoff);
        lastError = err;
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

function parseApiError(axiosError) {
  if (!axiosError.response) {
    throw new SimilaritySearchError(
      `Network error contacting Similarity Search API: ${axiosError.message}`,
      null,
      null
    );
  }
  const status = axiosError.response.status;
  const data = axiosError.response.data || {};
  const requestId = axiosError.response.headers['x-request-id'] || null;

  if (status === 401 || status === 403) {
    throw new SimilarityAuthError(requestId);
  }
  if (status === 429) {
    const retryAfter = parseInt(
      axiosError.response.headers['retry-after-ms'] ||
      axiosError.response.headers['retry-after'] * 1000 ||
      1000,
      10
    );
    throw new SimilarityRateLimitError(retryAfter, requestId);
  }
  if (status === 422) {
    throw new SimilarityValidationError(
      data.detail || 'Validation error from Similarity Search API'
    );
  }
  throw new SimilaritySearchError(
    data.detail || `Similarity Search API error: HTTP ${status}`,
    status,
    requestId
  );
}

class SimilaritySearchClient {
  constructor(options) {
    if (!options || typeof options !== 'object') {
      throw new SimilarityValidationError(
        'SimilaritySearchClient requires an options object with at least { apiKey }'
      );
    }
    if (!options.apiKey || typeof options.apiKey !== 'string' || options.apiKey.trim() === '') {
      throw new SimilarityValidationError(
        'options.apiKey must be a non-empty string — obtain one at https://api.nexus-similarity.io/keys'
      );
    }
    this._apiKey = options.apiKey.trim();
    this._timeoutMs = typeof options.timeoutMs === 'number' && options.timeoutMs > 0
      ? options.timeoutMs
      : DEFAULT_TIMEOUT_MS;
    this._retries = typeof options.retries === 'number' && options.retries >= 0
      ? Math.floor(options.retries)
      : DEFAULT_RETRIES;
    this._http = buildAxiosInstance(this._apiKey, this._timeoutMs);
  }

  async computeNmiCosineSimilarity(embeddingA, embeddingB, domain) {
    if (embeddingA == null) {
      throw new SimilarityValidationError('embeddingA is required and must not be null or undefined');
    }
    if (embeddingB == null) {
      throw new SimilarityValidationError('embeddingB is required and must not be null or undefined');
    }
    if (domain == null || typeof domain !== 'string') {
      throw new SimilarityValidationError("domain is required: 'text', 'image', or 'tabular'");
    }

    validateEmbeddingVector(embeddingA, 'embeddingA');
    validateEmbeddingVector(embeddingB, 'embeddingB');
    validateDomain(domain);

    if (embeddingA.length !== embeddingB.length) {
      throw new SimilarityValidationError(
        `embeddingA dimension (${embeddingA.length}) must equal embeddingB dimension (${embeddingB.length})`
      );
    }

    const pairHash = sha256HexOfPair(embeddingA, embeddingB);

    const body = {
      embedding_a: embeddingA,
      embedding_b: embeddingB,
      domain,
      pair_hash: pairHash,
    };

    return executeWithRetry(async () => {
      try {
        const response = await this._http.post('/similarity/nmi-cosine', body);
        return {
          compositeScore: response.data.composite_score,
          cosineComponent: response.data.cosine_component,
          nmiComponent: response.data.nmi_component,
          alpha: response.data.alpha,
          domain: response.data.domain,
          pairHash: response.data.pair_hash,
          latencyMs: response.data.latency_ms,
          requestId: response.headers['x-request-id'] || null,
        };
      } catch (err) {
        if (err.isAxiosError) parseApiError(err);
        throw err;
      }
    }, this._retries, 'computeNmiCosineSimilarity');
  }

  async rankEmbeddingsByNmiCosine(queryEmbedding, candidateEmbeddings, domain) {
    if (queryEmbedding == null) {
      throw new SimilarityValidationError('queryEmbedding is required and must not be null or undefined');
    }
    if (!Array.isArray(candidateEmbeddings)) {
      throw new SimilarityValidationError('candidateEmbeddings must be an array of embedding vectors');
    }
    if (candidateEmbeddings.length === 0) {
      throw new SimilarityValidationError('candidateEmbeddings must contain at least one vector');
    }
    if (candidateEmbeddings.length > MAX_BATCH_PAIRS) {
      throw new SimilarityValidationError(
        `candidateEmbeddings length ${candidateEmbeddings.length} exceeds max batch size ${MAX_BATCH_PAIRS}`
      );
    }
    if (domain == null || typeof domain !== 'string') {
      throw new SimilarityValidationError("domain is required: 'text', 'image', or 'tabular'");
    }

    validateEmbeddingVector(queryEmbedding, 'queryEmbedding');
    validateDomain(domain);

    for (let i = 0; i < candidateEmbeddings.length; i++) {
      validateEmbeddingVector(candidateEmbeddings[i], `candidateEmbeddings[${i}]`);
      if (candidateEmbeddings[i].length !== queryEmbedding.length) {
        throw new SimilarityValidationError(
          `candidateEmbeddings[${i}] dimension (${candidateEmbeddings[i].length}) ` +
          `must equal queryEmbedding dimension (${queryEmbedding.length})`
        );
      }
    }

    const body = {
      query_embedding: queryEmbedding,
      candidate_embeddings: candidateEmbeddings,
      domain,
    };

    return executeWithRetry(async () => {
      try {
        const response = await this._http.post('/similarity/rank', body);
        return {
          rankedIndices: response.data.ranked_indices,
          scores: response.data.scores,
          domain: response.data.domain,
          alpha: response.data.alpha,
          requestId: response.headers['x-request-id'] || null,
        };
      } catch (err) {
        if (err.isAxiosError) parseApiError(err);
        throw err;
      }
    }, this._retries, 'rankEmbeddingsByNmiCosine');
  }

  async batchComputeNmiCosineSimilarity(pairs, domain) {
    if (!Array.isArray(pairs)) {
      throw new SimilarityValidationError('pairs must be an array of { embeddingA, embeddingB } objects');
    }
    if (pairs.length === 0) {
      throw new SimilarityValidationError('pairs must contain at least one element');
    }
    if (pairs.length > MAX_BATCH_PAIRS) {
      throw new SimilarityValidationError(
        `pairs length ${pairs.length} exceeds max batch size ${MAX_BATCH_PAIRS}`
      );
    }
    if (domain == null || typeof domain !== 'string') {
      throw new SimilarityValidationError("domain is required: 'text', 'image', or 'tabular'");
    }
    validateDomain(domain);

    const serializedPairs = pairs.map((pair, idx) => {
      if (!pair || typeof pair !== 'object') {
        throw new SimilarityValidationError(`pairs[${idx}] must be an object with embeddingA and embeddingB`);
      }
      if (pair.embeddingA == null || pair.embeddingB == null) {
        throw new SimilarityValidationError(`pairs[${idx}] is missing embeddingA or embeddingB`);
      }
      validateEmbeddingVector(pair.embeddingA, `pairs[${idx}].embeddingA`);
      validateEmbeddingVector(pair.embeddingB, `pairs[${idx}].embeddingB`);
      if (pair.embeddingA.length !== pair.embeddingB.length) {
        throw new SimilarityValidationError(
          `pairs[${idx}]: embeddingA dimension (${pair.embeddingA.length}) ` +
          `must equal embeddingB dimension (${pair.embeddingB.length})`
        );
      }
      return {
        embedding_a: pair.embeddingA,
        embedding_b: pair.embeddingB,
        pair_hash: sha256HexOfPair(pair.embeddingA, pair.embeddingB),
      };
    });

    const body = { pairs: serializedPairs, domain };

    return executeWithRetry(async () => {
      try {
        const response = await this._http.post('/similarity/batch', body);
        return {
          results: response.data.results.map((r) => ({
            compositeScore: r.composite_score,
            cosineComponent: r.cosine_component,
            nmiComponent: r.nmi_component,
            pairHash: r.pair_hash,
          })),
          domain: response.data.domain,
          alpha: response.data.alpha,
          requestId: response.headers['x-request-id'] || null,
        };
      } catch (err) {
        if (err.isAxiosError) parseApiError(err);
        throw err;
      }
    }, this._retries, 'batchComputeNmiCosineSimilarity');
  }

  async fetchDomainCalibrationWeights(domain) {
    if (domain == null || typeof domain !== 'string') {
      throw new SimilarityValidationError("domain is required: 'text', 'image', or 'tabular'");
    }
    validateDomain(domain);

    return executeWithRetry(async () => {
      try {
        const response = await this._http.get(`/calibration/weights/${domain}`);
        return {
          domain: response.data.domain,
          alpha: response.data.alpha,
          beta: response.data.beta,
          calibratedAt: response.data.calibrated_at,
          trainingCallCount: response.data.training_call_count,
          ndcgImprovement: response.data.ndcg_improvement,
          requestId: response.headers['x-request-id'] || null,
        };
      } catch (err) {
        if (err.isAxiosError) parseApiError(err);
        throw err;
      }
    }, this._retries, 'fetchDomainCalibrationWeights');
  }
}

function createSimilaritySearchClient(options) {
  if (!options || typeof options !== 'object') {
    throw new SimilarityValidationError(
      'createSimilaritySearchClient requires an options object: { apiKey, [timeoutMs], [retries] }'
    );
  }
  return new SimilaritySearchClient(options);
}

const _defaultClient = { _instance: null };

async function mainMethod(data) {
  if (data == null || typeof data !== 'object') {
    throw new SimilarityValidationError(
      'mainMethod requires a data object: { apiKey, embeddingA, embeddingB, domain, [operation] }'
    );
  }

  const {
    apiKey,
    embeddingA,
    embeddingB,
    queryEmbedding,
    candidateEmbeddings,
    pairs,
    domain,
    operation,
    timeoutMs,
    retries,
  } = data;

  if (!apiKey || typeof apiKey !== 'string' || apiKey.trim() === '') {
    throw new SimilarityValidationError(
      'data.apiKey is required — obtain one at https://api.nexus-similarity.io/keys'
    );
  }

  const client = new SimilaritySearchClient({ apiKey, timeoutMs, retries });
  const op = operation || 'computeNmiCosineSimilarity';

  if (op === 'computeNmiCosineSimilarity') {
    return client.computeNmiCosineSimilarity(embeddingA, embeddingB, domain);
  }
  if (op === 'rankEmbeddingsByNmiCosine') {
    return client.rankEmbeddingsByNmiCosine(queryEmbedding, candidateEmbeddings, domain);
  }
  if (op === 'batchComputeNmiCosineSimilarity') {
    return client.batchComputeNmiCosineSimilarity(pairs, domain);
  }
  if (op === 'fetchDomainCalibrationWeights') {
    return client.fetchDomainCalibrationWeights(domain);
  }

  throw new SimilarityValidationError(
    `Unknown operation '${op}'. Valid operations: computeNmiCosineSimilarity, ` +
    `rankEmbeddingsByNmiCosine, batchComputeNmiCosineSimilarity, fetchDomainCalibrationWeights`
  );
}

module.exports = {
  mainMethod,
  createSimilaritySearchClient,
  SimilaritySearchClient,
  SimilaritySearchError,
  SimilarityAuthError,
  SimilarityRateLimitError,
  SimilarityValidationError,
  SUPPORTED_DOMAINS,
  MAX_BATCH_PAIRS,
  MAX_EMBEDDING_DIM,
};