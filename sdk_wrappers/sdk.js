
```javascript
'use strict';

const https = require('https');
const http = require('http');

const SIMILARITY_SEARCH_API_BASE_URL = process.env.SIMILARITY_SEARCH_API_URL || 'https://api.similarity-search.io';
const SIMILARITY_SEARCH_API_VERSION = 'v1';
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_BOOTSTRAP_ITERATIONS = 500;
const DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05;
const MAX_EMBEDDING_DIMENSIONS = 4096;
const MAX_VECTORS_PER_CALL = 1000;
const MAX_QUERY_VECTORS = 10;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, responseBody) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.responseBody = responseBody || null;
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
  constructor(retryAfterSeconds) {
    super(`Rate limit exceeded. Retry after ${retryAfterSeconds} seconds.`, 429, null);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

function validateApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new SimilaritySearchAuthError(
      'API key is required. Provide it via constructor options or SIMILARITY_SEARCH_API_KEY environment variable.'
    );
  }
  if (typeof apiKey !== 'string') {
    throw new SimilaritySearchAuthError(
      `API key must be a string, received ${typeof apiKey}.`
    );
  }
  if (apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError(
      'API key must not be empty.'
    );
  }
}

function validateEmbeddingVector(vector, label) {
  if (!Array.isArray(vector)) {
    throw new SimilaritySearchValidationError(
      `${label} must be an array of numbers, received ${typeof vector}.`
    );
  }
  if (vector.length === 0) {
    throw new SimilaritySearchValidationError(
      `${label} must not be empty.`
    );
  }
  if (vector.length > MAX_EMBEDDING_DIMENSIONS) {
    throw new SimilaritySearchValidationError(
      `${label} exceeds maximum allowed dimensions (${MAX_EMBEDDING_DIMENSIONS}). Received ${vector.length}.`
    );
  }
  for (let i = 0; i < vector.length; i++) {
    const val = vector[i];
    if (typeof val !== 'number' || !isFinite(val)) {
      throw new SimilaritySearchValidationError(
        `${label}[${i}] must be a finite number, received ${val}.`
      );
    }
  }
}

function validateVectorCorpus(corpus) {
  if (!Array.isArray(corpus)) {
    throw new SimilaritySearchValidationError(
      `corpus must be an array of embedding vectors, received ${typeof corpus}.`
    );
  }
  if (corpus.length === 0) {
    throw new SimilaritySearchValidationError(
      'corpus must contain at least one embedding vector.'
    );
  }
  if (corpus.length > MAX_VECTORS_PER_CALL) {
    throw new SimilaritySearchValidationError(
      `corpus exceeds maximum vectors per call (${MAX_VECTORS_PER_CALL}). Received ${corpus.length}.`
    );
  }
  const expectedDim = corpus[0].length;
  for (let i = 0; i < corpus.length; i++) {
    validateEmbeddingVector(corpus[i], `corpus[${i}]`);
    if (corpus[i].length !== expectedDim) {
      throw new SimilaritySearchValidationError(
        `All corpus vectors must have the same dimensionality. corpus[0] has ${expectedDim} dims, corpus[${i}] has ${corpus[i].length} dims.`
      );
    }
  }
}

function validateQueryVectors(queries) {
  if (!Array.isArray(queries)) {
    throw new SimilaritySearchValidationError(
      `queries must be an array of embedding vectors, received ${typeof queries}.`
    );
  }
  if (queries.length === 0) {
    throw new SimilaritySearchValidationError(
      'queries must contain at least one embedding vector.'
    );
  }
  if (queries.length > MAX_QUERY_VECTORS) {
    throw new SimilaritySearchValidationError(
      `queries exceeds maximum per call (${MAX_QUERY_VECTORS}). Received ${queries.length}.`
    );
  }
}

function validateTopK(topK, corpusSize) {
  if (topK === null || topK === undefined) return;
  if (typeof topK !== 'number' || !Number.isInteger(topK)) {
    throw new SimilaritySearchValidationError(
      `topK must be an integer, received ${typeof topK} (${topK}).`
    );
  }
  if (topK < 1) {
    throw new SimilaritySearchValidationError(
      `topK must be >= 1, received ${topK}.`
    );
  }
  if (corpusSize !== undefined && topK > corpusSize) {
    throw new SimilaritySearchValidationError(
      `topK (${topK}) cannot exceed corpus size (${corpusSize}).`
    );
  }
}

function validateBootstrapIterations(iterations) {
  if (iterations === null || iterations === undefined) return;
  if (typeof iterations !== 'number' || !Number.isInteger(iterations)) {
    throw new SimilaritySearchValidationError(
      `bootstrapIterations must be an integer, received ${typeof iterations}.`
    );
  }
  if (iterations < 100 || iterations > 10000) {
    throw new SimilaritySearchValidationError(
      `bootstrapIterations must be between 100 and 10000, received ${iterations}.`
    );
  }
}

function validateSignificanceThreshold(threshold) {
  if (threshold === null || threshold === undefined) return;
  if (typeof threshold !== 'number' || !isFinite(threshold)) {
    throw new SimilaritySearchValidationError(
      `significanceThreshold must be a finite number, received ${threshold}.`
    );
  }
  if (threshold <= 0 || threshold >= 1) {
    throw new SimilaritySearchValidationError(
      `significanceThreshold must be between 0 (exclusive) and 1 (exclusive), received ${threshold}.`
    );
  }
}

function httpRequest(options, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    const protocol = options.protocol === 'http:' ? http : https;
    const payload = body ? JSON.stringify(body) : null;

    const reqOptions = {
      hostname: options.hostname,
      port: options.port,
      path: options.path,
      method: options.method || 'GET',
      headers: Object.assign({}, options.headers),
    };

    if (payload) {
      reqOptions.headers['Content-Type'] = 'application/json';
      reqOptions.headers['Content-Length'] = Buffer.byteLength(payload);
    }

    const req = protocol.request(reqOptions, (res) => {
      let rawData = '';
      res.on('data', (chunk) => { rawData += chunk; });
      res.on('end', () => {
        resolve({ statusCode: res.statusCode, headers: res.headers, body: rawData });
      });
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new SimilaritySearchError(
        `Request timed out after ${timeoutMs}ms.`, null, null
      ));
    });

    req.on('error', (err) => {
      reject(new SimilaritySearchError(
        `Network error: ${err.message}`, null, null
      ));
    });

    if (payload) {
      req.write(payload);
    }
    req.end();
  });
}

function parseBaseUrl(rawUrl) {
  const trimmed = rawUrl.replace(/\/$/, '');
  let protocol = 'https:';
  let rest = trimmed;
  if (trimmed.startsWith('https://')) {
    protocol = 'https:';
    rest = trimmed.slice(8);
  } else if (trimmed.startsWith('http://')) {
    protocol = 'http:';
    rest = trimmed.slice(7);
  }
  const colonIdx = rest.indexOf(':');
  const slashIdx = rest.indexOf('/');
  let hostname, port, basePath;
  if (colonIdx !== -1 && (slashIdx === -1 || colonIdx < slashIdx)) {
    hostname = rest.slice(0, colonIdx);
    const afterColon = rest.slice(colonIdx + 1);
    const portEnd = afterColon.indexOf('/');
    if (portEnd === -1) {
      port = parseInt(afterColon, 10);
      basePath = '';
    } else {
      port = parseInt(afterColon.slice(0, portEnd), 10);
      basePath = afterColon.slice(portEnd);
    }
  } else if (slashIdx !== -1) {
    hostname = rest.slice(0, slashIdx);
    port = protocol === 'https:' ? 443 : 80;
    basePath = rest.slice(slashIdx);
  } else {
    hostname = rest;
    port = protocol === 'https:' ? 443 : 80;
    basePath = '';
  }
  return { protocol, hostname, port, basePath };
}

async function dispatchApiRequest(parsedBase, apiKey, method, endpointPath, body, timeoutMs) {
  const fullPath = `${parsedBase.basePath}/${SIMILARITY_SEARCH_API_VERSION}${endpointPath}`;
  const options = {
    protocol: parsedBase.protocol,
    hostname: parsedBase.hostname,
    port: parsedBase.port,
    path: fullPath,
    method,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Accept': 'application/json',
      'User-Agent': `similarity-search-sdk-js/1.0.0`,
    },
  };

  const response = await httpRequest(options, body, timeoutMs);

  if (response.statusCode === 401 || response.statusCode === 403) {
    throw new SimilaritySearchAuthError(
      'Authentication failed. Check your API key.'
    );
  }

  if (response.statusCode === 429) {
    const retryAfter = response.headers['retry-after']
      ? parseInt(response.headers['retry-after'], 10)
      : 60;
    throw new SimilaritySearchRateLimitError(retryAfter);
  }

  let parsed;
  try {
    parsed = JSON.parse(response.body);
  } catch (e) {
    throw new SimilaritySearchError(
      `Failed to parse API response as JSON. Status: ${response.statusCode}. Body: ${response.body.slice(0, 200)}`,
      response.statusCode,
      response.body
    );
  }

  if (response.statusCode === 422) {
    const detail = parsed.detail
      ? (Array.isArray(parsed.detail)
        ? parsed.detail.map(d => `${d.loc ? d.loc.join('.') : ''}: ${d.msg}`).join('; ')
        : String(parsed.detail))
      : 'Validation error.';
    throw new SimilaritySearchValidationError(detail);
  }

  if (response.statusCode >= 400) {
    const msg = parsed.message || parsed.detail || parsed.error || `API error ${response.statusCode}`;
    throw new SimilaritySearchError(String(msg), response.statusCode, parsed);
  }

  return parsed;
}

class SimilaritySearchClient {
  constructor(options) {
    if (options === null || options === undefined) {
      options = {};
    }
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError(
        'SimilaritySearchClient constructor expects an options object or no argument.'
      );
    }

    const apiKey = options.apiKey || process.env.SIMILARITY_SEARCH_API_KEY || null;
    validateApiKey(apiKey);

    this._apiKey = apiKey;
    this._timeoutMs = typeof options.timeoutMs === 'number' && options.timeoutMs > 0
      ? options.timeoutMs
      : DEFAULT_TIMEOUT_MS;

    const baseUrl = options.baseUrl || SIMILARITY_SEARCH_API_BASE_URL;
    this._parsedBase = parseBaseUrl(baseUrl);
  }

  async rankByCosineNmi(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'rankByCosineNmi requires a params object with at least { queryVector, corpus }.'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        `rankByCosineNmi params must be an object, received ${typeof params}.`
      );
    }

    const { queryVector, corpus, topK, bootstrapIterations, significanceThreshold, metadata } = params;

    validateEmbeddingVector(queryVector, 'queryVector');
    validateVectorCorpus(corpus);

    for (let i = 0; i < corpus.length; i++) {
      if (corpus[i].length !== queryVector.length) {
        throw new SimilaritySearchValidationError(
          `queryVector has ${queryVector.length} dims but corpus[${i}] has ${corpus[i].length} dims. All vectors must share dimensionality.`
        );
      }
    }

    validateTopK(topK, corpus.length);
    validateBootstrapIterations(bootstrapIterations);
    validateSignificanceThreshold(significanceThreshold);

    if (metadata !== null && metadata !== undefined && typeof metadata !== 'object') {
      throw new SimilaritySearchValidationError(
        `metadata must be an object or omitted, received ${typeof metadata}.`
      );
    }

    const body = {
      query_vector: queryVector,
      corpus,
      top_k: topK !== undefined ? topK : null,
      bootstrap_iterations: bootstrapIterations !== undefined ? bootstrapIterations : DEFAULT_BOOTSTRAP_ITERATIONS,
      significance_threshold: significanceThreshold !== undefined ? significanceThreshold : DEFAULT_SIGNIFICANCE_THRESHOLD,
    };
    if (metadata) body.metadata = metadata;

    return dispatchApiRequest(
      this._parsedBase, this._apiKey, 'POST', '/rank/cosine-nmi', body, this._timeoutMs
    );
  }

  async rankBatchQueriesByNmi(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'rankBatchQueriesByNmi requires a params object with at least { queryVectors, corpus }.'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        `rankBatchQueriesByNmi params must be an object, received ${typeof params}.`
      );
    }

    const { queryVectors, corpus, topK, bootstrapIterations, significanceThreshold } = params;

    validateQueryVectors(queryVectors);
    validateVectorCorpus(corpus);

    const expectedDim = queryVectors[0].length;
    for (let i = 0; i < queryVectors.length; i++) {
      validateEmbeddingVector(queryVectors[i], `queryVectors[${i}]`);
      if (queryVectors[i].length !== expectedDim) {
        throw new SimilaritySearchValidationError(
          `All query vectors must share dimensionality. queryVectors[0] has ${expectedDim} dims, queryVectors[${i}] has ${queryVectors[i].length} dims.`
        );
      }
    }

    for (let i = 0; i < corpus.length; i++) {
      if (corpus[i].length !== expectedDim) {
        throw new SimilaritySearchValidationError(
          `Query vectors have ${expectedDim} dims but corpus[${i}] has ${corpus[i].length} dims.`
        );
      }
    }

    validateTopK(topK, corpus.length);
    validateBootstrapIterations(bootstrapIterations);
    validateSignificanceThreshold(significanceThreshold);

    const body = {
      query_vectors: queryVectors,
      corpus,
      top_k: topK !== undefined ? topK : null,
      bootstrap_iterations: bootstrapIterations !== undefined ? bootstrapIterations : DEFAULT_BOOTSTRAP_ITERATIONS,
      significance_threshold: significanceThreshold !== undefined ? significanceThreshold : DEFAULT_SIGNIFICANCE_THRESHOLD,
    };

    return dispatchApiRequest(
      this._parsedBase, this._apiKey, 'POST', '/rank/batch-cosine-nmi', body, this._timeoutMs
    );
  }

  async estimateJointEntropyProfile(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'estimateJointEntropyProfile requires a params object with { vectorA, vectorB }.'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        `estimateJointEntropyProfile params must be an object, received ${typeof params}.`
      );
    }

    const { vectorA, vectorB, binStrategy } = params;

    validateEmbeddingVector(vectorA, 'vectorA');
    validateEmbeddingVector(vectorB, 'vectorB');

    if (vectorA.length !== vectorB.length) {
      throw new SimilaritySearchValidationError(
        `vectorA (${vectorA.length} dims) and vectorB (${vectorB.length} dims) must have the same dimensionality.`
      );
    }

    const allowedBinStrategies = ['freedman-diaconis', 'scott', 'fixed-16', 'fixed-32'];
    if (binStrategy !== null && binStrategy !== undefined) {
      if (!allowedBinStrategies.includes(binStrategy)) {
        throw new SimilaritySearchValidationError(
          `binStrategy must be one of ${allowedBinStrategies.join(', ')}, received '${binStrategy}'.`
        );
      }
    }

    const body = {
      vector_a: vectorA,
      vector_b: vectorB,
      bin_strategy: binStrategy || 'freedman-diaconis',
    };

    return dispatchApiRequest(
      this._parsedBase, this._apiKey, 'POST', '/entropy/joint-profile', body, this._timeoutMs
    );
  }

  async fetchNmiSignificanceBootstrap(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'fetchNmiSignificanceBootstrap requires a params object with { vectorA, vectorB, bootstrapIterations }.'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        `fetchNmiSignificanceBootstrap params must be an object, received ${typeof params}.`
      );
    }

    const { vectorA, vectorB, bootstrapIterations, confidenceLevel } = params;

    validateEmbeddingVector(vectorA, 'vectorA');
    validateEmbeddingVector(vectorB, 'vectorB');

    if (vectorA.length !== vectorB.length) {
      throw new SimilaritySearchValidationError(
        `vectorA (${vectorA.length} dims) and vectorB (${vectorB.length} dims) must have the same dimensionality.`
      );
    }

    validateBootstrapIterations(bootstrapIterations);

    if (confidenceLevel !== null && confidenceLevel !== undefined) {
      if (typeof confidenceLevel !== 'number' || !isFinite(confidenceLevel)) {
        throw new SimilaritySearchValidationError(
          `confidenceLevel must be a finite number, received ${confidenceLevel}.`
        );
      }
      if (confidenceLevel <= 0 || confidenceLevel >= 1) {
        throw new SimilaritySearchValidationError(
          `confidenceLevel must be between 0 (exclusive) and 1 (exclusive), received ${confidenceLevel}.`
        );
      }
    }

    const body = {
      vector_a: vectorA,
      vector_b: vectorB,
      bootstrap_iterations: bootstrapIterations !== undefined ? bootstrapIterations : DEFAULT_BOOTSTRAP_ITERATIONS,
      confidence_level: confidenceLevel !== undefined ? confidenceLevel : 0.95,
    };

    return dispatchApiRequest(
      this._parsedBase, this._apiKey, 'POST', '/bootstrap/nmi-significance', body, this._timeoutMs
    );
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        'mainMethod requires a data object. For ranked similarity search, provide { queryVector, corpus }. For batch queries, provide { queryVectors, corpus }.'
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        `mainMethod data must be an object, received ${typeof data}.`
      );
    }

    if (Array.isArray(data.queryVectors)) {
      return this.rankBatchQueriesByNmi(data);
    }

    if (Array.isArray(data.queryVector)) {
      return this.rankByCosineNmi(data);
    }

    throw new SimilaritySearchValidationError(
      'mainMethod data must include either queryVector (single query) or queryVectors (batch queries), along with a corpus array.'
    );
  }
}

function createSimilaritySearchClient(options) {
  return new SimilaritySearchClient(options || {});
}

const _defaultInstance = (() => {
  const envKey = process.env.SIMILARITY_SEARCH_API_KEY;
  if (envKey) {
    try {
      return new SimilaritySearchClient({ apiKey: envKey });
    } catch (e) {
      return null;
    }
  }
  return null;
})();

async function mainMethod(data) {
  if (!_defaultInstance) {
    throw new SimilaritySearchAuthError(
      'No default client available. Set SIMILARITY_SEARCH_API_KEY environment variable or use createClient() with explicit options.'
    );
  }
  return _defaultInstance.mainMethod(data);
}

async function rankByCosineNmi(params) {
  if (!_defaultInstance) {
    throw new SimilaritySearchAuthError(
      'No default client available. Set SIMILARITY_SEARCH_API_KEY environment variable or use createClient().'
    );
  }
  return _defaultInstance.rankByCosineNmi(params);
}

async function rankBatchQueriesByNmi(params) {
  if (!_defaultInstance) {
    throw new SimilaritySearchAuthError(
      'No default client available. Set SIMILARITY_SEARCH_API_KEY environment variable or use createClient().'
    );
  }
  return _defaultInstance.rankBatchQueriesByNmi(params);
}

async function estimateJointEntropyProfile(params) {
  if (!_defaultInstance) {
    throw new SimilaritySearchAuthError(
      'No default client available. Set SIMILARITY_SEARCH_API_KEY environment variable or use createClient().'
    );
  }
  return _defaultInstance.estimateJointEntropyProfile(params);
}

async function fetchNmiSignificanceBootstrap(params) {
  if (!_defaultInstance) {
    throw new SimilaritySearchAuthError(
      'No default client available. Set SIMILARITY_SEARCH_API_KEY environment variable or use createClient().'
    );
  }
  return _defaultInstance.fetchNmiSignificanceBootstrap(params);
}

module.exports = {
  createClient: createSimilaritySearchClient,
  SimilaritySearchClient,
  SimilaritySearchError,
  SimilaritySearchAuthError,
  SimilaritySearchValidationError,
  SimilaritySearchRateLimitError,
  mainMethod,
  rankByCosineNmi,
  rankBatchQueriesByNmi,
  estimateJointEntropyProfile,
  fetchNmiSignificanceBootstrap,
  DEFAULT_BOOTSTRAP_ITERATIONS,
  DEFAULT_SIGNIFICANCE_THRESHOLD,
  MAX_EMBEDDING_DIMENSIONS,
  MAX_VECTORS_PER_CALL,
  MAX_QUERY_VECTORS,
};
```