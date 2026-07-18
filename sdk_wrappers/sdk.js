const axios = require('axios');

const DEFAULT_BASE_URL = 'https://api.similaritysearch.nexus/v1';
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_TOP_K = 10;
const MAX_BATCH_SIZE = 500;
const MIN_TOP_K = 1;
const MAX_TOP_K = 500;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.body = body || null;
  }
}

class AuthenticationError extends SimilaritySearchError {
  constructor(body) {
    super('Invalid or missing API key for Similarity Search API', 401, body);
    this.name = 'AuthenticationError';
  }
}

class RateLimitError extends SimilaritySearchError {
  constructor(retryAfter, body) {
    super(`Rate limit exceeded. Retry after ${retryAfter}s`, 429, body);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

class ValidationError extends SimilaritySearchError {
  constructor(message, body) {
    super(message, 422, body);
    this.name = 'ValidationError';
  }
}

function assertNonEmptyString(value, label) {
  if (value === null || value === undefined) {
    throw new ValidationError(`${label} must not be null or undefined`);
  }
  if (typeof value !== 'string') {
    throw new ValidationError(`${label} must be a string, got ${typeof value}`);
  }
  if (value.trim().length === 0) {
    throw new ValidationError(`${label} must not be an empty string`);
  }
}

function assertNonEmptyArray(value, label) {
  if (value === null || value === undefined) {
    throw new ValidationError(`${label} must not be null or undefined`);
  }
  if (!Array.isArray(value)) {
    throw new ValidationError(`${label} must be an array, got ${typeof value}`);
  }
  if (value.length === 0) {
    throw new ValidationError(`${label} must contain at least one item`);
  }
}

function assertRecord(value, label) {
  if (value === null || value === undefined) {
    throw new ValidationError(`${label} must not be null or undefined`);
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new ValidationError(`${label} must be a plain object`);
  }
}

function assertInteger(value, label, min, max) {
  if (value === null || value === undefined) {
    throw new ValidationError(`${label} must not be null or undefined`);
  }
  if (!Number.isInteger(value)) {
    throw new ValidationError(`${label} must be an integer, got ${typeof value}`);
  }
  if (min !== undefined && value < min) {
    throw new ValidationError(`${label} must be >= ${min}, got ${value}`);
  }
  if (max !== undefined && value > max) {
    throw new ValidationError(`${label} must be <= ${max}, got ${value}`);
  }
}

function buildHttpClient(apiKey, baseUrl, timeoutMs) {
  return axios.create({
    baseURL: baseUrl,
    timeout: timeoutMs,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'X-Client': 'similarity-search-sdk-js/1.0.0',
    },
  });
}

function parseApiError(error) {
  if (!error.response) {
    throw new SimilaritySearchError(
      `Network error: ${error.message}`,
      null,
      null
    );
  }

  const { status, data, headers } = error.response;

  if (status === 401) {
    throw new AuthenticationError(data);
  }

  if (status === 429) {
    const retryAfter = parseInt(headers['retry-after'] || '60', 10);
    throw new RateLimitError(retryAfter, data);
  }

  if (status === 422) {
    const detail = (data && data.detail)
      ? (Array.isArray(data.detail)
          ? data.detail.map(d => d.msg || JSON.stringify(d)).join('; ')
          : String(data.detail))
      : 'Unprocessable entity';
    throw new ValidationError(detail, data);
  }

  throw new SimilaritySearchError(
    `API error ${status}: ${JSON.stringify(data)}`,
    status,
    data
  );
}

class SimilaritySearchClient {
  constructor(options) {
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new ValidationError(
        'SimilaritySearchClient requires an options object with at least { apiKey }'
      );
    }

    const apiKey = options.apiKey || process.env.SIMILARITY_SEARCH_API_KEY;
    assertNonEmptyString(apiKey, 'options.apiKey (or SIMILARITY_SEARCH_API_KEY env var)');

    this._apiKey = apiKey;
    this._baseUrl = (options.baseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
    this._timeoutMs = Number.isInteger(options.timeoutMs)
      ? options.timeoutMs
      : DEFAULT_TIMEOUT_MS;

    this._http = buildHttpClient(this._apiKey, this._baseUrl, this._timeoutMs);
  }

  async hybridSimilaritySearch(query, corpus, options) {
    if (query === null || query === undefined) {
      throw new ValidationError('query must not be null or undefined');
    }
    if (typeof query !== 'object' || Array.isArray(query)) {
      throw new ValidationError('query must be a plain object with feature key-value pairs');
    }
    if (Object.keys(query).length === 0) {
      throw new ValidationError('query must contain at least one feature');
    }

    assertNonEmptyArray(corpus, 'corpus');

    if (corpus.length > MAX_BATCH_SIZE) {
      throw new ValidationError(
        `corpus exceeds maximum batch size of ${MAX_BATCH_SIZE} items, got ${corpus.length}`
      );
    }

    corpus.forEach((item, idx) => {
      if (item === null || item === undefined || typeof item !== 'object' || Array.isArray(item)) {
        throw new ValidationError(
          `corpus[${idx}] must be a plain object with feature key-value pairs`
        );
      }
      if (Object.keys(item).length === 0) {
        throw new ValidationError(`corpus[${idx}] must contain at least one feature`);
      }
    });

    const topK = (options && Number.isInteger(options.topK))
      ? options.topK
      : DEFAULT_TOP_K;
    assertInteger(topK, 'options.topK', MIN_TOP_K, MAX_TOP_K);

    const featureTypes = (options && options.featureTypes) ? options.featureTypes : undefined;
    if (featureTypes !== undefined) {
      assertRecord(featureTypes, 'options.featureTypes');
      const validTypes = ['categorical', 'continuous', 'embedding'];
      for (const [key, val] of Object.entries(featureTypes)) {
        if (!validTypes.includes(val)) {
          throw new ValidationError(
            `options.featureTypes["${key}"] must be one of ${validTypes.join(', ')}, got "${val}"`
          );
        }
      }
    }

    const body = {
      query,
      corpus,
      top_k: topK,
    };

    if (featureTypes) {
      body.feature_types = featureTypes;
    }

    try {
      const response = await this._http.post('/hybrid-search', body);
      return response.data;
    } catch (error) {
      if (error instanceof SimilaritySearchError) throw error;
      parseApiError(error);
    }
  }

  async batchPairwiseHybridScore(pairs, options) {
    assertNonEmptyArray(pairs, 'pairs');

    if (pairs.length > MAX_BATCH_SIZE) {
      throw new ValidationError(
        `pairs exceeds maximum batch size of ${MAX_BATCH_SIZE}, got ${pairs.length}`
      );
    }

    pairs.forEach((pair, idx) => {
      if (!pair || typeof pair !== 'object' || Array.isArray(pair)) {
        throw new ValidationError(`pairs[${idx}] must be a plain object with { a, b }`);
      }
      if (!pair.a || typeof pair.a !== 'object' || Array.isArray(pair.a)) {
        throw new ValidationError(`pairs[${idx}].a must be a plain object with feature key-value pairs`);
      }
      if (!pair.b || typeof pair.b !== 'object' || Array.isArray(pair.b)) {
        throw new ValidationError(`pairs[${idx}].b must be a plain object with feature key-value pairs`);
      }
      if (Object.keys(pair.a).length === 0) {
        throw new ValidationError(`pairs[${idx}].a must contain at least one feature`);
      }
      if (Object.keys(pair.b).length === 0) {
        throw new ValidationError(`pairs[${idx}].b must contain at least one feature`);
      }
    });

    const featureTypes = (options && options.featureTypes) ? options.featureTypes : undefined;
    if (featureTypes !== undefined) {
      assertRecord(featureTypes, 'options.featureTypes');
      const validTypes = ['categorical', 'continuous', 'embedding'];
      for (const [key, val] of Object.entries(featureTypes)) {
        if (!validTypes.includes(val)) {
          throw new ValidationError(
            `options.featureTypes["${key}"] must be one of ${validTypes.join(', ')}, got "${val}"`
          );
        }
      }
    }

    const body = { pairs };
    if (featureTypes) {
      body.feature_types = featureTypes;
    }

    try {
      const response = await this._http.post('/batch-score', body);
      return response.data;
    } catch (error) {
      if (error instanceof SimilaritySearchError) throw error;
      parseApiError(error);
    }
  }

  async detectFeatureSchema(corpus) {
    assertNonEmptyArray(corpus, 'corpus');

    if (corpus.length > MAX_BATCH_SIZE) {
      throw new ValidationError(
        `corpus exceeds maximum batch size of ${MAX_BATCH_SIZE} items, got ${corpus.length}`
      );
    }

    corpus.forEach((item, idx) => {
      if (item === null || item === undefined || typeof item !== 'object' || Array.isArray(item)) {
        throw new ValidationError(
          `corpus[${idx}] must be a plain object with feature key-value pairs`
        );
      }
    });

    try {
      const response = await this._http.post('/detect-schema', { corpus });
      return response.data;
    } catch (error) {
      if (error instanceof SimilaritySearchError) throw error;
      parseApiError(error);
    }
  }

  async hybridWeightDiagnostics(corpus) {
    assertNonEmptyArray(corpus, 'corpus');

    if (corpus.length > MAX_BATCH_SIZE) {
      throw new ValidationError(
        `corpus exceeds maximum batch size of ${MAX_BATCH_SIZE} items, got ${corpus.length}`
      );
    }

    corpus.forEach((item, idx) => {
      if (item === null || item === undefined || typeof item !== 'object' || Array.isArray(item)) {
        throw new ValidationError(
          `corpus[${idx}] must be a plain object with feature key-value pairs`
        );
      }
    });

    try {
      const response = await this._http.post('/weight-diagnostics', { corpus });
      return response.data;
    } catch (error) {
      if (error instanceof SimilaritySearchError) throw error;
      parseApiError(error);
    }
  }
}

async function mainMethod(data) {
  if (data === null || data === undefined) {
    throw new ValidationError(
      'mainMethod: data must not be null or undefined. Expected { apiKey, query, corpus, options? }'
    );
  }
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new ValidationError(
      'mainMethod: data must be a plain object with { apiKey, query, corpus, options? }'
    );
  }

  const { apiKey, query, corpus, options } = data;

  const client = new SimilaritySearchClient({
    apiKey: apiKey || process.env.SIMILARITY_SEARCH_API_KEY,
    baseUrl: (options && options.baseUrl) ? options.baseUrl : DEFAULT_BASE_URL,
    timeoutMs: (options && options.timeoutMs) ? options.timeoutMs : DEFAULT_TIMEOUT_MS,
  });

  return client.hybridSimilaritySearch(query, corpus, options);
}

module.exports = {
  SimilaritySearchClient,
  mainMethod,
  SimilaritySearchError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  MAX_BATCH_SIZE,
  MIN_TOP_K,
  MAX_TOP_K,
};