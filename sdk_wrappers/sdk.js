const axios = require('axios');

const DEFAULT_BASE_URL = 'https://api.similaritysearch.io/v1';
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_MAX_RETRIES = 3;
const RETRY_BACKOFF_BASE_MS = 200;

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

class SimilaritySearchRateLimitError extends Error {
  constructor(message, retryAfterSeconds) {
    super(message);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds || null;
  }
}

class SimilaritySearchAPIError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'SimilaritySearchAPIError';
    this.statusCode = statusCode;
    this.body = body;
  }
}

function validateApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new SimilaritySearchAuthError(
      'API key is required. Pass { apiKey } to the client constructor or set SIMILARITY_SEARCH_API_KEY env var.'
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

function validateNonEmptyArray(value, fieldName) {
  if (value === null || value === undefined) {
    throw new SimilaritySearchValidationError(
      `'${fieldName}' is required and cannot be null or undefined.`
    );
  }
  if (!Array.isArray(value)) {
    throw new SimilaritySearchValidationError(
      `'${fieldName}' must be an array, received ${typeof value}.`
    );
  }
  if (value.length === 0) {
    throw new SimilaritySearchValidationError(
      `'${fieldName}' must contain at least one item.`
    );
  }
}

function validateItem(item, index, collectionName) {
  if (item === null || item === undefined || typeof item !== 'object' || Array.isArray(item)) {
    throw new SimilaritySearchValidationError(
      `${collectionName}[${index}] must be a plain object with feature key-value pairs, received ${Array.isArray(item) ? 'array' : typeof item}.`
    );
  }
  if (Object.keys(item).length === 0) {
    throw new SimilaritySearchValidationError(
      `${collectionName}[${index}] must have at least one feature key, received empty object.`
    );
  }
  for (const [key, val] of Object.entries(item)) {
    if (typeof key !== 'string' || key.trim().length === 0) {
      throw new SimilaritySearchValidationError(
        `${collectionName}[${index}] contains an invalid feature key: keys must be non-empty strings.`
      );
    }
    if (val === null || val === undefined) {
      throw new SimilaritySearchValidationError(
        `${collectionName}[${index}].${key} is null or undefined. Feature values must be numbers, strings, or booleans.`
      );
    }
    const t = typeof val;
    if (t !== 'number' && t !== 'string' && t !== 'boolean') {
      throw new SimilaritySearchValidationError(
        `${collectionName}[${index}].${key} has unsupported type '${t}'. Feature values must be number, string, or boolean.`
      );
    }
    if (t === 'number' && !isFinite(val)) {
      throw new SimilaritySearchValidationError(
        `${collectionName}[${index}].${key} is ${val}. Numeric feature values must be finite.`
      );
    }
  }
}

function validateNmiWeightedSearchInput(query, collection, options) {
  validateNonEmptyArray(query, 'query');
  validateNonEmptyArray(collection, 'collection');

  query.forEach((item, i) => validateItem(item, i, 'query'));
  collection.forEach((item, i) => validateItem(item, i, 'collection'));

  if (query.length > 10000) {
    throw new SimilaritySearchValidationError(
      `'query' contains ${query.length} items; maximum allowed is 10000 per call.`
    );
  }
  if (collection.length > 100000) {
    throw new SimilaritySearchValidationError(
      `'collection' contains ${collection.length} items; maximum allowed is 100000 per call.`
    );
  }

  if (options !== undefined && options !== null) {
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError(
        `'options' must be a plain object, received ${Array.isArray(options) ? 'array' : typeof options}.`
      );
    }
    if (options.topK !== undefined) {
      if (typeof options.topK !== 'number' || !Number.isInteger(options.topK) || options.topK < 1 || options.topK > 10000) {
        throw new SimilaritySearchValidationError(
          `'options.topK' must be an integer between 1 and 10000, received ${options.topK}.`
        );
      }
    }
    if (options.scoreThreshold !== undefined) {
      if (typeof options.scoreThreshold !== 'number' || options.scoreThreshold < 0 || options.scoreThreshold > 1) {
        throw new SimilaritySearchValidationError(
          `'options.scoreThreshold' must be a number between 0 and 1, received ${options.scoreThreshold}.`
        );
      }
    }
    if (options.nmiSmoothing !== undefined) {
      if (typeof options.nmiSmoothing !== 'number' || options.nmiSmoothing < 0) {
        throw new SimilaritySearchValidationError(
          `'options.nmiSmoothing' must be a non-negative number, received ${options.nmiSmoothing}.`
        );
      }
    }
    if (options.includeNmiWeights !== undefined && typeof options.includeNmiWeights !== 'boolean') {
      throw new SimilaritySearchValidationError(
        `'options.includeNmiWeights' must be a boolean, received ${typeof options.includeNmiWeights}.`
      );
    }
  }
}

function validateFeatureWeightsInput(collection) {
  validateNonEmptyArray(collection, 'collection');
  if (collection.length < 2) {
    throw new SimilaritySearchValidationError(
      `'collection' must contain at least 2 items to compute NMI feature weights; received ${collection.length}.`
    );
  }
  collection.forEach((item, i) => validateItem(item, i, 'collection'));
}

function validatePairwiseNmiMatrixInput(collectionA, collectionB) {
  validateNonEmptyArray(collectionA, 'collectionA');
  validateNonEmptyArray(collectionB, 'collectionB');
  collectionA.forEach((item, i) => validateItem(item, i, 'collectionA'));
  collectionB.forEach((item, i) => validateItem(item, i, 'collectionB'));

  const keysA = new Set(Object.keys(collectionA[0]));
  const keysB = new Set(Object.keys(collectionB[0]));
  const overlap = [...keysA].filter(k => keysB.has(k));
  if (overlap.length === 0) {
    throw new SimilaritySearchValidationError(
      'collectionA and collectionB share no common feature keys. Pairwise NMI matrix requires at least one overlapping feature dimension.'
    );
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isRetryableStatusCode(status) {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

class SimilaritySearchClient {
  constructor(options) {
    if (options === null || options === undefined) {
      options = {};
    }
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError(
        `SimilaritySearchClient constructor expects a plain options object, received ${Array.isArray(options) ? 'array' : typeof options}.`
      );
    }

    const apiKey = options.apiKey || process.env.SIMILARITY_SEARCH_API_KEY;
    validateApiKey(apiKey);

    this._apiKey = apiKey;
    this._baseUrl = (options.baseUrl || DEFAULT_BASE_URL).replace(/\/$/, '');
    this._timeoutMs = typeof options.timeoutMs === 'number' && options.timeoutMs > 0
      ? options.timeoutMs
      : DEFAULT_TIMEOUT_MS;
    this._maxRetries = typeof options.maxRetries === 'number' && options.maxRetries >= 0
      ? Math.floor(options.maxRetries)
      : DEFAULT_MAX_RETRIES;

    this._http = axios.create({
      baseURL: this._baseUrl,
      timeout: this._timeoutMs,
      headers: {
        'X-Api-Key': this._apiKey,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'similarity-search-sdk-js/1.0.0'
      }
    });
  }

  async _postWithRetry(path, body) {
    let attempt = 0;
    while (true) {
      try {
        const response = await this._http.post(path, body);
        return response.data;
      } catch (err) {
        if (err.response) {
          const status = err.response.status;
          const responseBody = err.response.data;

          if (status === 401 || status === 403) {
            throw new SimilaritySearchAuthError(
              `Authentication failed (HTTP ${status}): ${responseBody && responseBody.detail ? responseBody.detail : 'Invalid or missing API key.'}`
            );
          }

          if (status === 422) {
            const detail = responseBody && responseBody.detail
              ? JSON.stringify(responseBody.detail)
              : 'Unprocessable entity.';
            throw new SimilaritySearchValidationError(
              `Server rejected request payload (HTTP 422): ${detail}`
            );
          }

          if (status === 429) {
            const retryAfter = err.response.headers['retry-after']
              ? parseFloat(err.response.headers['retry-after'])
              : null;
            if (attempt >= this._maxRetries) {
              throw new SimilaritySearchRateLimitError(
                `Rate limit exceeded after ${attempt} retries. ${retryAfter ? 'Retry after ' + retryAfter + 's.' : ''}`,
                retryAfter
              );
            }
            const waitMs = retryAfter
              ? retryAfter * 1000
              : RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt);
            await sleep(waitMs);
            attempt++;
            continue;
          }

          if (isRetryableStatusCode(status) && attempt < this._maxRetries) {
            const waitMs = RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt);
            await sleep(waitMs);
            attempt++;
            continue;
          }

          throw new SimilaritySearchAPIError(
            `API error (HTTP ${status}): ${responseBody && responseBody.detail ? responseBody.detail : 'Unexpected server error.'}`,
            status,
            responseBody
          );
        }

        if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
          if (attempt < this._maxRetries) {
            const waitMs = RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt);
            await sleep(waitMs);
            attempt++;
            continue;
          }
          throw new SimilaritySearchAPIError(
            `Request timed out after ${this._timeoutMs}ms (${attempt + 1} attempts). Consider increasing 'timeoutMs' for large collections.`,
            null,
            null
          );
        }

        throw new SimilaritySearchAPIError(
          `Network error: ${err.message}`,
          null,
          null
        );
      }
    }
  }

  async nmiWeightedSearch(query, collection, options) {
    validateNmiWeightedSearchInput(query, collection, options);

    const body = {
      query,
      collection,
      options: options || {}
    };

    return await this._postWithRetry('/nmi-weighted-search', body);
  }

  async computeNmiFeatureWeights(collection) {
    validateFeatureWeightsInput(collection);

    const body = { collection };
    return await this._postWithRetry('/nmi-feature-weights', body);
  }

  async pairwiseNmiMatrix(collectionA, collectionB) {
    validatePairwiseNmiMatrixInput(collectionA, collectionB);

    const body = { collection_a: collectionA, collection_b: collectionB };
    return await this._postWithRetry('/pairwise-nmi-matrix', body);
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        "'data' is required. Expected { query, collection, options? }."
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        `'data' must be a plain object with { query, collection, options? }, received ${Array.isArray(data) ? 'array' : typeof data}.`
      );
    }
    if (!data.query && !data.collection) {
      throw new SimilaritySearchValidationError(
        "'data' must include 'query' and 'collection' arrays. Received neither."
      );
    }
    return await this.nmiWeightedSearch(data.query, data.collection, data.options);
  }
}

function createClient(options) {
  return new SimilaritySearchClient(options || {});
}

const _defaultClient = {
  _instance: null,
  _getOrCreate() {
    if (!this._instance) {
      const apiKey = process.env.SIMILARITY_SEARCH_API_KEY;
      if (!apiKey) {
        throw new SimilaritySearchAuthError(
          'No API key found. Set SIMILARITY_SEARCH_API_KEY env var or use createClient({ apiKey }) instead of the module-level shorthand.'
        );
      }
      this._instance = new SimilaritySearchClient({ apiKey });
    }
    return this._instance;
  }
};

async function mainMethod(data) {
  return _defaultClient._getOrCreate().mainMethod(data);
}

async function nmiWeightedSearch(query, collection, options) {
  return _defaultClient._getOrCreate().nmiWeightedSearch(query, collection, options);
}

async function computeNmiFeatureWeights(collection) {
  return _defaultClient._getOrCreate().computeNmiFeatureWeights(collection);
}

async function pairwiseNmiMatrix(collectionA, collectionB) {
  return _defaultClient._getOrCreate().pairwiseNmiMatrix(collectionA, collectionB);
}

module.exports = {
  createClient,
  SimilaritySearchClient,
  mainMethod,
  nmiWeightedSearch,
  computeNmiFeatureWeights,
  pairwiseNmiMatrix,
  SimilaritySearchAuthError,
  SimilaritySearchValidationError,
  SimilaritySearchRateLimitError,
  SimilaritySearchAPIError
};