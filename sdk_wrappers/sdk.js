
```javascript
'use strict';

const https = require('https');
const http = require('http');
const { URL } = require('url');

const SIMILARITY_SEARCH_API_DEFAULT_BASE_URL = 'https://api.similaritysearch.io/v1';
const SIMILARITY_SEARCH_API_DEFAULT_TIMEOUT_MS = 30000;
const SIMILARITY_SEARCH_API_DEFAULT_MAX_RETRIES = 3;
const SIMILARITY_SEARCH_API_DEFAULT_RETRY_DELAY_MS = 500;

class SimilaritySearchValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'SimilaritySearchValidationError';
    this.field = field || null;
  }
}

class SimilaritySearchAuthError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SimilaritySearchAuthError';
  }
}

class SimilaritySearchRateLimitError extends Error {
  constructor(message, retryAfterMs) {
    super(message);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterMs = retryAfterMs || null;
  }
}

class SimilaritySearchAPIError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'SimilaritySearchAPIError';
    this.statusCode = statusCode;
    this.body = body || null;
  }
}

class SimilaritySearchTimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SimilaritySearchTimeoutError';
  }
}

function validateNonEmptyString(value, fieldName) {
  if (value === null || value === undefined) {
    throw new SimilaritySearchValidationError(
      `${fieldName} is required and cannot be null or undefined`,
      fieldName
    );
  }
  if (typeof value !== 'string') {
    throw new SimilaritySearchValidationError(
      `${fieldName} must be a string, got ${typeof value}`,
      fieldName
    );
  }
  if (value.trim().length === 0) {
    throw new SimilaritySearchValidationError(
      `${fieldName} cannot be an empty string`,
      fieldName
    );
  }
}

function validateFeatureRecord(record, fieldName) {
  if (record === null || record === undefined) {
    throw new SimilaritySearchValidationError(
      `${fieldName} is required and cannot be null or undefined`,
      fieldName
    );
  }
  if (typeof record !== 'object' || Array.isArray(record)) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must be a plain object mapping feature names to values`,
      fieldName
    );
  }
  const keys = Object.keys(record);
  if (keys.length === 0) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must contain at least one feature`,
      fieldName
    );
  }
  for (const key of keys) {
    if (typeof key !== 'string' || key.trim().length === 0) {
      throw new SimilaritySearchValidationError(
        `${fieldName}: all feature keys must be non-empty strings`,
        fieldName
      );
    }
    const val = record[key];
    if (
      val === null ||
      val === undefined ||
      (typeof val !== 'string' &&
        typeof val !== 'number' &&
        typeof val !== 'boolean' &&
        !Array.isArray(val))
    ) {
      throw new SimilaritySearchValidationError(
        `${fieldName}: feature "${key}" must be a string, number, boolean, or numeric array`,
        fieldName
      );
    }
    if (Array.isArray(val)) {
      for (let i = 0; i < val.length; i++) {
        if (typeof val[i] !== 'number' || !isFinite(val[i])) {
          throw new SimilaritySearchValidationError(
            `${fieldName}: feature "${key}" array element at index ${i} must be a finite number`,
            fieldName
          );
        }
      }
      if (val.length === 0) {
        throw new SimilaritySearchValidationError(
          `${fieldName}: feature "${key}" numeric array cannot be empty`,
          fieldName
        );
      }
    }
  }
}

function validateCorpusItems(items, fieldName) {
  if (!Array.isArray(items)) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must be an array of corpus items`,
      fieldName
    );
  }
  if (items.length === 0) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must contain at least one item`,
      fieldName
    );
  }
  if (items.length > 100000) {
    throw new SimilaritySearchValidationError(
      `${fieldName} exceeds maximum corpus size of 100,000 items`,
      fieldName
    );
  }
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (typeof item !== 'object' || item === null || Array.isArray(item)) {
      throw new SimilaritySearchValidationError(
        `${fieldName}[${i}] must be a plain object`,
        fieldName
      );
    }
    if (!item.id) {
      throw new SimilaritySearchValidationError(
        `${fieldName}[${i}] must have an "id" property`,
        fieldName
      );
    }
    validateNonEmptyString(String(item.id), `${fieldName}[${i}].id`);
    if (!item.features) {
      throw new SimilaritySearchValidationError(
        `${fieldName}[${i}] must have a "features" property`,
        fieldName
      );
    }
    validateFeatureRecord(item.features, `${fieldName}[${i}].features`);
  }
}

function validateTopK(topK) {
  if (topK === null || topK === undefined) return;
  if (typeof topK !== 'number' || !Number.isInteger(topK)) {
    throw new SimilaritySearchValidationError(
      'topK must be an integer',
      'topK'
    );
  }
  if (topK < 1 || topK > 1000) {
    throw new SimilaritySearchValidationError(
      'topK must be between 1 and 1000',
      'topK'
    );
  }
}

function validateScoreThreshold(threshold) {
  if (threshold === null || threshold === undefined) return;
  if (typeof threshold !== 'number' || !isFinite(threshold)) {
    throw new SimilaritySearchValidationError(
      'scoreThreshold must be a finite number',
      'scoreThreshold'
    );
  }
  if (threshold < 0.0 || threshold > 1.0) {
    throw new SimilaritySearchValidationError(
      'scoreThreshold must be between 0.0 and 1.0',
      'scoreThreshold'
    );
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function makeHttpRequest(options, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(options.url);
    const isHttps = parsedUrl.protocol === 'https:';
    const transport = isHttps ? https : http;

    const reqOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: parsedUrl.pathname + (parsedUrl.search || ''),
      method: options.method || 'POST',
      headers: options.headers || {},
    };

    const req = transport.request(reqOptions, (res) => {
      let rawData = '';
      res.on('data', (chunk) => {
        rawData += chunk;
      });
      res.on('end', () => {
        resolve({ statusCode: res.statusCode, headers: res.headers, body: rawData });
      });
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new SimilaritySearchTimeoutError(
        `Request timed out after ${timeoutMs}ms`
      ));
    });

    req.on('error', (err) => {
      reject(new SimilaritySearchAPIError(
        `Network error: ${err.message}`,
        null,
        null
      ));
    });

    if (body) {
      req.write(body);
    }
    req.end();
  });
}

function parseResponseBody(rawBody, statusCode) {
  if (!rawBody || rawBody.trim().length === 0) {
    return null;
  }
  try {
    return JSON.parse(rawBody);
  } catch (e) {
    throw new SimilaritySearchAPIError(
      `Server returned non-JSON response (status ${statusCode}): ${rawBody.slice(0, 200)}`,
      statusCode,
      rawBody
    );
  }
}

function handleHttpError(statusCode, parsedBody, rawBody) {
  if (statusCode === 401 || statusCode === 403) {
    const msg = (parsedBody && parsedBody.detail) || 'Authentication failed. Check your API key.';
    throw new SimilaritySearchAuthError(msg);
  }
  if (statusCode === 429) {
    let retryAfterMs = null;
    if (parsedBody && parsedBody.retry_after_ms) {
      retryAfterMs = parsedBody.retry_after_ms;
    }
    throw new SimilaritySearchRateLimitError(
      'Rate limit exceeded. Reduce request frequency or upgrade your plan.',
      retryAfterMs
    );
  }
  if (statusCode === 422) {
    const detail = (parsedBody && parsedBody.detail) || JSON.stringify(parsedBody);
    throw new SimilaritySearchValidationError(
      `Server rejected request payload: ${detail}`,
      null
    );
  }
  if (statusCode >= 400) {
    const msg = (parsedBody && parsedBody.detail) || `API error (status ${statusCode})`;
    throw new SimilaritySearchAPIError(msg, statusCode, parsedBody || rawBody);
  }
}

class SimilaritySearchClient {
  constructor(config) {
    if (!config || typeof config !== 'object') {
      throw new SimilaritySearchValidationError(
        'SimilaritySearchClient requires a configuration object with at least an "apiKey" property',
        'config'
      );
    }

    validateNonEmptyString(config.apiKey, 'config.apiKey');

    this._apiKey = config.apiKey;
    this._baseUrl = (config.baseUrl || SIMILARITY_SEARCH_API_DEFAULT_BASE_URL).replace(/\/$/, '');
    this._timeoutMs = config.timeoutMs !== undefined ? config.timeoutMs : SIMILARITY_SEARCH_API_DEFAULT_TIMEOUT_MS;
    this._maxRetries = config.maxRetries !== undefined ? config.maxRetries : SIMILARITY_SEARCH_API_DEFAULT_MAX_RETRIES;
    this._retryDelayMs = config.retryDelayMs !== undefined ? config.retryDelayMs : SIMILARITY_SEARCH_API_DEFAULT_RETRY_DELAY_MS;

    if (typeof this._timeoutMs !== 'number' || this._timeoutMs < 1000) {
      throw new SimilaritySearchValidationError(
        'config.timeoutMs must be a number >= 1000',
        'config.timeoutMs'
      );
    }
    if (typeof this._maxRetries !== 'number' || this._maxRetries < 0 || this._maxRetries > 10) {
      throw new SimilaritySearchValidationError(
        'config.maxRetries must be a number between 0 and 10',
        'config.maxRetries'
      );
    }
  }

  _buildHeaders(extraHeaders) {
    return Object.assign(
      {
        'Authorization': `Bearer ${this._apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'similarity-search-sdk-js/1.0.0',
      },
      extraHeaders || {}
    );
  }

  async _requestWithRetry(method, path, payload) {
    const url = `${this._baseUrl}${path}`;
    const body = payload !== undefined ? JSON.stringify(payload) : undefined;
    const headers = this._buildHeaders(
      body ? { 'Content-Length': Buffer.byteLength(body, 'utf8').toString() } : {}
    );

    let lastError = null;
    for (let attempt = 0; attempt <= this._maxRetries; attempt++) {
      if (attempt > 0) {
        const delay = this._retryDelayMs * Math.pow(2, attempt - 1);
        await sleep(delay);
      }
      try {
        const response = await makeHttpRequest(
          { url, method, headers },
          body,
          this._timeoutMs
        );
        const parsed = parseResponseBody(response.body, response.statusCode);
        if (response.statusCode === 429) {
          const retryAfterMs = (parsed && parsed.retry_after_ms) || null;
          if (attempt < this._maxRetries) {
            const waitMs = retryAfterMs || this._retryDelayMs * Math.pow(2, attempt);
            await sleep(waitMs);
            lastError = new SimilaritySearchRateLimitError(
              'Rate limit exceeded. Retrying...',
              retryAfterMs
            );
            continue;
          }
          throw new SimilaritySearchRateLimitError(
            'Rate limit exceeded after all retries. Reduce request frequency or upgrade your plan.',
            retryAfterMs
          );
        }
        if (response.statusCode >= 500 && attempt < this._maxRetries) {
          lastError = new SimilaritySearchAPIError(
            `Server error (status ${response.statusCode}), retrying...`,
            response.statusCode,
            parsed
          );
          continue;
        }
        handleHttpError(response.statusCode, parsed, response.body);
        return parsed;
      } catch (err) {
        if (
          err instanceof SimilaritySearchAuthError ||
          err instanceof SimilaritySearchValidationError
        ) {
          throw err;
        }
        if (err instanceof SimilaritySearchTimeoutError && attempt < this._maxRetries) {
          lastError = err;
          continue;
        }
        if (attempt >= this._maxRetries) {
          throw err;
        }
        lastError = err;
      }
    }
    throw lastError;
  }

  async hybridSimilaritySearch(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'hybridSimilaritySearch requires a params object',
        'params'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        'params must be a plain object',
        'params'
      );
    }

    validateFeatureRecord(params.query, 'params.query');
    validateCorpusItems(params.corpus, 'params.corpus');
    validateTopK(params.topK);
    validateScoreThreshold(params.scoreThreshold);

    const payload = {
      query: params.query,
      corpus: params.corpus,
      top_k: params.topK !== undefined ? params.topK : 10,
    };

    if (params.scoreThreshold !== undefined && params.scoreThreshold !== null) {
      payload.score_threshold = params.scoreThreshold;
    }

    if (params.explainScores !== undefined) {
      payload.explain_scores = Boolean(params.explainScores);
    }

    return this._requestWithRetry('POST', '/search/hybrid', payload);
  }

  async batchHybridSimilaritySearch(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'batchHybridSimilaritySearch requires a params object',
        'params'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        'params must be a plain object',
        'params'
      );
    }

    if (!Array.isArray(params.queries)) {
      throw new SimilaritySearchValidationError(
        'params.queries must be an array of feature records',
        'params.queries'
      );
    }
    if (params.queries.length === 0) {
      throw new SimilaritySearchValidationError(
        'params.queries must contain at least one query',
        'params.queries'
      );
    }
    if (params.queries.length > 50) {
      throw new SimilaritySearchValidationError(
        'params.queries cannot exceed 50 queries per batch call',
        'params.queries'
      );
    }

    for (let i = 0; i < params.queries.length; i++) {
      validateFeatureRecord(params.queries[i], `params.queries[${i}]`);
    }

    validateCorpusItems(params.corpus, 'params.corpus');
    validateTopK(params.topK);
    validateScoreThreshold(params.scoreThreshold);

    const payload = {
      queries: params.queries,
      corpus: params.corpus,
      top_k: params.topK !== undefined ? params.topK : 10,
    };

    if (params.scoreThreshold !== undefined && params.scoreThreshold !== null) {
      payload.score_threshold = params.scoreThreshold;
    }

    if (params.explainScores !== undefined) {
      payload.explain_scores = Boolean(params.explainScores);
    }

    return this._requestWithRetry('POST', '/search/hybrid/batch', payload);
  }

  async explainHybridWeights(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'explainHybridWeights requires a params object',
        'params'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        'params must be a plain object',
        'params'
      );
    }

    validateFeatureRecord(params.sampleRecord, 'params.sampleRecord');

    const payload = {
      sample_record: params.sampleRecord,
    };

    return this._requestWithRetry('POST', '/search/explain-weights', payload);
  }

  async getUsage() {
    return this._requestWithRetry('GET', '/account/usage', undefined);
  }
}

function createSimilaritySearchClient(config) {
  return new SimilaritySearchClient(config);
}

const _defaultClientHolder = { client: null };

function _getOrCreateDefaultClient() {
  if (!_defaultClientHolder.client) {
    const apiKey = process.env.SIMILARITY_SEARCH_API_KEY;
    if (!apiKey || apiKey.trim().length === 0) {
      throw new SimilaritySearchAuthError(
        'No API key provided. Either pass apiKey to createSimilaritySearchClient() ' +
        'or set the SIMILARITY_SEARCH_API_KEY environment variable.'
      );
    }
    _defaultClientHolder.client = new SimilaritySearchClient({ apiKey });
  }
  return _defaultClientHolder.client;
}

async function mainMethod(data) {
  if (data === null || data === undefined) {
    throw new SimilaritySearchValidationError(
      'mainMethod requires a data object with "query" and "corpus" properties',
      'data'
    );
  }
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new SimilaritySearchValidationError(
      'data must be a plain object',
      'data'
    );
  }

  const client = _getOrCreateDefaultClient();
  return client.hybridSimilaritySearch(data);
}

module.exports = {
  mainMethod,
  createSimilaritySearchClient,
  SimilaritySearchClient,
  SimilaritySearchValidationError,
  SimilaritySearchAuthError,
  SimilaritySearchRateLimitError,
  SimilaritySearchAPIError,
  SimilaritySearchTimeoutError,
};
```