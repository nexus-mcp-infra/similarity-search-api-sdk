
```javascript
'use strict';

const https = require('https');
const http = require('http');
const { URL } = require('url');

const SIMILARITY_SEARCH_API_BASE_URL = 'https://api.similaritysearch.io/v1';
const SIMILARITY_SEARCH_DEFAULT_TIMEOUT_MS = 30000;
const SIMILARITY_SEARCH_MAX_CORPUS_SIZE = 100000;
const SIMILARITY_SEARCH_MAX_VECTOR_DIM = 8192;
const SIMILARITY_SEARCH_MAX_TOP_K = 1000;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, code, details) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.code = code || 'UNKNOWN_ERROR';
    this.details = details || null;
  }
}

class SimilaritySearchAuthError extends SimilaritySearchError {
  constructor(message) {
    super(message, 401, 'AUTH_ERROR');
    this.name = 'SimilaritySearchAuthError';
  }
}

class SimilaritySearchValidationError extends SimilaritySearchError {
  constructor(message, details) {
    super(message, 422, 'VALIDATION_ERROR', details);
    this.name = 'SimilaritySearchValidationError';
  }
}

class SimilaritySearchRateLimitError extends SimilaritySearchError {
  constructor(retryAfterMs) {
    super('Rate limit exceeded', 429, 'RATE_LIMIT_EXCEEDED');
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterMs = retryAfterMs || null;
  }
}

function validateApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new SimilaritySearchAuthError(
      'API key is required. Pass it as { apiKey } in the client constructor or set SIMILARITY_SEARCH_API_KEY env variable.'
    );
  }
  if (typeof apiKey !== 'string') {
    throw new SimilaritySearchAuthError(
      `API key must be a string, received ${typeof apiKey}.`
    );
  }
  if (apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError('API key must not be empty.');
  }
}

function validateVector(vec, label) {
  if (!Array.isArray(vec)) {
    throw new SimilaritySearchValidationError(
      `${label} must be an array of numbers, received ${typeof vec}.`
    );
  }
  if (vec.length === 0) {
    throw new SimilaritySearchValidationError(`${label} must not be empty.`);
  }
  if (vec.length > SIMILARITY_SEARCH_MAX_VECTOR_DIM) {
    throw new SimilaritySearchValidationError(
      `${label} dimensionality ${vec.length} exceeds maximum ${SIMILARITY_SEARCH_MAX_VECTOR_DIM}.`
    );
  }
  for (let i = 0; i < vec.length; i++) {
    if (typeof vec[i] !== 'number' || !isFinite(vec[i])) {
      throw new SimilaritySearchValidationError(
        `${label}[${i}] is not a finite number: ${vec[i]}.`
      );
    }
  }
}

function validateTokenizedText(tokens, label) {
  if (!Array.isArray(tokens)) {
    throw new SimilaritySearchValidationError(
      `${label} must be an array of strings (tokenized text), received ${typeof tokens}.`
    );
  }
  if (tokens.length === 0) {
    throw new SimilaritySearchValidationError(`${label} must not be empty.`);
  }
  for (let i = 0; i < tokens.length; i++) {
    if (typeof tokens[i] !== 'string') {
      throw new SimilaritySearchValidationError(
        `${label}[${i}] must be a string token, received ${typeof tokens[i]}.`
      );
    }
  }
}

function validateTabularFeatures(features, label) {
  if (typeof features !== 'object' || features === null || Array.isArray(features)) {
    throw new SimilaritySearchValidationError(
      `${label} must be a plain object of { featureName: number }, received ${Array.isArray(features) ? 'array' : typeof features}.`
    );
  }
  const keys = Object.keys(features);
  if (keys.length === 0) {
    throw new SimilaritySearchValidationError(`${label} must have at least one feature.`);
  }
  for (const key of keys) {
    if (typeof features[key] !== 'number' || !isFinite(features[key])) {
      throw new SimilaritySearchValidationError(
        `${label}.${key} must be a finite number, received ${features[key]}.`
      );
    }
  }
}

function detectPayloadType(query) {
  if (query === null || query === undefined) {
    throw new SimilaritySearchValidationError(
      'query must not be null or undefined.'
    );
  }
  if (Array.isArray(query)) {
    if (query.length === 0) {
      throw new SimilaritySearchValidationError('query array must not be empty.');
    }
    if (typeof query[0] === 'number') return 'vector';
    if (typeof query[0] === 'string') return 'tokenized_text';
    throw new SimilaritySearchValidationError(
      'query array elements must be all numbers (vector) or all strings (tokenized_text).'
    );
  }
  if (typeof query === 'object') return 'tabular';
  throw new SimilaritySearchValidationError(
    `query must be an array (vector/tokenized_text) or object (tabular), received ${typeof query}.`
  );
}

function validateCorpusEntry(entry, index, queryType) {
  if (entry === null || entry === undefined) {
    throw new SimilaritySearchValidationError(
      `corpus[${index}] must not be null or undefined.`
    );
  }
  const hasId = typeof entry.id !== 'undefined';
  if (!hasId) {
    throw new SimilaritySearchValidationError(
      `corpus[${index}] must have an 'id' field.`
    );
  }
  if (typeof entry.id !== 'string' && typeof entry.id !== 'number') {
    throw new SimilaritySearchValidationError(
      `corpus[${index}].id must be a string or number.`
    );
  }
  if (!('payload' in entry)) {
    throw new SimilaritySearchValidationError(
      `corpus[${index}] must have a 'payload' field.`
    );
  }
  const entryType = detectPayloadType(entry.payload);
  if (entryType !== queryType) {
    throw new SimilaritySearchValidationError(
      `corpus[${index}].payload type '${entryType}' does not match query type '${queryType}'. All corpus entries must use the same payload type as the query.`
    );
  }
}

function httpRequest(method, urlString, headers, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    let parsedUrl;
    try {
      parsedUrl = new URL(urlString);
    } catch (e) {
      return reject(new SimilaritySearchError(`Invalid URL: ${urlString}`, null, 'INVALID_URL'));
    }

    const isHttps = parsedUrl.protocol === 'https:';
    const transport = isHttps ? https : http;
    const port = parsedUrl.port
      ? parseInt(parsedUrl.port, 10)
      : isHttps ? 443 : 80;

    const bodyBuffer = body ? Buffer.from(JSON.stringify(body), 'utf8') : null;

    const requestHeaders = Object.assign({}, headers);
    if (bodyBuffer) {
      requestHeaders['Content-Type'] = 'application/json';
      requestHeaders['Content-Length'] = bodyBuffer.length;
    }

    const options = {
      hostname: parsedUrl.hostname,
      port,
      path: parsedUrl.pathname + parsedUrl.search,
      method,
      headers: requestHeaders,
      timeout: timeoutMs,
    };

    const req = transport.request(options, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const rawBody = Buffer.concat(chunks).toString('utf8');
        let parsed = null;
        try {
          parsed = rawBody.length > 0 ? JSON.parse(rawBody) : null;
        } catch (e) {
          return reject(
            new SimilaritySearchError(
              `Failed to parse response JSON: ${rawBody.slice(0, 200)}`,
              res.statusCode,
              'PARSE_ERROR'
            )
          );
        }
        resolve({ statusCode: res.statusCode, headers: res.headers, body: parsed });
      });
      res.on('error', (err) => reject(
        new SimilaritySearchError(`Response stream error: ${err.message}`, null, 'STREAM_ERROR')
      ));
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new SimilaritySearchError(
        `Request timed out after ${timeoutMs}ms`,
        null,
        'TIMEOUT'
      ));
    });

    req.on('error', (err) => reject(
      new SimilaritySearchError(`Network error: ${err.message}`, null, 'NETWORK_ERROR')
    ));

    if (bodyBuffer) req.write(bodyBuffer);
    req.end();
  });
}

function mapHttpErrorToSdkError(statusCode, responseBody, retryAfterHeader) {
  const message = (responseBody && responseBody.detail)
    ? (typeof responseBody.detail === 'string'
        ? responseBody.detail
        : JSON.stringify(responseBody.detail))
    : `HTTP ${statusCode}`;

  if (statusCode === 401 || statusCode === 403) {
    return new SimilaritySearchAuthError(message);
  }
  if (statusCode === 422) {
    return new SimilaritySearchValidationError(message, responseBody && responseBody.detail);
  }
  if (statusCode === 429) {
    const retryAfterMs = retryAfterHeader ? parseInt(retryAfterHeader, 10) * 1000 : null;
    return new SimilaritySearchRateLimitError(retryAfterMs);
  }
  return new SimilaritySearchError(message, statusCode, `HTTP_${statusCode}`);
}

class SimilaritySearchClient {
  constructor(options) {
    if (options === null || options === undefined) {
      options = {};
    }
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError(
        `SimilaritySearchClient constructor expects a plain options object, received ${typeof options}.`
      );
    }

    const apiKey = options.apiKey || process.env.SIMILARITY_SEARCH_API_KEY;
    validateApiKey(apiKey);

    this._apiKey = apiKey;
    this._baseUrl = (options.baseUrl && typeof options.baseUrl === 'string')
      ? options.baseUrl.replace(/\/$/, '')
      : SIMILARITY_SEARCH_API_BASE_URL;
    this._timeoutMs = (typeof options.timeoutMs === 'number' && options.timeoutMs > 0)
      ? options.timeoutMs
      : SIMILARITY_SEARCH_DEFAULT_TIMEOUT_MS;
    this._userAgent = `similarity-search-sdk-js/1.0.0 node/${process.version}`;
  }

  _buildHeaders() {
    return {
      'Authorization': `Bearer ${this._apiKey}`,
      'Accept': 'application/json',
      'User-Agent': this._userAgent,
    };
  }

  async _post(path, body) {
    const url = `${this._baseUrl}${path}`;
    const response = await httpRequest('POST', url, this._buildHeaders(), body, this._timeoutMs);

    if (response.statusCode >= 200 && response.statusCode < 300) {
      return response.body;
    }

    const retryAfter = response.headers && response.headers['retry-after']
      ? response.headers['retry-after']
      : null;
    throw mapHttpErrorToSdkError(response.statusCode, response.body, retryAfter);
  }

  async _get(path, queryParams) {
    let queryString = '';
    if (queryParams && Object.keys(queryParams).length > 0) {
      const params = new URLSearchParams();
      for (const [k, v] of Object.entries(queryParams)) {
        if (v !== undefined && v !== null) params.append(k, String(v));
      }
      queryString = '?' + params.toString();
    }
    const url = `${this._baseUrl}${path}${queryString}`;
    const response = await httpRequest('GET', url, this._buildHeaders(), null, this._timeoutMs);

    if (response.statusCode >= 200 && response.statusCode < 300) {
      return response.body;
    }

    const retryAfter = response.headers && response.headers['retry-after']
      ? response.headers['retry-after']
      : null;
    throw mapHttpErrorToSdkError(response.statusCode, response.body, retryAfter);
  }

  async nmiWeightedSearch(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'nmiWeightedSearch requires a params object with at least { query, corpus }.'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        `nmiWeightedSearch expects a plain object, received ${typeof params}.`
      );
    }

    const { query, corpus, topK, nmiThreshold, includeConfidenceInterval, domain } = params;

    if (query === null || query === undefined) {
      throw new SimilaritySearchValidationError(
        'params.query is required and must not be null or undefined.'
      );
    }
    if (corpus === null || corpus === undefined) {
      throw new SimilaritySearchValidationError(
        'params.corpus is required and must not be null or undefined.'
      );
    }
    if (!Array.isArray(corpus)) {
      throw new SimilaritySearchValidationError(
        `params.corpus must be an array of corpus entries, received ${typeof corpus}.`
      );
    }
    if (corpus.length === 0) {
      throw new SimilaritySearchValidationError(
        'params.corpus must contain at least one entry.'
      );
    }
    if (corpus.length > SIMILARITY_SEARCH_MAX_CORPUS_SIZE) {
      throw new SimilaritySearchValidationError(
        `params.corpus length ${corpus.length} exceeds maximum ${SIMILARITY_SEARCH_MAX_CORPUS_SIZE}. For larger corpora, use the batch endpoint.`
      );
    }

    const queryType = detectPayloadType(query);

    if (queryType === 'vector') {
      validateVector(query, 'params.query');
    } else if (queryType === 'tokenized_text') {
      validateTokenizedText(query, 'params.query');
    } else {
      validateTabularFeatures(query, 'params.query');
    }

    for (let i = 0; i < corpus.length; i++) {
      validateCorpusEntry(corpus[i], i, queryType);
      if (queryType === 'vector') {
        validateVector(corpus[i].payload, `params.corpus[${i}].payload`);
        if (corpus[i].payload.length !== query.length) {
          throw new SimilaritySearchValidationError(
            `params.corpus[${i}].payload dimensionality ${corpus[i].payload.length} does not match query dimensionality ${query.length}.`
          );
        }
      } else if (queryType === 'tokenized_text') {
        validateTokenizedText(corpus[i].payload, `params.corpus[${i}].payload`);
      } else {
        validateTabularFeatures(corpus[i].payload, `params.corpus[${i}].payload`);
      }
    }

    const resolvedTopK = topK !== undefined ? topK : 10;
    if (typeof resolvedTopK !== 'number' || !Number.isInteger(resolvedTopK) || resolvedTopK < 1) {
      throw new SimilaritySearchValidationError(
        `params.topK must be a positive integer, received ${topK}.`
      );
    }
    if (resolvedTopK > SIMILARITY_SEARCH_MAX_TOP_K) {
      throw new SimilaritySearchValidationError(
        `params.topK ${resolvedTopK} exceeds maximum ${SIMILARITY_SEARCH_MAX_TOP_K}.`
      );
    }
    if (resolvedTopK > corpus.length) {
      throw new SimilaritySearchValidationError(
        `params.topK ${resolvedTopK} exceeds corpus size ${corpus.length}.`
      );
    }

    if (nmiThreshold !== undefined) {
      if (typeof nmiThreshold !== 'number' || !isFinite(nmiThreshold) || nmiThreshold < 0 || nmiThreshold > 1) {
        throw new SimilaritySearchValidationError(
          `params.nmiThreshold must be a number in [0, 1], received ${nmiThreshold}.`
        );
      }
    }

    if (includeConfidenceInterval !== undefined && typeof includeConfidenceInterval !== 'boolean') {
      throw new SimilaritySearchValidationError(
        `params.includeConfidenceInterval must be a boolean, received ${typeof includeConfidenceInterval}.`
      );
    }

    if (domain !== undefined && typeof domain !== 'string') {
      throw new SimilaritySearchValidationError(
        `params.domain must be a string, received ${typeof domain}.`
      );
    }

    const requestBody = {
      query: { type: queryType, payload: query },
      corpus: corpus.map((entry) => ({
        id: entry.id,
        payload: entry.payload,
        metadata: entry.metadata || null,
      })),
      top_k: resolvedTopK,
    };

    if (nmiThreshold !== undefined) requestBody.nmi_threshold = nmiThreshold;
    if (includeConfidenceInterval !== undefined) requestBody.include_confidence_interval = includeConfidenceInterval;
    if (domain !== undefined) requestBody.domain = domain;

    return await this._post('/search/nmi-cosine', requestBody);
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        'mainMethod requires a data object. Expected { query, corpus, topK?, nmiThreshold?, includeConfidenceInterval?, domain? }.'
      );
    }
    return this.nmiWeightedSearch(data);
  }

  async batchNmiWeightedSearch(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'batchNmiWeightedSearch requires a params object with { queries, corpus }.'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        `batchNmiWeightedSearch expects a plain object, received ${typeof params}.`
      );
    }

    const { queries, corpus, topK, nmiThreshold, domain } = params;

    if (!Array.isArray(queries) || queries.length === 0) {
      throw new SimilaritySearchValidationError(
        'params.queries must be a non-empty array of query payloads.'
      );
    }
    if (queries.length > 256) {
      throw new SimilaritySearchValidationError(
        `params.queries length ${queries.length} exceeds batch maximum 256.`
      );
    }
    if (!Array.isArray(corpus) || corpus.length === 0) {
      throw new SimilaritySearchValidationError(
        'params.corpus must be a non-empty array of corpus entries.'
      );
    }

    const queryType = detectPayloadType(queries[0]);
    for (let i = 0; i < queries.length; i++) {
      const qt = detectPayloadType(queries[i]);
      if (qt !== queryType) {
        throw new SimilaritySearchValidationError(
          `All queries must use the same payload type. queries[0] is '${queryType}' but queries[${i}] is '${qt}'.`
        );
      }
    }
    for (let i = 0; i < corpus.length; i++) {
      validateCorpusEntry(corpus[i], i, queryType);
    }

    const resolvedTopK = topK !== undefined ? topK : 10;
    if (typeof resolvedTopK !== 'number' || !Number.isInteger(resolvedTopK) || resolvedTopK < 1 || resolvedTopK > SIMILARITY_SEARCH_MAX_TOP_K) {
      throw new SimilaritySearchValidationError(
        `params.topK must be an integer in [1, ${SIMILARITY_SEARCH_MAX_TOP_K}], received ${topK}.`
      );
    }

    const requestBody = {
      queries: queries.map((q) => ({ type: queryType, payload: q })),
      corpus: corpus.map((entry) => ({
        id: entry.id,
        payload: entry.payload,
        metadata: entry.metadata || null,
      })),
      top_k: resolvedTopK,
    };

    if (nmiThreshold !== undefined) requestBody.nmi_threshold = nmiThreshold;
    if (domain !== undefined) requestBody.domain = domain;

    return await this._post('/search/nmi-cosine/batch', requestBody);
  }

  async getSearchQuota() {
    return await this._get('/quota');
  }

  async getDomainNmiPercentiles(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'getDomainNmiPercentiles requires a params object with { domain }.'
      );
    }
    const { domain, percentiles } = params;
    if (!domain || typeof domain !== 'string' || domain.trim().length === 0) {
      throw new SimilaritySearchValidationError(
        'params.domain must be a non-empty string.'
      );
    }
    const queryParams = { domain };
    if (percentiles !== undefined) {
      if (!Array.isArray(percentiles) || percentiles.some((p) => typeof p !== 'number' || p < 0 || p > 100)) {
        throw new SimilaritySearchValidationError(
          'params.percentiles must be an array of numbers in [0, 100].'
        );
      }
      queryParams.percentiles = percentiles.join(',');
    }
    return await this._get('/domains/nmi-percentiles', queryParams);
  }
}

function createSimilaritySearchClient(options) {
  return new SimilaritySearchClient(options || {});
}

let _defaultClient = null;

function _getOrCreateDefaultClient() {
  if (!_defaultClient) {
    const apiKey = process.env.SIMILARITY_SEARCH_API_KEY;
    if (!apiKey) {
      throw new SimilaritySearchAuthError(
        'No default client configured and SIMILARITY_SEARCH_API_KEY environment variable is not set. ' +
        'Either set the env variable or create a client with require("similarity_search_api").createClient({ apiKey: "..." }).'
      );
    }
    _defaultClient = new SimilaritySearchClient({ apiKey });
  }
  return _defaultClient;
}

async function mainMethod(data) {
  return _getOrCreateDefaultClient().mainMethod(data);
}

async function nmiWeightedSearch(params) {
  return _getOrCreateDefaultClient().nmiWeightedSearch(params);
}

async function batchNmiWeightedSearch(params) {
  return _getOrCreateDefaultClient().batchNmiWeightedSearch(params);
}

async function getSearchQuota() {
  return _getOrCreateDefaultClient().getSearchQuota();
}

async function getDomainNmiPercentiles(params) {
  return _getOrCreateDefaultClient().getDomainNmiPercentiles(params);
}

module.exports = {
  createClient: createSimilaritySearchClient,
  SimilaritySearchClient,
  mainMethod,
  nmiWeightedSearch,
  batchNmiWeightedSearch,
  getSearchQuota,
  getDomainNmiPercentiles,
  SimilaritySearchError,
  SimilaritySearchAuthError,
  SimilaritySearchValidationError,
  SimilaritySearchRateLimitError,
  SIMILARITY_SEARCH_MAX_CORPUS_SIZE,
  SIMILARITY_SEARCH_MAX_VECTOR_DIM,
  SIMILARITY_SEARCH_MAX_TOP_K,
};
```