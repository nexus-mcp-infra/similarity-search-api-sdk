
```javascript
'use strict';

const https = require('https');
const http = require('http');
const { URL } = require('url');

const SIMILARITY_SEARCH_API_VERSION = '1.0.0';
const DEFAULT_BASE_URL = 'https://api.similaritysearch.io/v1';
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_DELAY_MS = 500;

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
    super('Rate limit exceeded. Retry after ' + retryAfterSeconds + ' seconds.', 429, null);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

function isFiniteNumber(val) {
  return typeof val === 'number' && isFinite(val);
}

function isNonEmptyArray(val) {
  return Array.isArray(val) && val.length > 0;
}

function isNumericVector(arr) {
  if (!isNonEmptyArray(arr)) return false;
  return arr.every(isFiniteNumber);
}

function isUniformDimension(vectors) {
  const dim = vectors[0].length;
  return vectors.every(function (v) { return v.length === dim; });
}

function validateVectorPayload(corpus, query, options) {
  if (!isNonEmptyArray(corpus)) {
    throw new SimilaritySearchValidationError(
      'corpus must be a non-empty array of numeric vectors.'
    );
  }
  if (corpus.length > 10000) {
    throw new SimilaritySearchValidationError(
      'corpus exceeds maximum allowed size of 10000 vectors per request.'
    );
  }
  for (var i = 0; i < corpus.length; i++) {
    if (!isNumericVector(corpus[i])) {
      throw new SimilaritySearchValidationError(
        'corpus[' + i + '] must be a non-empty array of finite numbers.'
      );
    }
    if (corpus[i].length > 4096) {
      throw new SimilaritySearchValidationError(
        'corpus[' + i + '] exceeds maximum vector dimension of 4096.'
      );
    }
  }
  if (!isUniformDimension(corpus)) {
    throw new SimilaritySearchValidationError(
      'All vectors in corpus must have the same dimension.'
    );
  }
  if (!isNumericVector(query)) {
    throw new SimilaritySearchValidationError(
      'query must be a non-empty array of finite numbers.'
    );
  }
  if (query.length !== corpus[0].length) {
    throw new SimilaritySearchValidationError(
      'query dimension (' + query.length + ') must match corpus dimension (' + corpus[0].length + ').'
    );
  }
  if (options !== undefined && options !== null) {
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError('options must be a plain object if provided.');
    }
    if (options.top_k !== undefined) {
      if (!Number.isInteger(options.top_k) || options.top_k < 1 || options.top_k > 1000) {
        throw new SimilaritySearchValidationError('options.top_k must be an integer between 1 and 1000.');
      }
    }
    if (options.nmi_weight !== undefined) {
      if (!isFiniteNumber(options.nmi_weight) || options.nmi_weight < 0 || options.nmi_weight > 1) {
        throw new SimilaritySearchValidationError('options.nmi_weight must be a number between 0 and 1.');
      }
    }
    if (options.entropy_bins !== undefined) {
      if (!Number.isInteger(options.entropy_bins) || options.entropy_bins < 2 || options.entropy_bins > 256) {
        throw new SimilaritySearchValidationError('options.entropy_bins must be an integer between 2 and 256.');
      }
    }
  }
}

function validateTextPayload(corpus_texts, query_text, options) {
  if (!isNonEmptyArray(corpus_texts)) {
    throw new SimilaritySearchValidationError(
      'corpus_texts must be a non-empty array of strings.'
    );
  }
  if (corpus_texts.length > 5000) {
    throw new SimilaritySearchValidationError(
      'corpus_texts exceeds maximum allowed size of 5000 items per request.'
    );
  }
  for (var i = 0; i < corpus_texts.length; i++) {
    if (typeof corpus_texts[i] !== 'string' || corpus_texts[i].trim().length === 0) {
      throw new SimilaritySearchValidationError(
        'corpus_texts[' + i + '] must be a non-empty string.'
      );
    }
    if (corpus_texts[i].length > 8192) {
      throw new SimilaritySearchValidationError(
        'corpus_texts[' + i + '] exceeds maximum length of 8192 characters.'
      );
    }
  }
  if (typeof query_text !== 'string' || query_text.trim().length === 0) {
    throw new SimilaritySearchValidationError(
      'query_text must be a non-empty string.'
    );
  }
  if (query_text.length > 8192) {
    throw new SimilaritySearchValidationError(
      'query_text exceeds maximum length of 8192 characters.'
    );
  }
  if (options !== undefined && options !== null) {
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError('options must be a plain object if provided.');
    }
    if (options.top_k !== undefined) {
      if (!Number.isInteger(options.top_k) || options.top_k < 1 || options.top_k > 1000) {
        throw new SimilaritySearchValidationError('options.top_k must be an integer between 1 and 1000.');
      }
    }
  }
}

function validateBatchPayload(queries, options) {
  if (!isNonEmptyArray(queries)) {
    throw new SimilaritySearchValidationError(
      'queries must be a non-empty array of query objects.'
    );
  }
  if (queries.length > 50) {
    throw new SimilaritySearchValidationError(
      'queries exceeds maximum batch size of 50 per request.'
    );
  }
  for (var i = 0; i < queries.length; i++) {
    var q = queries[i];
    if (typeof q !== 'object' || q === null || Array.isArray(q)) {
      throw new SimilaritySearchValidationError('queries[' + i + '] must be a plain object.');
    }
    if (!isNonEmptyArray(q.corpus)) {
      throw new SimilaritySearchValidationError('queries[' + i + '].corpus must be a non-empty array.');
    }
    if (!isNumericVector(q.query)) {
      throw new SimilaritySearchValidationError('queries[' + i + '].query must be a non-empty array of finite numbers.');
    }
  }
}

function httpRequest(method, url, apiKey, body, timeoutMs) {
  return new Promise(function (resolve, reject) {
    var parsedUrl;
    try {
      parsedUrl = new URL(url);
    } catch (e) {
      return reject(new SimilaritySearchError('Invalid base URL: ' + url));
    }

    var isHttps = parsedUrl.protocol === 'https:';
    var transport = isHttps ? https : http;
    var port = parsedUrl.port
      ? parseInt(parsedUrl.port, 10)
      : (isHttps ? 443 : 80);

    var bodyString = body ? JSON.stringify(body) : '';
    var headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': 'Bearer ' + apiKey,
      'X-SDK-Version': SIMILARITY_SEARCH_API_VERSION,
      'X-SDK-Language': 'javascript-node',
      'Content-Length': Buffer.byteLength(bodyString)
    };

    var options = {
      hostname: parsedUrl.hostname,
      port: port,
      path: parsedUrl.pathname + (parsedUrl.search || ''),
      method: method,
      headers: headers
    };

    var req = transport.request(options, function (res) {
      var chunks = [];
      res.on('data', function (chunk) { chunks.push(chunk); });
      res.on('end', function () {
        var rawBody = Buffer.concat(chunks).toString('utf8');
        var parsed = null;
        try {
          parsed = JSON.parse(rawBody);
        } catch (e) {
          parsed = { raw: rawBody };
        }
        resolve({ statusCode: res.statusCode, headers: res.headers, body: parsed });
      });
      res.on('error', reject);
    });

    req.setTimeout(timeoutMs, function () {
      req.destroy();
      reject(new SimilaritySearchError('Request timed out after ' + timeoutMs + 'ms.'));
    });

    req.on('error', function (err) {
      reject(new SimilaritySearchError('Network error: ' + err.message));
    });

    if (bodyString) {
      req.write(bodyString);
    }
    req.end();
  });
}

function sleep(ms) {
  return new Promise(function (resolve) { setTimeout(resolve, ms); });
}

function buildUrl(baseUrl, path) {
  return baseUrl.replace(/\/$/, '') + path;
}

function SimilaritySearchClient(config) {
  if (!config || typeof config !== 'object') {
    throw new SimilaritySearchValidationError(
      'SimilaritySearchClient requires a config object with at least { apiKey }.'
    );
  }
  if (typeof config.apiKey !== 'string' || config.apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError(
      'config.apiKey must be a non-empty string. Obtain your key at https://api.similaritysearch.io.'
    );
  }

  this._apiKey = config.apiKey.trim();
  this._baseUrl = typeof config.baseUrl === 'string' && config.baseUrl.trim().length > 0
    ? config.baseUrl.trim()
    : DEFAULT_BASE_URL;
  this._timeoutMs = Number.isInteger(config.timeoutMs) && config.timeoutMs > 0
    ? config.timeoutMs
    : DEFAULT_TIMEOUT_MS;
  this._maxRetries = Number.isInteger(config.maxRetries) && config.maxRetries >= 0
    ? config.maxRetries
    : DEFAULT_MAX_RETRIES;
  this._retryDelayMs = Number.isInteger(config.retryDelayMs) && config.retryDelayMs >= 0
    ? config.retryDelayMs
    : DEFAULT_RETRY_DELAY_MS;
}

SimilaritySearchClient.prototype._executeWithRetry = async function (method, path, body) {
  var url = buildUrl(this._baseUrl, path);
  var lastError = null;
  var attempt = 0;

  while (attempt <= this._maxRetries) {
    var response;
    try {
      response = await httpRequest(method, url, this._apiKey, body, this._timeoutMs);
    } catch (networkErr) {
      lastError = networkErr;
      attempt++;
      if (attempt <= this._maxRetries) {
        await sleep(this._retryDelayMs * attempt);
      }
      continue;
    }

    var status = response.statusCode;
    var respBody = response.body;

    if (status === 200 || status === 201) {
      return respBody;
    }

    if (status === 401) {
      throw new SimilaritySearchAuthError(
        'Authentication failed. Check your API key. Server message: ' +
        (respBody && respBody.detail ? respBody.detail : JSON.stringify(respBody))
      );
    }

    if (status === 422) {
      throw new SimilaritySearchValidationError(
        'Validation error from server: ' +
        (respBody && respBody.detail ? JSON.stringify(respBody.detail) : JSON.stringify(respBody))
      );
    }

    if (status === 429) {
      var retryAfter = response.headers && response.headers['retry-after']
        ? parseInt(response.headers['retry-after'], 10)
        : 60;
      throw new SimilaritySearchRateLimitError(retryAfter);
    }

    if (status >= 500 && attempt < this._maxRetries) {
      lastError = new SimilaritySearchError(
        'Server error ' + status + ': ' + JSON.stringify(respBody),
        status,
        respBody
      );
      attempt++;
      await sleep(this._retryDelayMs * attempt);
      continue;
    }

    throw new SimilaritySearchError(
      'Request failed with status ' + status + ': ' +
      (respBody && respBody.detail ? JSON.stringify(respBody.detail) : JSON.stringify(respBody)),
      status,
      respBody
    );
  }

  throw lastError || new SimilaritySearchError('Max retries exceeded without a successful response.');
};

/**
 * nmiCosineRankedSearch
 *
 * Sends a corpus of dense vectors and a query vector to the Similarity Search API.
 * The API computes NMI-cosine fused scores with adaptive entropy-calibrated weighting
 * derived from the full corpus distribution sent in this request.
 *
 * @param {Array<Array<number>>} corpus - Array of numeric vectors forming the search space.
 *   Min: 1 item, Max: 10000 items. All vectors must share the same dimension (max 4096).
 * @param {Array<number>} query - Query vector. Must match corpus dimension.
 * @param {Object} [options] - Optional tuning parameters.
 *   @param {number} [options.top_k=10] - Number of ranked results to return (1-1000).
 *   @param {number} [options.nmi_weight] - Override adaptive NMI weight (0-1). Omit to use entropy-derived value.
 *   @param {number} [options.entropy_bins=32] - Histogram bins for entropy estimation (2-256).
 *   @param {Array<string>} [options.ids] - Optional string IDs for corpus items, returned in results.
 *
 * @returns {Promise<Object>} Ranked results with fused scores, individual cosine/NMI scores,
 *   entropy diagnostics, and request metadata.
 *
 * Use this when: you have pre-computed dense vectors and want stateless ranked similarity.
 * Do NOT use this when: your corpus changes per item (use nmiCosineRankedTextSearch for on-the-fly encoding),
 *   or you need ANN approximate search over millions of vectors (use a dedicated vector DB instead).
 */
SimilaritySearchClient.prototype.nmiCosineRankedSearch = async function (corpus, query, options) {
  if (corpus === null || corpus === undefined) {
    throw new SimilaritySearchValidationError('corpus must not be null or undefined.');
  }
  if (query === null || query === undefined) {
    throw new SimilaritySearchValidationError('query must not be null or undefined.');
  }

  validateVectorPayload(corpus, query, options);

  var payload = {
    corpus: corpus,
    query: query
  };

  if (options && typeof options === 'object') {
    if (options.top_k !== undefined) payload.top_k = options.top_k;
    if (options.nmi_weight !== undefined) payload.nmi_weight = options.nmi_weight;
    if (options.entropy_bins !== undefined) payload.entropy_bins = options.entropy_bins;
    if (Array.isArray(options.ids)) {
      if (options.ids.length !== corpus.length) {
        throw new SimilaritySearchValidationError(
          'options.ids length (' + options.ids.length + ') must match corpus length (' + corpus.length + ').'
        );
      }
      payload.ids = options.ids;
    }
  }

  return this._executeWithRetry('POST', '/search/vectors', payload);
};

/**
 * nmiCosineRankedTextSearch
 *
 * Sends raw text strings for corpus and query. The API encodes them server-side,
 * then applies NMI-cosine fusion with entropy-adaptive weighting on the resulting embeddings.
 *
 * @param {Array<string>} corpus_texts - Array of strings to search within.
 *   Min: 1 item, Max: 5000 items. Each string max 8192 characters.
 * @param {string} query_text - Query string. Max 8192 characters.
 * @param {Object} [options] - Optional tuning parameters.
 *   @param {number} [options.top_k=10] - Number of results (1-1000).
 *   @param {string} [options.embedding_model='default'] - Server-side embedding model identifier.
 *
 * @returns {Promise<Object>} Ranked results with fused NMI-cosine scores, matched text snippets,
 *   entropy diagnostics, and per-item cosine/NMI breakdowns.
 *
 * Use this when: you have raw text and want zero-infrastructure semantic search in one HTTP call.
 * Do NOT use this when: you already have dense vectors (use nmiCosineRankedSearch to avoid re-encoding overhead),
 *   or when corpus exceeds 5000 items per request (split into batches or use a persistent store).
 */
SimilaritySearchClient.prototype.nmiCosineRankedTextSearch = async function (corpus_texts, query_text, options) {
  if (corpus_texts === null || corpus_texts === undefined) {
    throw new SimilaritySearchValidationError('corpus_texts must not be null or undefined.');
  }
  if (query_text === null || query_text === undefined) {
    throw new SimilaritySearchValidationError('query_text must not be null or undefined.');
  }

  validateTextPayload(corpus_texts, query_text, options);

  var payload = {
    corpus_texts: corpus_texts,
    query_text: query_text
  };

  if (options && typeof options === 'object') {
    if (options.top_k !== undefined) payload.top_k = options.top_k;
    if (typeof options.embedding_model === 'string' && options.embedding_model.trim().length > 0) {
      payload.embedding_model = options.embedding_model.trim();
    }
  }

  return this._executeWithRetry('POST', '/search/texts', payload);
};

/**
 * batchNmiCosineRankedSearch
 *
 * Executes multiple independent nmiCosineRankedSearch operations in a single HTTP call.
 * Each query in the batch is an independent corpus+query pair; results are returned
 * in the same order as the input queries array.
 *
 * @param {Array<Object>} queries - Array of query objects.
 *   Each object must have:
 *     - corpus {Array<Array<number>>}: vector corpus for this sub-query.
 *     - query {Array<number>}: query vector.
 *   Each object may optionally have:
 *     - top_k {number}: per-subquery result count override (1-1000).
 *     - ids {Array<string>}: corpus item IDs for this sub-query.
 *   Max 50 sub-queries per batch call.
 * @param {Object} [options] - Batch-level options.
 *   @param {number} [options.default_top_k=10] - Default top_k applied to sub-queries lacking their own.
 *
 * @returns {Promise<Object>} Object with 'results' array, each element being the ranked output
 *   for the corresponding input query, plus batch-level metadata (total_latency_ms, per_query_latency_ms).
 *
 * Use this when: you need to run multiple independent searches and want to amortize HTTP overhead.
 * Do NOT use this when: sub-queries share the same corpus (use nmiCosineRankedSearch per query instead
 *   to avoid resending the same corpus data multiple times).
 */
SimilaritySearchClient.prototype.batchNmiCosineRankedSearch = async function (queries, options) {
  if (queries === null || queries === undefined) {
    throw new SimilaritySearchValidationError('queries must not be null or undefined.');
  }

  validateBatchPayload(queries);

  var payload = { queries: queries };

  if (options && typeof options === 'object') {
    if (options.default_top_k !== undefined) {
      if (!Number.isInteger(options.default_top_k) || options.default_top_k < 1 || options.default_top_k > 1000) {
        throw new SimilaritySearchValidationError('options.default_top_k must be an integer between 1 and 1000.');
      }
      payload.default_top_k = options.default_top_k;
    }
  }

  return this._executeWithRetry('POST', '/search/batch', payload);
};

/**
 * entropyDiagnostics
 *
 * Computes entropy and distribution diagnostics for a given corpus without performing search.
 * Returns marginal entropy H(corpus), per-dimension entropy profile, uniformity score,
 * and the adaptive NMI weight w_nmi = H(corpus) / (H(corpus) + baseline_entropy) that would
 * be applied in a search call against this corpus.
 *
 * @param {Array<Array<number>>} corpus - Numeric vector corpus to analyze.
 *   Min: 2 items, Max: 10000 items.
 * @param {Object} [options]
 *   @param {number} [options.entropy_bins=32] - Histogram bins for entropy estimation (2-256).
 *   @param {number} [options.baseline_entropy] - Custom baseline entropy override (must be > 0).
 *
 * @returns {Promise<Object>} Entropy diagnostics including H_corpus, w_nmi_adaptive,
 *   per_dimension_entropy, uniformity_score, and recommended_entropy_bins.
 *
 * Use this when: you want to understand how the adaptive NMI weight will behave before committing
 *   to a search call, or to diagnose why results seem too conservative or too aggressive.
 * Do NOT use this when: you just want search results — the overhead of a separate diagnostics call
 *   is unnecessary since search responses already include entropy metadata.
 */
SimilaritySearchClient.prototype.entropyDiagnostics = async function (corpus, options) {
  if (corpus === null || corpus === undefined) {
    throw new SimilaritySearchValidationError('corpus must not be null or undefined.');
  }
  if (!isNonEmptyArray(corpus) || corpus.length < 2) {
    throw new SimilaritySearchValidationError(
      'corpus must contain at least 2 vectors for entropy computation.'
    );
  }
  if (corpus.length > 10000) {
    throw new SimilaritySearchValidationError(
      'corpus exceeds maximum allowed size of 10000 vectors.'
    );
  }
  for (var i = 0; i < corpus.length; i++) {
    if (!isNumericVector(corpus[i])) {
      throw new SimilaritySearchValidationError(
        'corpus[' + i + '] must be a non-empty array of finite numbers.'
      );
    }
  }
  if (!isUniformDimension(corpus)) {
    throw new SimilaritySearchValidationError(
      'All vectors in corpus must have the same dimension.'
    );
  }

  var payload = { corpus: corpus };

  if (options && typeof options === 'object') {
    if (options.entropy_bins !== undefined) {
      if (!Number.isInteger(options.entropy_bins) || options.entropy_bins < 2 || options.entropy_bins > 256) {
        throw new SimilaritySearchValidationError('options.entropy_bins must be an integer between 2 and 256.');
      }
      payload.entropy_bins = options.entropy_bins;
    }
    if (options.baseline_entropy !== undefined) {
      if (!isFiniteNumber(options.baseline_entropy) || options.baseline_entropy <= 0) {
        throw new SimilaritySearchValidationError('options.baseline_entropy must be a positive finite number.');
      }
      payload.baseline_entropy = options.baseline_entropy;
    }
  }

  return this._executeWithRetry('POST', '/diagnostics/entropy', payload);
};

/**
 * mainMethod
 *
 * Unified entry point matching the documented client.mainMethod(data) contract.
 * Dispatches to the appropriate specialized method based on the shape of `data`.
 *
 * Dispatch rules (checked in order):
 *   1. data.corpus_texts && data.query_text  -> nmiCosineRankedTextSearch
 *   2. data.queries (array)                  -> batchNmiCosineRankedSearch
 *   3. data.corpus && data.query === undefined && data.diagnostics_only -> entropyDiagnostics
 *   4. data.corpus && data.query             -> nmiCosineRankedSearch
 *
 * @param {Object} data - Must be a non-null plain object. Required shape depends on dispatch target above.
 * @returns {Promise<Object>} Result from the dispatched method.
 */
SimilaritySearchClient.prototype.mainMethod = async function (data) {
  if (data === null || data === undefined) {
    throw new SimilaritySearchValidationError(
      'mainMethod requires a non-null data object. ' +
      'Provide { corpus, query } for vector search, { corpus_texts, query_text } for text search, ' +
      '{ queries } for batch search, or { corpus, diagnostics_only: true } for entropy diagnostics.'
    );
  }
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new SimilaritySearchValidationError(
      'mainMethod data must be a plain object, got: ' + typeof data
    );
  }

  if (data.corpus_texts !== undefined && data.query_text !== undefined) {
    return this.nmiCosineRankedTextSearch(data.corpus_texts, data.query_text, data.options);
  }

  if (data.queries !== undefined) {
    return this.batchNmiCosineRankedSearch(data.queries, data.options);
  }

  if (data.corpus !== undefined && data.diagnostics_only === true) {
    return this.entropyDiagnostics(data.corpus, data.options);
  }

  if (data.corpus !== undefined && data.query !== undefined) {
    return this.nmiCosineRankedSearch(data.corpus, data.query, data.options);
  }

  throw new SimilaritySearchValidationError(
    'mainMethod could not dispatch: data must contain one of: ' +
    '(corpus + query), (corpus_texts + query_text), (queries), or (corpus + diagnostics_only:true). ' +
    'Received keys: [' + Object.keys(data).join(', ') + '].'
  );
};

function createClient(config) {
  if (!config || typeof config !== 'object') {
    throw new SimilaritySearchValidationError(
      'createClient requires a config object with at least { apiKey: "your-key" }.'
    );
  }
  return new SimilaritySearchClient(config);
}

module.exports = createClient;
module.exports.createClient = createClient;
module.exports.SimilaritySearchClient = SimilaritySearchClient;
module.exports.SimilaritySearchError = SimilaritySearchError;
module.exports.SimilaritySearchAuthError = SimilaritySearchAuthError;
module.exports.SimilaritySearchValidationError = SimilaritySearchValidationError;
module.exports.SimilaritySearchRateLimitError = SimilaritySearchRateLimitError;
module.exports.SDK_VERSION = SIMILARITY_SEARCH_API_VERSION;
```