
```javascript
'use strict';

const https = require('https');
const http = require('http');

const SIMILARITY_SEARCH_DEFAULT_BASE_URL = 'https://api.similaritysearch.io/v1';
const SIMILARITY_SEARCH_DEFAULT_TIMEOUT_MS = 30000;
const SIMILARITY_SEARCH_DEFAULT_ALPHA = 0.6;
const SIMILARITY_SEARCH_MAX_ITEMS = 10000;
const SIMILARITY_SEARCH_MIN_ALPHA = 0.0;
const SIMILARITY_SEARCH_MAX_ALPHA = 1.0;

class SimilaritySearchAuthError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SimilaritySearchAuthError';
    this.statusCode = 401;
  }
}

class SimilaritySearchValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'SimilaritySearchValidationError';
    this.statusCode = 422;
    this.field = field || null;
  }
}

class SimilaritySearchRateLimitError extends Error {
  constructor(message, retryAfterSeconds) {
    super(message);
    this.name = 'SimilaritySearchRateLimitError';
    this.statusCode = 429;
    this.retryAfterSeconds = retryAfterSeconds || null;
  }
}

class SimilaritySearchAPIError extends Error {
  constructor(message, statusCode, responseBody) {
    super(message);
    this.name = 'SimilaritySearchAPIError';
    this.statusCode = statusCode;
    this.responseBody = responseBody || null;
  }
}

function assertNonEmptyString(value, fieldName) {
  if (value === null || value === undefined) {
    throw new SimilaritySearchValidationError(
      `${fieldName} is required and cannot be null or undefined`,
      fieldName
    );
  }
  if (typeof value !== 'string') {
    throw new SimilaritySearchValidationError(
      `${fieldName} must be a string, received ${typeof value}`,
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

function assertNonEmptyArray(value, fieldName) {
  if (value === null || value === undefined) {
    throw new SimilaritySearchValidationError(
      `${fieldName} is required and cannot be null or undefined`,
      fieldName
    );
  }
  if (!Array.isArray(value)) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must be an array, received ${typeof value}`,
      fieldName
    );
  }
  if (value.length === 0) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must contain at least one item`,
      fieldName
    );
  }
}

function assertPlainObject(value, fieldName) {
  if (value === null || value === undefined) {
    throw new SimilaritySearchValidationError(
      `${fieldName} is required and cannot be null or undefined`,
      fieldName
    );
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new SimilaritySearchValidationError(
      `${fieldName} must be a plain object`,
      fieldName
    );
  }
}

function validateAlpha(alpha) {
  if (typeof alpha !== 'number' || isNaN(alpha)) {
    throw new SimilaritySearchValidationError(
      `alpha must be a number between ${SIMILARITY_SEARCH_MIN_ALPHA} and ${SIMILARITY_SEARCH_MAX_ALPHA}`,
      'alpha'
    );
  }
  if (alpha < SIMILARITY_SEARCH_MIN_ALPHA || alpha > SIMILARITY_SEARCH_MAX_ALPHA) {
    throw new SimilaritySearchValidationError(
      `alpha must be between ${SIMILARITY_SEARCH_MIN_ALPHA} and ${SIMILARITY_SEARCH_MAX_ALPHA}, received ${alpha}`,
      'alpha'
    );
  }
}

function validateTopK(topK) {
  if (typeof topK !== 'number' || !Number.isInteger(topK)) {
    throw new SimilaritySearchValidationError('top_k must be a positive integer', 'top_k');
  }
  if (topK < 1) {
    throw new SimilaritySearchValidationError('top_k must be at least 1', 'top_k');
  }
}

function validateCandidateItems(items) {
  assertNonEmptyArray(items, 'candidate_items');
  if (items.length > SIMILARITY_SEARCH_MAX_ITEMS) {
    throw new SimilaritySearchValidationError(
      `candidate_items exceeds maximum of ${SIMILARITY_SEARCH_MAX_ITEMS} items, received ${items.length}`,
      'candidate_items'
    );
  }
  items.forEach((item, index) => {
    if (item === null || item === undefined || typeof item !== 'object' || Array.isArray(item)) {
      throw new SimilaritySearchValidationError(
        `candidate_items[${index}] must be a plain object`,
        `candidate_items[${index}]`
      );
    }
    if (Object.keys(item).length === 0) {
      throw new SimilaritySearchValidationError(
        `candidate_items[${index}] cannot be an empty object`,
        `candidate_items[${index}]`
      );
    }
  });
}

function rawHttpRequest(options, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    const transport = options.protocol === 'http:' ? http : https;
    const bodyBuffer = Buffer.from(JSON.stringify(body), 'utf8');

    const reqOptions = {
      hostname: options.hostname,
      port: options.port,
      path: options.path,
      method: options.method || 'POST',
      headers: Object.assign({}, options.headers, {
        'Content-Type': 'application/json',
        'Content-Length': bodyBuffer.length,
      }),
    };

    const req = transport.request(reqOptions, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const rawBody = Buffer.concat(chunks).toString('utf8');
        let parsed = null;
        try {
          parsed = JSON.parse(rawBody);
        } catch (_) {
          parsed = { detail: rawBody };
        }
        resolve({ statusCode: res.statusCode, headers: res.headers, body: parsed });
      });
      res.on('error', reject);
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new SimilaritySearchAPIError(
        `Request timed out after ${timeoutMs}ms`,
        408,
        null
      ));
    });

    req.on('error', (err) => {
      reject(new SimilaritySearchAPIError(
        `Network error: ${err.message}`,
        0,
        null
      ));
    });

    req.write(bodyBuffer);
    req.end();
  });
}

function parseBaseUrl(baseUrl) {
  assertNonEmptyString(baseUrl, 'baseUrl');
  let url;
  try {
    url = new URL(baseUrl);
  } catch (_) {
    throw new SimilaritySearchValidationError(
      `baseUrl is not a valid URL: ${baseUrl}`,
      'baseUrl'
    );
  }
  return url;
}

function handleHttpResponse(response) {
  const { statusCode, headers, body } = response;

  if (statusCode === 401 || statusCode === 403) {
    throw new SimilaritySearchAuthError(
      body && body.detail
        ? body.detail
        : 'Authentication failed. Verify your API key is correct and has not expired.'
    );
  }

  if (statusCode === 429) {
    const retryAfter = headers['retry-after']
      ? parseInt(headers['retry-after'], 10)
      : null;
    throw new SimilaritySearchRateLimitError(
      body && body.detail
        ? body.detail
        : 'Rate limit exceeded.',
      retryAfter
    );
  }

  if (statusCode === 422) {
    const detail =
      body && body.detail
        ? JSON.stringify(body.detail)
        : 'Unprocessable entity: the server rejected the request payload.';
    throw new SimilaritySearchValidationError(detail, null);
  }

  if (statusCode < 200 || statusCode >= 300) {
    throw new SimilaritySearchAPIError(
      body && body.detail
        ? body.detail
        : `Unexpected HTTP ${statusCode} from Similarity Search API.`,
      statusCode,
      body
    );
  }

  return body;
}

class SimilaritySearchClient {
  constructor(options) {
    if (options === null || options === undefined) {
      throw new SimilaritySearchValidationError(
        'SimilaritySearchClient requires an options object with at least { apiKey }',
        'options'
      );
    }
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new SimilaritySearchValidationError(
        'options must be a plain object',
        'options'
      );
    }

    assertNonEmptyString(options.apiKey, 'options.apiKey');

    this._apiKey = options.apiKey;
    this._baseUrl = parseBaseUrl(options.baseUrl || SIMILARITY_SEARCH_DEFAULT_BASE_URL);
    this._timeoutMs =
      typeof options.timeoutMs === 'number' && options.timeoutMs > 0
        ? options.timeoutMs
        : SIMILARITY_SEARCH_DEFAULT_TIMEOUT_MS;
    this._defaultAlpha =
      typeof options.defaultAlpha === 'number'
        ? options.defaultAlpha
        : SIMILARITY_SEARCH_DEFAULT_ALPHA;

    validateAlpha(this._defaultAlpha);
  }

  _buildRequestOptions(path) {
    return {
      protocol: this._baseUrl.protocol,
      hostname: this._baseUrl.hostname,
      port: this._baseUrl.port || (this._baseUrl.protocol === 'https:' ? 443 : 80),
      path: this._baseUrl.pathname.replace(/\/$/, '') + path,
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this._apiKey}`,
        'User-Agent': 'similarity-search-sdk-js/1.0.0',
        'Accept': 'application/json',
      },
    };
  }

  async rankByCombinedNMICosine(query, candidateItems, options) {
    assertPlainObject(query, 'query');
    if (Object.keys(query).length === 0) {
      throw new SimilaritySearchValidationError(
        'query cannot be an empty object — it must contain at least one feature field',
        'query'
      );
    }

    validateCandidateItems(candidateItems);

    const callOptions = options || {};
    const alpha =
      typeof callOptions.alpha === 'number'
        ? callOptions.alpha
        : this._defaultAlpha;

    validateAlpha(alpha);

    const topK =
      typeof callOptions.topK === 'number'
        ? callOptions.topK
        : 10;

    validateTopK(topK);

    const payload = {
      query,
      candidate_items: candidateItems,
      alpha,
      top_k: topK,
    };

    if (callOptions.categoricalFields !== undefined) {
      if (!Array.isArray(callOptions.categoricalFields)) {
        throw new SimilaritySearchValidationError(
          'categoricalFields must be an array of field name strings',
          'categoricalFields'
        );
      }
      callOptions.categoricalFields.forEach((f, i) => {
        if (typeof f !== 'string' || f.trim().length === 0) {
          throw new SimilaritySearchValidationError(
            `categoricalFields[${i}] must be a non-empty string`,
            `categoricalFields[${i}]`
          );
        }
      });
      payload.categorical_fields = callOptions.categoricalFields;
    }

    if (callOptions.numericFields !== undefined) {
      if (!Array.isArray(callOptions.numericFields)) {
        throw new SimilaritySearchValidationError(
          'numericFields must be an array of field name strings',
          'numericFields'
        );
      }
      payload.numeric_fields = callOptions.numericFields;
    }

    const reqOptions = this._buildRequestOptions('/rank');
    const response = await rawHttpRequest(reqOptions, payload, this._timeoutMs);
    const body = handleHttpResponse(response);

    return {
      results: body.results,
      meta: {
        alpha_used: body.meta && body.meta.alpha_used !== undefined ? body.meta.alpha_used : alpha,
        nmi_weight: body.meta && body.meta.nmi_weight !== undefined ? body.meta.nmi_weight : parseFloat((1 - alpha).toFixed(4)),
        cosine_weight: body.meta && body.meta.cosine_weight !== undefined ? body.meta.cosine_weight : parseFloat(alpha.toFixed(4)),
        total_candidates: body.meta && body.meta.total_candidates !== undefined ? body.meta.total_candidates : candidateItems.length,
        returned_count: body.results ? body.results.length : 0,
        latency_ms: body.meta && body.meta.latency_ms !== undefined ? body.meta.latency_ms : null,
      },
    };
  }

  async computePairwiseNMI(itemA, itemB, options) {
    assertPlainObject(itemA, 'itemA');
    assertPlainObject(itemB, 'itemB');

    if (Object.keys(itemA).length === 0) {
      throw new SimilaritySearchValidationError(
        'itemA cannot be an empty object',
        'itemA'
      );
    }
    if (Object.keys(itemB).length === 0) {
      throw new SimilaritySearchValidationError(
        'itemB cannot be an empty object',
        'itemB'
      );
    }

    const callOptions = options || {};
    const payload = { item_a: itemA, item_b: itemB };

    if (callOptions.categoricalFields !== undefined) {
      if (!Array.isArray(callOptions.categoricalFields)) {
        throw new SimilaritySearchValidationError(
          'categoricalFields must be an array of field name strings',
          'categoricalFields'
        );
      }
      payload.categorical_fields = callOptions.categoricalFields;
    }

    const reqOptions = this._buildRequestOptions('/nmi');
    const response = await rawHttpRequest(reqOptions, payload, this._timeoutMs);
    const body = handleHttpResponse(response);

    return {
      nmi_score: body.nmi_score,
      cosine_score: body.cosine_score,
      fields_used: body.fields_used || null,
    };
  }

  async batchRankByCombinedNMICosine(queries, candidateItems, options) {
    assertNonEmptyArray(queries, 'queries');
    if (queries.length > 100) {
      throw new SimilaritySearchValidationError(
        'batchRankByCombinedNMICosine accepts at most 100 queries per call',
        'queries'
      );
    }

    queries.forEach((q, i) => {
      if (q === null || q === undefined || typeof q !== 'object' || Array.isArray(q)) {
        throw new SimilaritySearchValidationError(
          `queries[${i}] must be a plain object`,
          `queries[${i}]`
        );
      }
      if (Object.keys(q).length === 0) {
        throw new SimilaritySearchValidationError(
          `queries[${i}] cannot be an empty object`,
          `queries[${i}]`
        );
      }
    });

    validateCandidateItems(candidateItems);

    const callOptions = options || {};
    const alpha =
      typeof callOptions.alpha === 'number'
        ? callOptions.alpha
        : this._defaultAlpha;

    validateAlpha(alpha);

    const topK =
      typeof callOptions.topK === 'number'
        ? callOptions.topK
        : 10;

    validateTopK(topK);

    const payload = {
      queries,
      candidate_items: candidateItems,
      alpha,
      top_k: topK,
    };

    if (callOptions.categoricalFields !== undefined) {
      if (!Array.isArray(callOptions.categoricalFields)) {
        throw new SimilaritySearchValidationError(
          'categoricalFields must be an array of field name strings',
          'categoricalFields'
        );
      }
      payload.categorical_fields = callOptions.categoricalFields;
    }

    if (callOptions.numericFields !== undefined) {
      if (!Array.isArray(callOptions.numericFields)) {
        throw new SimilaritySearchValidationError(
          'numericFields must be an array of field name strings',
          'numericFields'
        );
      }
      payload.numeric_fields = callOptions.numericFields;
    }

    const reqOptions = this._buildRequestOptions('/batch-rank');
    const response = await rawHttpRequest(reqOptions, payload, this._timeoutMs);
    const body = handleHttpResponse(response);

    return {
      batch_results: body.batch_results,
      meta: {
        alpha_used: body.meta && body.meta.alpha_used !== undefined ? body.meta.alpha_used : alpha,
        query_count: queries.length,
        total_candidates: candidateItems.length,
        latency_ms: body.meta && body.meta.latency_ms !== undefined ? body.meta.latency_ms : null,
      },
    };
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        'data is required and cannot be null or undefined',
        'data'
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        'data must be a plain object with { query, candidateItems } and optionally { alpha, topK, categoricalFields, numericFields }',
        'data'
      );
    }

    const { query, candidateItems, ...rest } = data;
    return this.rankByCombinedNMICosine(query, candidateItems, rest);
  }
}

function createSimilaritySearchClient(apiKey, options) {
  if (
    apiKey !== null &&
    apiKey !== undefined &&
    typeof apiKey === 'object' &&
    !Array.isArray(apiKey)
  ) {
    return new SimilaritySearchClient(apiKey);
  }

  assertNonEmptyString(apiKey, 'apiKey');
  const opts = Object.assign({}, options || {}, { apiKey });
  return new SimilaritySearchClient(opts);
}

const defaultClient = {
  _instance: null,

  _getInstance() {
    if (!this._instance) {
      const apiKey = process.env.SIMILARITY_SEARCH_API_KEY;
      if (!apiKey || apiKey.trim().length === 0) {
        throw new SimilaritySearchAuthError(
          'No API key found. Set SIMILARITY_SEARCH_API_KEY environment variable or instantiate SimilaritySearchClient directly with { apiKey }.'
        );
      }
      this._instance = new SimilaritySearchClient({ apiKey });
    }
    return this._instance;
  },

  async mainMethod(data) {
    return this._getInstance().mainMethod(data);
  },

  async rankByCombinedNMICosine(query, candidateItems, options) {
    return this._getInstance().rankByCombinedNMICosine(query, candidateItems, options);
  },

  async computePairwiseNMI(itemA, itemB, options) {
    return this._getInstance().computePairwiseNMI(itemA, itemB, options);
  },

  async batchRankByCombinedNMICosine(queries, candidateItems, options) {
    return this._getInstance().batchRankByCombinedNMICosine(queries, candidateItems, options);
  },
};

module.exports = defaultClient;
module.exports.SimilaritySearchClient = SimilaritySearchClient;
module.exports.createSimilaritySearchClient = createSimilaritySearchClient;
module.exports.SimilaritySearchAuthError = SimilaritySearchAuthError;
module.exports.SimilaritySearchValidationError = SimilaritySearchValidationError;
module.exports.SimilaritySearchRateLimitError = SimilaritySearchRateLimitError;
module.exports.SimilaritySearchAPIError = SimilaritySearchAPIError;
module.exports.SIMILARITY_SEARCH_DEFAULT_ALPHA = SIMILARITY_SEARCH_DEFAULT_ALPHA;
module.exports.SIMILARITY_SEARCH_MAX_ITEMS = SIMILARITY_SEARCH_MAX_ITEMS;
```