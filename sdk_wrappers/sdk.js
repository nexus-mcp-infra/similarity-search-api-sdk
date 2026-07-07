
```javascript
'use strict';

const https = require('https');
const http = require('http');

const SIMILARITY_SEARCH_API_BASE_URL = process.env.SIMILARITY_SEARCH_API_URL || 'https://api.nexus-similarity.io/v1';
const SIMILARITY_SEARCH_API_KEY = process.env.SIMILARITY_SEARCH_API_KEY || '';
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_BOOTSTRAP_SAMPLES = 500;
const MAX_QUERY_FEATURES = 512;
const MAX_CANDIDATES = 10000;
const MAX_TOP_K = 100;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.body = body || null;
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
    super('Rate limit exceeded. Retry after ' + retryAfterSeconds + ' seconds.');
    this.name = 'SimilaritySearchRateLimitError';
    this.statusCode = 429;
    this.retryAfterSeconds = retryAfterSeconds || null;
  }
}

function validateFeatureRecord(record, label) {
  if (record === null || record === undefined) {
    throw new SimilaritySearchValidationError(label + ' must not be null or undefined.');
  }
  if (typeof record !== 'object' || Array.isArray(record)) {
    throw new SimilaritySearchValidationError(label + ' must be a plain object mapping feature names to values.');
  }
  const keys = Object.keys(record);
  if (keys.length === 0) {
    throw new SimilaritySearchValidationError(label + ' must have at least one feature.');
  }
  if (keys.length > MAX_QUERY_FEATURES) {
    throw new SimilaritySearchValidationError(
      label + ' exceeds maximum feature count of ' + MAX_QUERY_FEATURES + '. Got ' + keys.length + '.'
    );
  }
  for (const key of keys) {
    if (typeof key !== 'string' || key.trim().length === 0) {
      throw new SimilaritySearchValidationError(label + ' feature keys must be non-empty strings.');
    }
    const val = record[key];
    if (val === null || val === undefined) {
      throw new SimilaritySearchValidationError(
        label + ' feature "' + key + '" has a null/undefined value. Provide a string or number.'
      );
    }
    if (typeof val !== 'string' && typeof val !== 'number') {
      throw new SimilaritySearchValidationError(
        label + ' feature "' + key + '" has type ' + typeof val + '. Only string and number are accepted.'
      );
    }
    if (typeof val === 'number' && !isFinite(val)) {
      throw new SimilaritySearchValidationError(
        label + ' feature "' + key + '" is ' + val + '. Only finite numbers are accepted.'
      );
    }
  }
}

function validateCandidateList(candidates) {
  if (!Array.isArray(candidates)) {
    throw new SimilaritySearchValidationError('candidates must be an array of feature record objects.');
  }
  if (candidates.length === 0) {
    throw new SimilaritySearchValidationError('candidates array must not be empty.');
  }
  if (candidates.length > MAX_CANDIDATES) {
    throw new SimilaritySearchValidationError(
      'candidates array exceeds maximum size of ' + MAX_CANDIDATES + '. Got ' + candidates.length + '.'
    );
  }
  for (let i = 0; i < candidates.length; i++) {
    validateFeatureRecord(candidates[i], 'candidates[' + i + ']');
  }
}

function buildRequestOptions(method, path, apiKey, timeoutMs) {
  const parsedBase = new URL(SIMILARITY_SEARCH_API_BASE_URL);
  const isHttps = parsedBase.protocol === 'https:';
  const port = parsedBase.port
    ? parseInt(parsedBase.port, 10)
    : isHttps ? 443 : 80;

  return {
    _isHttps: isHttps,
    hostname: parsedBase.hostname,
    port: port,
    path: parsedBase.pathname.replace(/\/$/, '') + path,
    method: method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + apiKey,
      'User-Agent': 'nexus-similarity-search-sdk-js/1.0.0',
      'Accept': 'application/json',
    },
    timeout: timeoutMs,
  };
}

function executeRequest(options, bodyObject) {
  return new Promise((resolve, reject) => {
    const { _isHttps, ...nodeOptions } = options;
    const transport = _isHttps ? https : http;
    const bodyString = JSON.stringify(bodyObject);
    nodeOptions.headers['Content-Length'] = Buffer.byteLength(bodyString);

    const req = transport.request(nodeOptions, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const raw = Buffer.concat(chunks).toString('utf8');
        let parsed;
        try {
          parsed = JSON.parse(raw);
        } catch (_) {
          reject(new SimilaritySearchError(
            'API returned non-JSON response with status ' + res.statusCode + ': ' + raw.slice(0, 200),
            res.statusCode,
            raw
          ));
          return;
        }

        if (res.statusCode === 200 || res.statusCode === 201) {
          resolve(parsed);
          return;
        }
        if (res.statusCode === 401 || res.statusCode === 403) {
          reject(new SimilaritySearchAuthError(
            (parsed && parsed.detail) || 'Authentication failed. Check your SIMILARITY_SEARCH_API_KEY.'
          ));
          return;
        }
        if (res.statusCode === 422) {
          const detail = parsed && parsed.detail
            ? (Array.isArray(parsed.detail)
              ? parsed.detail.map(d => d.msg || JSON.stringify(d)).join('; ')
              : String(parsed.detail))
            : 'Validation error';
          reject(new SimilaritySearchValidationError(detail));
          return;
        }
        if (res.statusCode === 429) {
          const retryAfter = res.headers['retry-after']
            ? parseFloat(res.headers['retry-after'])
            : null;
          reject(new SimilaritySearchRateLimitError(retryAfter));
          return;
        }
        reject(new SimilaritySearchError(
          (parsed && parsed.detail) || 'Unexpected API error with status ' + res.statusCode,
          res.statusCode,
          parsed
        ));
      });
      res.on('error', (err) => {
        reject(new SimilaritySearchError('Response stream error: ' + err.message));
      });
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new SimilaritySearchError(
        'Request timed out after ' + nodeOptions.timeout + 'ms. Consider reducing candidates size or increasing timeout.'
      ));
    });

    req.on('error', (err) => {
      reject(new SimilaritySearchError('Network error: ' + err.message));
    });

    req.write(bodyString);
    req.end();
  });
}

class SimilaritySearchClient {
  constructor(options) {
    options = options || {};

    this._apiKey = options.apiKey || SIMILARITY_SEARCH_API_KEY;
    if (!this._apiKey || typeof this._apiKey !== 'string' || this._apiKey.trim().length === 0) {
      throw new SimilaritySearchAuthError(
        'API key is required. Pass it as options.apiKey or set the SIMILARITY_SEARCH_API_KEY environment variable.'
      );
    }

    this._timeoutMs = typeof options.timeoutMs === 'number' && options.timeoutMs > 0
      ? options.timeoutMs
      : DEFAULT_TIMEOUT_MS;
  }

  async hybridSimilaritySearch(query, candidates, options) {
    if (query === null || query === undefined) {
      throw new SimilaritySearchValidationError('query must not be null or undefined.');
    }
    validateFeatureRecord(query, 'query');
    validateCandidateList(candidates);

    options = options || {};

    const topK = options.topK !== undefined ? options.topK : 10;
    if (typeof topK !== 'number' || !Number.isInteger(topK) || topK < 1 || topK > MAX_TOP_K) {
      throw new SimilaritySearchValidationError(
        'options.topK must be an integer between 1 and ' + MAX_TOP_K + '. Got: ' + topK
      );
    }

    const bootstrapSamples = options.bootstrapSamples !== undefined
      ? options.bootstrapSamples
      : DEFAULT_BOOTSTRAP_SAMPLES;
    if (typeof bootstrapSamples !== 'number' || !Number.isInteger(bootstrapSamples) || bootstrapSamples < 100 || bootstrapSamples > 2000) {
      throw new SimilaritySearchValidationError(
        'options.bootstrapSamples must be an integer between 100 and 2000. Got: ' + bootstrapSamples
      );
    }

    const confidenceLevel = options.confidenceLevel !== undefined ? options.confidenceLevel : 0.95;
    if (typeof confidenceLevel !== 'number' || confidenceLevel <= 0 || confidenceLevel >= 1) {
      throw new SimilaritySearchValidationError(
        'options.confidenceLevel must be a number in (0, 1). Got: ' + confidenceLevel
      );
    }

    const body = {
      query,
      candidates,
      top_k: topK,
      bootstrap_samples: bootstrapSamples,
      confidence_level: confidenceLevel,
    };

    if (options.featureWeightOverride !== undefined) {
      if (
        typeof options.featureWeightOverride !== 'object' ||
        options.featureWeightOverride === null ||
        Array.isArray(options.featureWeightOverride)
      ) {
        throw new SimilaritySearchValidationError(
          'options.featureWeightOverride must be a plain object mapping feature names to positive numbers.'
        );
      }
      for (const [k, v] of Object.entries(options.featureWeightOverride)) {
        if (typeof v !== 'number' || v <= 0 || !isFinite(v)) {
          throw new SimilaritySearchValidationError(
            'options.featureWeightOverride["' + k + '"] must be a finite positive number. Got: ' + v
          );
        }
      }
      body.feature_weight_override = options.featureWeightOverride;
    }

    const reqOptions = buildRequestOptions('POST', '/hybrid-similarity-search', this._apiKey, this._timeoutMs);
    return executeRequest(reqOptions, body);
  }

  async scoreFeaturePair(featureA, featureB, options) {
    if (featureA === null || featureA === undefined) {
      throw new SimilaritySearchValidationError('featureA must not be null or undefined.');
    }
    if (featureB === null || featureB === undefined) {
      throw new SimilaritySearchValidationError('featureB must not be null or undefined.');
    }
    if (!Array.isArray(featureA) || featureA.length === 0) {
      throw new SimilaritySearchValidationError('featureA must be a non-empty array of string or number values.');
    }
    if (!Array.isArray(featureB) || featureB.length === 0) {
      throw new SimilaritySearchValidationError('featureB must be a non-empty array of string or number values.');
    }
    if (featureA.length !== featureB.length) {
      throw new SimilaritySearchValidationError(
        'featureA and featureB must have the same length. Got ' + featureA.length + ' vs ' + featureB.length + '.'
      );
    }
    if (featureA.length > MAX_QUERY_FEATURES) {
      throw new SimilaritySearchValidationError(
        'Feature arrays exceed maximum length of ' + MAX_QUERY_FEATURES + '. Got ' + featureA.length + '.'
      );
    }

    for (let i = 0; i < featureA.length; i++) {
      const va = featureA[i];
      const vb = featureB[i];
      if (typeof va !== 'string' && typeof va !== 'number') {
        throw new SimilaritySearchValidationError('featureA[' + i + '] must be string or number. Got ' + typeof va + '.');
      }
      if (typeof vb !== 'string' && typeof vb !== 'number') {
        throw new SimilaritySearchValidationError('featureB[' + i + '] must be string or number. Got ' + typeof vb + '.');
      }
      if (typeof va === 'number' && !isFinite(va)) {
        throw new SimilaritySearchValidationError('featureA[' + i + '] is not finite: ' + va + '.');
      }
      if (typeof vb === 'number' && !isFinite(vb)) {
        throw new SimilaritySearchValidationError('featureB[' + i + '] is not finite: ' + vb + '.');
      }
    }

    options = options || {};
    const bootstrapSamples = options.bootstrapSamples !== undefined
      ? options.bootstrapSamples
      : DEFAULT_BOOTSTRAP_SAMPLES;
    if (typeof bootstrapSamples !== 'number' || !Number.isInteger(bootstrapSamples) || bootstrapSamples < 100 || bootstrapSamples > 2000) {
      throw new SimilaritySearchValidationError(
        'options.bootstrapSamples must be an integer between 100 and 2000. Got: ' + bootstrapSamples
      );
    }

    const body = {
      feature_a: featureA,
      feature_b: featureB,
      bootstrap_samples: bootstrapSamples,
    };

    const reqOptions = buildRequestOptions('POST', '/score-feature-pair', this._apiKey, this._timeoutMs);
    return executeRequest(reqOptions, body);
  }

  async detectFeatureSchema(records) {
    if (!Array.isArray(records) || records.length === 0) {
      throw new SimilaritySearchValidationError('records must be a non-empty array of feature record objects.');
    }
    if (records.length > 1000) {
      throw new SimilaritySearchValidationError(
        'records array exceeds maximum size of 1000 for schema detection. Got ' + records.length + '.'
      );
    }
    for (let i = 0; i < records.length; i++) {
      validateFeatureRecord(records[i], 'records[' + i + ']');
    }

    const body = { records };
    const reqOptions = buildRequestOptions('POST', '/detect-feature-schema', this._apiKey, this._timeoutMs);
    return executeRequest(reqOptions, body);
  }

  async computeNmiMatrix(records, options) {
    if (!Array.isArray(records) || records.length < 2) {
      throw new SimilaritySearchValidationError(
        'records must be an array of at least 2 feature record objects for NMI matrix computation.'
      );
    }
    if (records.length > 5000) {
      throw new SimilaritySearchValidationError(
        'records array exceeds maximum size of 5000 for NMI matrix. Got ' + records.length + '.'
      );
    }
    for (let i = 0; i < records.length; i++) {
      validateFeatureRecord(records[i], 'records[' + i + ']');
    }

    options = options || {};
    const normalizeByJointEntropy = options.normalizeByJointEntropy !== undefined
      ? options.normalizeByJointEntropy
      : true;
    if (typeof normalizeByJointEntropy !== 'boolean') {
      throw new SimilaritySearchValidationError(
        'options.normalizeByJointEntropy must be a boolean. Got: ' + typeof normalizeByJointEntropy
      );
    }

    const body = {
      records,
      normalize_by_joint_entropy: normalizeByJointEntropy,
    };

    const reqOptions = buildRequestOptions('POST', '/compute-nmi-matrix', this._apiKey, this._timeoutMs);
    return executeRequest(reqOptions, body);
  }
}

function createSimilaritySearchClient(options) {
  return new SimilaritySearchClient(options);
}

const _defaultClient = (() => {
  try {
    return new SimilaritySearchClient({});
  } catch (_) {
    return null;
  }
})();

async function mainMethod(data) {
  if (data === null || data === undefined) {
    throw new SimilaritySearchValidationError(
      'data must not be null or undefined. Expected { query, candidates, options? }.'
    );
  }
  if (typeof data !== 'object' || Array.isArray(data)) {
    throw new SimilaritySearchValidationError(
      'data must be a plain object with shape { query, candidates, options? }.'
    );
  }
  if (data.query === undefined) {
    throw new SimilaritySearchValidationError(
      'data.query is required. Provide a feature record object.'
    );
  }
  if (data.candidates === undefined) {
    throw new SimilaritySearchValidationError(
      'data.candidates is required. Provide an array of feature record objects.'
    );
  }

  const client = _defaultClient || createSimilaritySearchClient({});
  return client.hybridSimilaritySearch(data.query, data.candidates, data.options);
}

module.exports = {
  mainMethod,
  createSimilaritySearchClient,
  SimilaritySearchClient,
  SimilaritySearchError,
  SimilaritySearchAuthError,
  SimilaritySearchValidationError,
  SimilaritySearchRateLimitError,
};
```