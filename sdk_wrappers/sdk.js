const axios = require('axios');

const SIMILARITY_SEARCH_API_VERSION = 'v1';
const SIMILARITY_SEARCH_BASE_URL = 'https://api.similarity-search.nexus';
const SIMILARITY_SEARCH_DEFAULT_TIMEOUT_MS = 30000;
const SIMILARITY_SEARCH_MAX_VECTORS_PER_REQUEST = 1000;
const SIMILARITY_SEARCH_MIN_VECTOR_DIM = 1;
const SIMILARITY_SEARCH_MAX_VECTOR_DIM = 4096;

class SimilaritySearchAuthError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SimilaritySearchAuthError';
  }
}

class SimilaritySearchValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'SimilaritySearchValidationError';
    this.field = field || null;
  }
}

class SimilaritySearchAPIError extends Error {
  constructor(message, statusCode, responseBody) {
    super(message);
    this.name = 'SimilaritySearchAPIError';
    this.statusCode = statusCode || null;
    this.responseBody = responseBody || null;
  }
}

class SimilaritySearchRateLimitError extends Error {
  constructor(message, retryAfterSeconds) {
    super(message);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds || null;
  }
}

function validateApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new SimilaritySearchAuthError(
      'API key is required. Pass it as { apiKey } to SimilaritySearchClient or set SIMILARITY_SEARCH_API_KEY env var.'
    );
  }
  if (typeof apiKey !== 'string') {
    throw new SimilaritySearchAuthError(
      'API key must be a string, received ' + typeof apiKey
    );
  }
  if (apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError('API key must not be empty.');
  }
}

function validateVector(vector, fieldName) {
  if (!Array.isArray(vector)) {
    throw new SimilaritySearchValidationError(
      fieldName + ' must be an array of numbers, received ' + typeof vector,
      fieldName
    );
  }
  if (vector.length < SIMILARITY_SEARCH_MIN_VECTOR_DIM) {
    throw new SimilaritySearchValidationError(
      fieldName + ' must have at least ' + SIMILARITY_SEARCH_MIN_VECTOR_DIM + ' dimension(s).',
      fieldName
    );
  }
  if (vector.length > SIMILARITY_SEARCH_MAX_VECTOR_DIM) {
    throw new SimilaritySearchValidationError(
      fieldName + ' exceeds maximum dimension ' + SIMILARITY_SEARCH_MAX_VECTOR_DIM + '. Got ' + vector.length + '.',
      fieldName
    );
  }
  for (let i = 0; i < vector.length; i++) {
    if (typeof vector[i] !== 'number' || !isFinite(vector[i])) {
      throw new SimilaritySearchValidationError(
        fieldName + '[' + i + '] must be a finite number, got ' + vector[i],
        fieldName
      );
    }
  }
}

function validateVectorList(vectors, fieldName, maxCount) {
  if (!Array.isArray(vectors)) {
    throw new SimilaritySearchValidationError(
      fieldName + ' must be an array of vectors, received ' + typeof vectors,
      fieldName
    );
  }
  if (vectors.length === 0) {
    throw new SimilaritySearchValidationError(
      fieldName + ' must contain at least one vector.',
      fieldName
    );
  }
  if (vectors.length > maxCount) {
    throw new SimilaritySearchValidationError(
      fieldName + ' exceeds maximum count ' + maxCount + '. Got ' + vectors.length + '.',
      fieldName
    );
  }
  const refDim = vectors[0].length;
  for (let i = 0; i < vectors.length; i++) {
    validateVector(vectors[i], fieldName + '[' + i + ']');
    if (vectors[i].length !== refDim) {
      throw new SimilaritySearchValidationError(
        fieldName + ': all vectors must share the same dimensionality. ' +
        'Expected ' + refDim + ' (from index 0) but index ' + i + ' has ' + vectors[i].length + '.',
        fieldName
      );
    }
  }
}

function validateTopK(topK) {
  if (topK === undefined || topK === null) return;
  if (typeof topK !== 'number' || !Number.isInteger(topK)) {
    throw new SimilaritySearchValidationError('topK must be an integer.', 'topK');
  }
  if (topK < 1 || topK > 500) {
    throw new SimilaritySearchValidationError('topK must be between 1 and 500, got ' + topK + '.', 'topK');
  }
}

function validateNmiWeight(nmiWeight) {
  if (nmiWeight === undefined || nmiWeight === null) return;
  if (typeof nmiWeight !== 'number' || !isFinite(nmiWeight)) {
    throw new SimilaritySearchValidationError('nmiWeight must be a finite number.', 'nmiWeight');
  }
  if (nmiWeight < 0 || nmiWeight > 1) {
    throw new SimilaritySearchValidationError(
      'nmiWeight must be in [0, 1]. Got ' + nmiWeight + '. The complementary cosine weight is derived as (1 - nmiWeight).',
      'nmiWeight'
    );
  }
}

async function handleAxiosError(error) {
  if (error.response) {
    const status = error.response.status;
    const body = error.response.data;

    if (status === 401 || status === 403) {
      throw new SimilaritySearchAuthError(
        'Authentication failed (HTTP ' + status + '). Verify your API key. ' +
        (body && body.detail ? body.detail : '')
      );
    }

    if (status === 429) {
      const retryAfter = error.response.headers && error.response.headers['retry-after']
        ? parseFloat(error.response.headers['retry-after'])
        : null;
      throw new SimilaritySearchRateLimitError(
        'Rate limit exceeded. ' + (retryAfter ? 'Retry after ' + retryAfter + 's.' : 'Check X-RateLimit headers.'),
        retryAfter
      );
    }

    if (status === 422) {
      const detail = body && body.detail ? JSON.stringify(body.detail) : 'Unprocessable entity.';
      throw new SimilaritySearchValidationError('Server rejected payload (HTTP 422): ' + detail);
    }

    throw new SimilaritySearchAPIError(
      'API returned HTTP ' + status + '. ' + (body && body.detail ? body.detail : JSON.stringify(body)),
      status,
      body
    );
  }

  if (error.code === 'ECONNABORTED' || (error.message && error.message.toLowerCase().includes('timeout'))) {
    throw new SimilaritySearchAPIError(
      'Request timed out. Increase the timeout option or reduce payload size.',
      null,
      null
    );
  }

  throw new SimilaritySearchAPIError(
    'Network error: ' + error.message,
    null,
    null
  );
}

class SimilaritySearchClient {
  constructor(options) {
    const opts = options || {};
    const apiKey = opts.apiKey || process.env.SIMILARITY_SEARCH_API_KEY;
    validateApiKey(apiKey);

    this._apiKey = apiKey;
    this._baseURL = (opts.baseURL || SIMILARITY_SEARCH_BASE_URL).replace(/\/$/, '');
    this._timeout = typeof opts.timeoutMs === 'number' ? opts.timeoutMs : SIMILARITY_SEARCH_DEFAULT_TIMEOUT_MS;
    this._maxRetries = typeof opts.maxRetries === 'number' ? opts.maxRetries : 2;

    this._http = axios.create({
      baseURL: this._baseURL,
      timeout: this._timeout,
      headers: {
        'Authorization': 'Bearer ' + this._apiKey,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-SDK-Version': 'similarity-search-sdk-js/1.0.0',
      },
    });
  }

  async _postWithRetry(endpoint, payload, attempt) {
    attempt = attempt || 0;
    try {
      const response = await this._http.post(
        '/' + SIMILARITY_SEARCH_API_VERSION + endpoint,
        payload
      );
      return response.data;
    } catch (error) {
      if (
        error.response &&
        (error.response.status === 429 || error.response.status >= 500) &&
        attempt < this._maxRetries
      ) {
        const backoffMs = Math.pow(2, attempt) * 500;
        await new Promise(function(resolve) { setTimeout(resolve, backoffMs); });
        return this._postWithRetry(endpoint, payload, attempt + 1);
      }
      await handleAxiosError(error);
    }
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        'data must be an object with { query, candidates } — received null or undefined.',
        'data'
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        'data must be a plain object with { query, candidates }, received ' + typeof data,
        'data'
      );
    }
    return this.rankCandidatesByNMICosineScore({
      query: data.query,
      candidates: data.candidates,
      topK: data.topK,
      nmiWeight: data.nmiWeight,
      entropyBins: data.entropyBins,
    });
  }

  async rankCandidatesByNMICosineScore(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'params must be an object with { query, candidates }.',
        'params'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        'params must be a plain object, received ' + typeof params,
        'params'
      );
    }

    validateVector(params.query, 'query');
    validateVectorList(params.candidates, 'candidates', SIMILARITY_SEARCH_MAX_VECTORS_PER_REQUEST);

    if (params.candidates[0].length !== params.query.length) {
      throw new SimilaritySearchValidationError(
        'candidates vectors must have the same dimensionality as query. ' +
        'query has ' + params.query.length + ' dims, candidates[0] has ' + params.candidates[0].length + ' dims.',
        'candidates'
      );
    }

    validateTopK(params.topK);
    validateNmiWeight(params.nmiWeight);

    if (params.entropyBins !== undefined && params.entropyBins !== null) {
      if (typeof params.entropyBins !== 'number' || !Number.isInteger(params.entropyBins)) {
        throw new SimilaritySearchValidationError('entropyBins must be an integer.', 'entropyBins');
      }
      if (params.entropyBins < 2 || params.entropyBins > 256) {
        throw new SimilaritySearchValidationError('entropyBins must be between 2 and 256.', 'entropyBins');
      }
    }

    const payload = {
      query: params.query,
      candidates: params.candidates,
    };

    if (params.topK !== undefined && params.topK !== null) payload.top_k = params.topK;
    if (params.nmiWeight !== undefined && params.nmiWeight !== null) payload.nmi_weight = params.nmiWeight;
    if (params.entropyBins !== undefined && params.entropyBins !== null) payload.entropy_bins = params.entropyBins;

    return this._postWithRetry('/similarity/rank', payload);
  }

  async computePairwiseNMICosineMatrix(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'params must be an object with { vectors }.',
        'params'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        'params must be a plain object, received ' + typeof params,
        'params'
      );
    }

    validateVectorList(params.vectors, 'vectors', 200);

    if (params.nmiWeight !== undefined && params.nmiWeight !== null) {
      validateNmiWeight(params.nmiWeight);
    }
    if (params.entropyBins !== undefined && params.entropyBins !== null) {
      if (typeof params.entropyBins !== 'number' || !Number.isInteger(params.entropyBins)) {
        throw new SimilaritySearchValidationError('entropyBins must be an integer.', 'entropyBins');
      }
      if (params.entropyBins < 2 || params.entropyBins > 256) {
        throw new SimilaritySearchValidationError('entropyBins must be between 2 and 256.', 'entropyBins');
      }
    }

    const payload = {
      vectors: params.vectors,
    };

    if (params.nmiWeight !== undefined && params.nmiWeight !== null) payload.nmi_weight = params.nmiWeight;
    if (params.entropyBins !== undefined && params.entropyBins !== null) payload.entropy_bins = params.entropyBins;

    return this._postWithRetry('/similarity/matrix', payload);
  }

  async extractDimensionEntropyWeights(params) {
    if (params === null || params === undefined) {
      throw new SimilaritySearchValidationError(
        'params must be an object with { vectors }.',
        'params'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new SimilaritySearchValidationError(
        'params must be a plain object, received ' + typeof params,
        'params'
      );
    }

    validateVectorList(params.vectors, 'vectors', SIMILARITY_SEARCH_MAX_VECTORS_PER_REQUEST);

    if (params.entropyBins !== undefined && params.entropyBins !== null) {
      if (typeof params.entropyBins !== 'number' || !Number.isInteger(params.entropyBins)) {
        throw new SimilaritySearchValidationError('entropyBins must be an integer.', 'entropyBins');
      }
      if (params.entropyBins < 2 || params.entropyBins > 256) {
        throw new SimilaritySearchValidationError('entropyBins must be between 2 and 256.', 'entropyBins');
      }
    }

    const payload = {
      vectors: params.vectors,
    };

    if (params.entropyBins !== undefined && params.entropyBins !== null) payload.entropy_bins = params.entropyBins;

    return this._postWithRetry('/similarity/entropy-weights', payload);
  }
}

function createSimilaritySearchClient(options) {
  return new SimilaritySearchClient(options);
}

module.exports = createSimilaritySearchClient;
module.exports.SimilaritySearchClient = SimilaritySearchClient;
module.exports.SimilaritySearchAuthError = SimilaritySearchAuthError;
module.exports.SimilaritySearchValidationError = SimilaritySearchValidationError;
module.exports.SimilaritySearchAPIError = SimilaritySearchAPIError;
module.exports.SimilaritySearchRateLimitError = SimilaritySearchRateLimitError;
module.exports.SIMILARITY_SEARCH_MAX_VECTORS_PER_REQUEST = SIMILARITY_SEARCH_MAX_VECTORS_PER_REQUEST;
module.exports.SIMILARITY_SEARCH_MAX_VECTOR_DIM = SIMILARITY_SEARCH_MAX_VECTOR_DIM;