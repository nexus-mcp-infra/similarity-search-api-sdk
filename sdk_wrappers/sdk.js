const axios = require('axios');

const SIMILARITY_SEARCH_API_BASE_URL = process.env.SIMILARITY_SEARCH_API_BASE_URL || 'https://api.similaritysearch.io/v1';
const SIMILARITY_SEARCH_API_TIMEOUT_MS = 30000;

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
  constructor(message, retryAfterMs) {
    super(message);
    this.name = 'SimilaritySearchRateLimitError';
    this.retryAfterMs = retryAfterMs || null;
  }
}

class SimilaritySearchApiError extends Error {
  constructor(message, statusCode, responseBody) {
    super(message);
    this.name = 'SimilaritySearchApiError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

function assertNonEmptyFloatArray(value, fieldName) {
  if (value === null || value === undefined) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must not be null or undefined`);
  }
  if (!Array.isArray(value)) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must be an array of numbers, got ${typeof value}`);
  }
  if (value.length === 0) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must not be an empty array`);
  }
  for (let i = 0; i < value.length; i++) {
    if (typeof value[i] !== 'number' || !isFinite(value[i])) {
      throw new SimilaritySearchValidationError(
        `'${fieldName}[${i}]' must be a finite number, got ${value[i]}`
      );
    }
  }
}

function assertFloat01(value, fieldName) {
  if (typeof value !== 'number' || !isFinite(value)) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must be a finite number, got ${typeof value}`);
  }
  if (value < 0 || value > 1) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must be between 0 and 1 inclusive, got ${value}`);
  }
}

function assertPositiveInteger(value, fieldName) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must be a positive integer, got ${value}`);
  }
}

function assertNonEmptyString(value, fieldName) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new SimilaritySearchValidationError(`'${fieldName}' must be a non-empty string, got ${JSON.stringify(value)}`);
  }
}

function assertVectorArrayConsistentDimension(vectors, queryVector) {
  const dim = queryVector.length;
  for (let i = 0; i < vectors.length; i++) {
    if (!Array.isArray(vectors[i]) || vectors[i].length !== dim) {
      throw new SimilaritySearchValidationError(
        `vectors[${i}] has dimension ${Array.isArray(vectors[i]) ? vectors[i].length : 'N/A'}, expected ${dim} to match query_vector`
      );
    }
  }
}

function buildAuthenticatedAxiosInstance(apiKey) {
  if (!apiKey || typeof apiKey !== 'string' || apiKey.trim().length === 0) {
    throw new SimilaritySearchAuthError(
      'A valid API key must be provided. Set it via SimilaritySearchClient({ apiKey }) or the SIMILARITY_SEARCH_API_KEY environment variable.'
    );
  }
  return axios.create({
    baseURL: SIMILARITY_SEARCH_API_BASE_URL,
    timeout: SIMILARITY_SEARCH_API_TIMEOUT_MS,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-SDK-Client': 'similarity-search-sdk-js/1.0.0',
    },
  });
}

function mapAxiosErrorToSdkError(error) {
  if (error.response) {
    const { status, data } = error.response;
    const body = data || {};
    if (status === 401 || status === 403) {
      return new SimilaritySearchAuthError(
        `Authentication failed (HTTP ${status}): ${body.detail || body.message || 'Invalid or missing API key'}`
      );
    }
    if (status === 429) {
      const retryAfter = error.response.headers['retry-after'];
      const retryAfterMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : null;
      return new SimilaritySearchRateLimitError(
        `Rate limit exceeded (HTTP 429): ${body.detail || body.message || 'Too many requests'}`,
        retryAfterMs
      );
    }
    if (status === 422) {
      const detail = body.detail
        ? (Array.isArray(body.detail) ? body.detail.map(e => `${e.loc ? e.loc.join('.') : 'field'}: ${e.msg}`).join('; ') : body.detail)
        : 'Unprocessable entity';
      return new SimilaritySearchValidationError(`Server rejected input (HTTP 422): ${detail}`);
    }
    return new SimilaritySearchApiError(
      `API error (HTTP ${status}): ${body.detail || body.message || 'Unknown server error'}`,
      status,
      body
    );
  }
  if (error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'))) {
    return new SimilaritySearchApiError(
      `Request timed out after ${SIMILARITY_SEARCH_API_TIMEOUT_MS}ms. Consider reducing vector dimensionality or candidate set size.`,
      null,
      null
    );
  }
  return new SimilaritySearchApiError(
    `Network error: ${error.message}`,
    null,
    null
  );
}

class SimilaritySearchClient {
  constructor({ apiKey, baseUrl } = {}) {
    const resolvedApiKey = apiKey || process.env.SIMILARITY_SEARCH_API_KEY;
    this._http = buildAuthenticatedAxiosInstance(resolvedApiKey);
    if (baseUrl) {
      if (typeof baseUrl !== 'string' || baseUrl.trim().length === 0) {
        throw new SimilaritySearchValidationError("'baseUrl' must be a non-empty string");
      }
      this._http.defaults.baseURL = baseUrl.replace(/\/$/, '');
    }
  }

  async nmiCosineCompositeScore({ query_vector, candidate_vectors, alpha = 0.5, domain_tag = null }) {
    assertNonEmptyFloatArray(query_vector, 'query_vector');
    if (candidate_vectors === null || candidate_vectors === undefined) {
      throw new SimilaritySearchValidationError("'candidate_vectors' must not be null or undefined");
    }
    if (!Array.isArray(candidate_vectors) || candidate_vectors.length === 0) {
      throw new SimilaritySearchValidationError("'candidate_vectors' must be a non-empty array of vectors");
    }
    if (candidate_vectors.length > 50000) {
      throw new SimilaritySearchValidationError(
        `'candidate_vectors' exceeds maximum of 50000 items per request, got ${candidate_vectors.length}`
      );
    }
    candidate_vectors.forEach((vec, i) => assertNonEmptyFloatArray(vec, `candidate_vectors[${i}]`));
    assertVectorArrayConsistentDimension(candidate_vectors, query_vector);
    assertFloat01(alpha, 'alpha');
    if (domain_tag !== null && domain_tag !== undefined) {
      assertNonEmptyString(domain_tag, 'domain_tag');
      if (domain_tag.length > 64) {
        throw new SimilaritySearchValidationError("'domain_tag' must not exceed 64 characters");
      }
    }

    const payload = { query_vector, candidate_vectors, alpha };
    if (domain_tag) payload.domain_tag = domain_tag;

    try {
      const response = await this._http.post('/score/nmi-cosine-composite', payload);
      return response.data;
    } catch (err) {
      throw mapAxiosErrorToSdkError(err);
    }
  }

  async rankedNmiCosineSearch({ query_vector, candidate_vectors, alpha = 0.5, top_k = 10, domain_tag = null }) {
    assertNonEmptyFloatArray(query_vector, 'query_vector');
    if (!Array.isArray(candidate_vectors) || candidate_vectors.length === 0) {
      throw new SimilaritySearchValidationError("'candidate_vectors' must be a non-empty array of vectors");
    }
    if (candidate_vectors.length > 50000) {
      throw new SimilaritySearchValidationError(
        `'candidate_vectors' exceeds maximum of 50000 items per request, got ${candidate_vectors.length}`
      );
    }
    candidate_vectors.forEach((vec, i) => assertNonEmptyFloatArray(vec, `candidate_vectors[${i}]`));
    assertVectorArrayConsistentDimension(candidate_vectors, query_vector);
    assertFloat01(alpha, 'alpha');
    assertPositiveInteger(top_k, 'top_k');
    if (top_k > candidate_vectors.length) {
      throw new SimilaritySearchValidationError(
        `'top_k' (${top_k}) cannot exceed the number of candidate_vectors (${candidate_vectors.length})`
      );
    }
    if (domain_tag !== null && domain_tag !== undefined) {
      assertNonEmptyString(domain_tag, 'domain_tag');
      if (domain_tag.length > 64) {
        throw new SimilaritySearchValidationError("'domain_tag' must not exceed 64 characters");
      }
    }

    const payload = { query_vector, candidate_vectors, alpha, top_k };
    if (domain_tag) payload.domain_tag = domain_tag;

    try {
      const response = await this._http.post('/search/ranked-nmi-cosine', payload);
      return response.data;
    } catch (err) {
      throw mapAxiosErrorToSdkError(err);
    }
  }

  async pairwiseNmiCosineMatrix({ vectors, alpha = 0.5, domain_tag = null }) {
    if (!Array.isArray(vectors) || vectors.length < 2) {
      throw new SimilaritySearchValidationError("'vectors' must be an array of at least 2 vectors");
    }
    if (vectors.length > 1000) {
      throw new SimilaritySearchValidationError(
        `'vectors' exceeds maximum of 1000 items for pairwise matrix (would produce ${vectors.length * vectors.length} scores), got ${vectors.length}`
      );
    }
    vectors.forEach((vec, i) => assertNonEmptyFloatArray(vec, `vectors[${i}]`));
    assertVectorArrayConsistentDimension(vectors.slice(1), vectors[0]);
    assertFloat01(alpha, 'alpha');
    if (domain_tag !== null && domain_tag !== undefined) {
      assertNonEmptyString(domain_tag, 'domain_tag');
      if (domain_tag.length > 64) {
        throw new SimilaritySearchValidationError("'domain_tag' must not exceed 64 characters");
      }
    }

    const payload = { vectors, alpha };
    if (domain_tag) payload.domain_tag = domain_tag;

    try {
      const response = await this._http.post('/matrix/pairwise-nmi-cosine', payload);
      return response.data;
    } catch (err) {
      throw mapAxiosErrorToSdkError(err);
    }
  }

  async suggestOptimalAlphaForDomain({ domain_tag }) {
    assertNonEmptyString(domain_tag, 'domain_tag');
    if (domain_tag.length > 64) {
      throw new SimilaritySearchValidationError("'domain_tag' must not exceed 64 characters");
    }

    try {
      const response = await this._http.get('/alpha/domain-suggestion', {
        params: { domain_tag },
      });
      return response.data;
    } catch (err) {
      throw mapAxiosErrorToSdkError(err);
    }
  }

  async duplicateDetectionByNmiCosineThreshold({ vectors, ids = null, alpha = 0.5, composite_threshold = 0.85, domain_tag = null }) {
    if (!Array.isArray(vectors) || vectors.length < 2) {
      throw new SimilaritySearchValidationError("'vectors' must be an array of at least 2 vectors");
    }
    if (vectors.length > 10000) {
      throw new SimilaritySearchValidationError(
        `'vectors' exceeds maximum of 10000 items for duplicate detection, got ${vectors.length}`
      );
    }
    vectors.forEach((vec, i) => assertNonEmptyFloatArray(vec, `vectors[${i}]`));
    assertVectorArrayConsistentDimension(vectors.slice(1), vectors[0]);
    assertFloat01(alpha, 'alpha');
    assertFloat01(composite_threshold, 'composite_threshold');
    if (composite_threshold < 0.5) {
      throw new SimilaritySearchValidationError(
        `'composite_threshold' below 0.5 will produce excessive false positives in duplicate detection; minimum accepted is 0.5, got ${composite_threshold}`
      );
    }
    if (ids !== null && ids !== undefined) {
      if (!Array.isArray(ids) || ids.length !== vectors.length) {
        throw new SimilaritySearchValidationError(
          `'ids' must be an array of the same length as 'vectors' (${vectors.length}), got ${Array.isArray(ids) ? ids.length : typeof ids}`
        );
      }
      ids.forEach((id, i) => {
        if (typeof id !== 'string' && typeof id !== 'number') {
          throw new SimilaritySearchValidationError(`'ids[${i}]' must be a string or number, got ${typeof id}`);
        }
      });
    }
    if (domain_tag !== null && domain_tag !== undefined) {
      assertNonEmptyString(domain_tag, 'domain_tag');
      if (domain_tag.length > 64) {
        throw new SimilaritySearchValidationError("'domain_tag' must not exceed 64 characters");
      }
    }

    const payload = { vectors, alpha, composite_threshold };
    if (ids) payload.ids = ids;
    if (domain_tag) payload.domain_tag = domain_tag;

    try {
      const response = await this._http.post('/duplicates/nmi-cosine-threshold', payload);
      return response.data;
    } catch (err) {
      throw mapAxiosErrorToSdkError(err);
    }
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new SimilaritySearchValidationError(
        "'data' must not be null or undefined. Expected an object with at least 'query_vector' and 'candidate_vectors'."
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new SimilaritySearchValidationError(
        "'data' must be a plain object. Expected keys: query_vector, candidate_vectors, and optionally alpha, top_k, domain_tag."
      );
    }
    const { query_vector, candidate_vectors, alpha, top_k, domain_tag } = data;
    return this.rankedNmiCosineSearch({
      query_vector,
      candidate_vectors,
      alpha: alpha !== undefined ? alpha : 0.5,
      top_k: top_k !== undefined ? top_k : 10,
      domain_tag: domain_tag || null,
    });
  }
}

function createSimilaritySearchClient(options = {}) {
  return new SimilaritySearchClient(options);
}

const _defaultClient = {
  _instance: null,
  _getOrCreate() {
    if (!this._instance) {
      const apiKey = process.env.SIMILARITY_SEARCH_API_KEY;
      if (!apiKey) {
        throw new SimilaritySearchAuthError(
          'No API key found. Set SIMILARITY_SEARCH_API_KEY in your environment or instantiate SimilaritySearchClient({ apiKey }) explicitly.'
        );
      }
      this._instance = new SimilaritySearchClient({ apiKey });
    }
    return this._instance;
  },
  async mainMethod(data) {
    return this._getOrCreate().mainMethod(data);
  },
  async nmiCosineCompositeScore(params) {
    return this._getOrCreate().nmiCosineCompositeScore(params);
  },
  async rankedNmiCosineSearch(params) {
    return this._getOrCreate().rankedNmiCosineSearch(params);
  },
  async pairwiseNmiCosineMatrix(params) {
    return this._getOrCreate().pairwiseNmiCosineMatrix(params);
  },
  async suggestOptimalAlphaForDomain(params) {
    return this._getOrCreate().suggestOptimalAlphaForDomain(params);
  },
  async duplicateDetectionByNmiCosineThreshold(params) {
    return this._getOrCreate().duplicateDetectionByNmiCosineThreshold(params);
  },
};

module.exports = _defaultClient;
module.exports.SimilaritySearchClient = SimilaritySearchClient;
module.exports.createSimilaritySearchClient = createSimilaritySearchClient;
module.exports.SimilaritySearchAuthError = SimilaritySearchAuthError;
module.exports.SimilaritySearchValidationError = SimilaritySearchValidationError;
module.exports.SimilaritySearchRateLimitError = SimilaritySearchRateLimitError;
module.exports.SimilaritySearchApiError = SimilaritySearchApiError;