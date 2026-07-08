const axios = require('axios');

class SimilaritySearchClient {
  constructor({ apiKey, baseUrl = 'https://api.similarity-search.io/v1', timeout = 30000 } = {}) {
    if (!apiKey || typeof apiKey !== 'string' || apiKey.trim() === '') {
      throw new Error('SimilaritySearchClient requires a non-empty apiKey string');
    }
    this.apiKey = apiKey.trim();
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.http = axios.create({
      baseURL: this.baseUrl,
      timeout,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'X-SDK-Version': '1.0.0',
        'X-SDK-Language': 'javascript'
      }
    });

    this.http.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          const { status, data } = error.response;
          const message = (data && data.detail) ? data.detail : JSON.stringify(data);
          const err = new SimilaritySearchAPIError(message, status, data);
          return Promise.reject(err);
        }
        if (error.code === 'ECONNABORTED') {
          return Promise.reject(new SimilaritySearchAPIError('Request timed out', 408, null));
        }
        return Promise.reject(new SimilaritySearchAPIError(error.message, 0, null));
      }
    );
  }

  async rankByHybridScore({ query, corpus, alpha = 0.5, inputType = 'dense_vectors', topK = null }) {
    if (query === null || query === undefined) {
      throw new TypeError('rankByHybridScore: query must be a non-null array');
    }
    if (!Array.isArray(query) || query.length === 0) {
      throw new TypeError('rankByHybridScore: query must be a non-empty array');
    }
    if (!Array.isArray(corpus) || corpus.length === 0) {
      throw new TypeError('rankByHybridScore: corpus must be a non-empty array of vectors or distributions');
    }
    if (typeof alpha !== 'number' || alpha < 0 || alpha > 1) {
      throw new RangeError('rankByHybridScore: alpha must be a number in [0.0, 1.0]');
    }
    if (!['dense_vectors', 'discrete_distributions'].includes(inputType)) {
      throw new TypeError("rankByHybridScore: inputType must be 'dense_vectors' or 'discrete_distributions'");
    }
    if (topK !== null && (!Number.isInteger(topK) || topK < 1)) {
      throw new RangeError('rankByHybridScore: topK must be a positive integer or null');
    }

    const payload = { query, corpus, alpha, input_type: inputType };
    if (topK !== null) payload.top_k = topK;

    const response = await this.http.post('/rank', payload);
    return response.data;
  }

  async scorePair({ vectorA, vectorB, alpha = 0.5, inputType = 'dense_vectors' }) {
    if (!Array.isArray(vectorA) || vectorA.length === 0) {
      throw new TypeError('scorePair: vectorA must be a non-empty array');
    }
    if (!Array.isArray(vectorB) || vectorB.length === 0) {
      throw new TypeError('scorePair: vectorB must be a non-empty array');
    }
    if (vectorA.length !== vectorB.length) {
      throw new RangeError(`scorePair: vectorA (length ${vectorA.length}) and vectorB (length ${vectorB.length}) must have equal length`);
    }
    if (typeof alpha !== 'number' || alpha < 0 || alpha > 1) {
      throw new RangeError('scorePair: alpha must be a number in [0.0, 1.0]');
    }
    if (!['dense_vectors', 'discrete_distributions'].includes(inputType)) {
      throw new TypeError("scorePair: inputType must be 'dense_vectors' or 'discrete_distributions'");
    }

    const payload = { vector_a: vectorA, vector_b: vectorB, alpha, input_type: inputType };
    const response = await this.http.post('/score', payload);
    return response.data;
  }

  async batchRank({ queries, corpus, alpha = 0.5, inputType = 'dense_vectors', topK = null }) {
    if (!Array.isArray(queries) || queries.length === 0) {
      throw new TypeError('batchRank: queries must be a non-empty array of query vectors');
    }
    if (queries.length > 50) {
      throw new RangeError('batchRank: maximum 50 queries per batch call');
    }
    if (!Array.isArray(corpus) || corpus.length === 0) {
      throw new TypeError('batchRank: corpus must be a non-empty array');
    }
    if (typeof alpha !== 'number' || alpha < 0 || alpha > 1) {
      throw new RangeError('batchRank: alpha must be a number in [0.0, 1.0]');
    }
    if (!['dense_vectors', 'discrete_distributions'].includes(inputType)) {
      throw new TypeError("batchRank: inputType must be 'dense_vectors' or 'discrete_distributions'");
    }
    if (topK !== null && (!Number.isInteger(topK) || topK < 1)) {
      throw new RangeError('batchRank: topK must be a positive integer or null');
    }

    const payload = { queries, corpus, alpha, input_type: inputType };
    if (topK !== null) payload.top_k = topK;

    const response = await this.http.post('/batch_rank', payload);
    return response.data;
  }

  async introspectNMIComponents({ vectorA, vectorB, inputType = 'dense_vectors' }) {
    if (!Array.isArray(vectorA) || vectorA.length === 0) {
      throw new TypeError('introspectNMIComponents: vectorA must be a non-empty array');
    }
    if (!Array.isArray(vectorB) || vectorB.length === 0) {
      throw new TypeError('introspectNMIComponents: vectorB must be a non-empty array');
    }
    if (vectorA.length !== vectorB.length) {
      throw new RangeError(`introspectNMIComponents: length mismatch — vectorA: ${vectorA.length}, vectorB: ${vectorB.length}`);
    }
    if (!['dense_vectors', 'discrete_distributions'].includes(inputType)) {
      throw new TypeError("introspectNMIComponents: inputType must be 'dense_vectors' or 'discrete_distributions'");
    }

    const payload = { vector_a: vectorA, vector_b: vectorB, input_type: inputType };
    const response = await this.http.post('/introspect', payload);
    return response.data;
  }

  async healthCheck() {
    const response = await this.http.get('/health');
    return response.data;
  }
}

class SimilaritySearchAPIError extends Error {
  constructor(message, statusCode, responseBody) {
    super(message);
    this.name = 'SimilaritySearchAPIError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

function createClient(options) {
  if (!options || typeof options !== 'object') {
    throw new TypeError('createClient: options must be an object with at least { apiKey }');
  }
  return new SimilaritySearchClient(options);
}

const _defaultClient = { _instance: null };

async function mainMethod(data) {
  if (!data || typeof data !== 'object') {
    throw new TypeError('mainMethod: data must be a non-null object');
  }

  const apiKey = data.apiKey || process.env.SIMILARITY_SEARCH_API_KEY;
  if (!apiKey) {
    throw new Error(
      'mainMethod: no apiKey provided. Pass data.apiKey or set SIMILARITY_SEARCH_API_KEY environment variable'
    );
  }

  const client = createClient({
    apiKey,
    baseUrl: data.baseUrl,
    timeout: data.timeout
  });

  const operation = data.operation || 'rankByHybridScore';
  const operationMap = {
    rankByHybridScore: () => client.rankByHybridScore(data),
    scorePair: () => client.scorePair(data),
    batchRank: () => client.batchRank(data),
    introspectNMIComponents: () => client.introspectNMIComponents(data),
    healthCheck: () => client.healthCheck()
  };

  if (!operationMap[operation]) {
    throw new TypeError(
      `mainMethod: unknown operation '${operation}'. Valid operations: ${Object.keys(operationMap).join(', ')}`
    );
  }

  return operationMap[operation]();
}

module.exports = mainMethod;
module.exports.createClient = createClient;
module.exports.SimilaritySearchClient = SimilaritySearchClient;
module.exports.SimilaritySearchAPIError = SimilaritySearchAPIError;
module.exports.mainMethod = mainMethod;