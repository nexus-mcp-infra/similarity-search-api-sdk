// --- PATCH sdk_route_grounding_manual_backfill ---
// similarity-search-api-sdk, sdk_wrappers/sdk.js
//
// Regrounded contra el codigo real deployado (core/similarity_search_api_api.py
// / https://similarity-search-api-production.up.railway.app/openapi.json),
// transliterado desde sdk_wrappers/similarity_search_api_sdk.py ya
// corregido en este mismo patch -- mismo principio que CLAUDE.md SS9.43
// (sdk.js debe reflejar el sdk.py real, no reinventar su propio esquema).
//
// Esta version de sdk.js nunca fue transliterada de nada real -- las 4
// rutas que llamaba (/similarity/nmi-cosine, /similarity/nmi-cosine/batch,
// /similarity/entropy-calibration, /similarity/pairwise-rank) no
// corresponden a NINGUNA ruta real del servidor, y el dominio tambien
// era fabricado (api.similarity-search.nexus, distinto incluso del
// dominio fabricado que tenia sdk.py -- confirma que los dos SDKs se
// generaron sin verse entre si). Auth header tambien incorrecto
// (Authorization: Bearer en vez de X-API-Key). Reescrito para exponer
// los mismos 3 metodos reales que sdk.py (search/computeCalibratedAlpha/
// scorePair) -- batchComputeNmiCosineScores/inspectEntropyCalibration/
// rankPairwiseNmiCosine no tenian contraparte real en el servidor y se
// eliminan en vez de dejarlos apuntando a endpoints inventados.

const axios = require('axios');

const BASE_URL = process.env.SIMILARITY_API_URL || 'https://similarity-search-api-production.up.railway.app';
const DEFAULT_TIMEOUT_MS = 30000;
const MAX_CORPUS_ITEMS = 500000;
const MIN_ALPHA = 0.0;
const MAX_ALPHA = 1.0;

class SimilaritySearchError extends Error {
  constructor(message, statusCode, details) {
    super(message);
    this.name = 'SimilaritySearchError';
    this.statusCode = statusCode || null;
    this.details = details || null;
  }
}

class AuthenticationError extends SimilaritySearchError {
  constructor() {
    super('Missing or invalid API key. Set SIMILARITY_API_KEY or pass apiKey in options.', 401);
    this.name = 'AuthenticationError';
  }
}

class ValidationError extends SimilaritySearchError {
  constructor(message) {
    super(message, 422);
    this.name = 'ValidationError';
  }
}

class RateLimitError extends SimilaritySearchError {
  constructor(retryAfterSeconds) {
    super('Rate limit exceeded.', 429);
    this.name = 'RateLimitError';
    this.retryAfterSeconds = retryAfterSeconds || null;
  }
}

function resolveApiKey(options) {
  // --- PATCH sdk_x402_only_auth_regrounding ---
  // Optional as of repo commit 4e09c52 (2026-08-25): the server dropped the
  // X-API-Key gate on search()/computeCalibratedAlpha()/scorePair(), x402
  // payment alone is sufficient now, and Stripe metered billing explicitly
  // excludes these 3 routes (_NEXUS_BILLING_EXCLUDED_PATHS in
  // core/similarity_search_api_api.py). No longer throws when absent.
  const key = (options && options.apiKey) || process.env.SIMILARITY_API_KEY;
  if (!key || typeof key !== 'string' || key.trim().length === 0) {
    return null;
  }
  return key.trim();
}

function buildAxiosInstance(apiKey, timeoutMs) {
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client': 'similarity-search-sdk-js/1.0.0'
  };
  if (apiKey) {
    // Still sent when the caller does pass one, for forward-compat with any
    // future re-gating (e.g. the deprecated /similarity/calibrate-alpha stub,
    // which kept its gate) -- see resolveApiKey() above.
    headers['X-API-Key'] = apiKey;
  }
  return axios.create({
    baseURL: BASE_URL,
    timeout: timeoutMs || DEFAULT_TIMEOUT_MS,
    headers
  });
}

function wrapAxiosError(err) {
  if (!err.response) {
    return new SimilaritySearchError(
      `Network error: ${err.message}`,
      null,
      { originalMessage: err.message }
    );
  }
  const { status, data } = err.response;
  const detail = (data && data.detail) || (data && data.message) || JSON.stringify(data);
  if (status === 401 || status === 403) return new AuthenticationError();
  if (status === 429) {
    const retryAfter = err.response.headers['retry-after']
      ? parseInt(err.response.headers['retry-after'], 10)
      : null;
    return new RateLimitError(retryAfter);
  }
  if (status === 422) return new ValidationError(detail);
  return new SimilaritySearchError(detail, status, data);
}

function validateVector(vec, label) {
  if (!Array.isArray(vec) || vec.length === 0) {
    throw new ValidationError(`${label} must be a non-empty array of numbers.`);
  }
  for (let i = 0; i < vec.length; i++) {
    if (typeof vec[i] !== 'number' || !isFinite(vec[i])) {
      throw new ValidationError(`${label}[${i}] must be a finite number, got: ${vec[i]}.`);
    }
  }
}

// corpus: array of [id, vector] pairs -- mirrors sdk.py's
// list[tuple[str, list[float]]]. The real server has no text-embedding
// step, every corpus item (and the query) must already be a numeric
// vector with an id label (CorpusVector{id, vector}).
function validateCorpusItems(items) {
  if (!Array.isArray(items) || items.length === 0) {
    throw new ValidationError('corpus must be a non-empty array of [id, vector] pairs.');
  }
  if (items.length > MAX_CORPUS_ITEMS) {
    throw new ValidationError(
      `corpus exceeds maximum of ${MAX_CORPUS_ITEMS} items per request. Received: ${items.length}.`
    );
  }
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (!Array.isArray(item) || item.length !== 2) {
      throw new ValidationError(`corpus[${i}] must be an [id, vector] pair.`);
    }
    const [id, vector] = item;
    if (typeof id !== 'string' || id.length === 0) {
      throw new ValidationError(`corpus[${i}][0] (id) must be a non-empty string.`);
    }
    validateVector(vector, `corpus[${i}][1]`);
  }
}

function validateAlpha(alpha) {
  if (alpha === undefined || alpha === null) return;
  if (typeof alpha !== 'number' || !isFinite(alpha)) {
    throw new ValidationError('alpha must be a finite number between 0.0 and 1.0.');
  }
  if (alpha < MIN_ALPHA || alpha > MAX_ALPHA) {
    throw new ValidationError(
      `alpha must be between ${MIN_ALPHA} and ${MAX_ALPHA}. Received: ${alpha}.`
    );
  }
}

function validateTopK(topK) {
  if (topK === undefined || topK === null) return;
  if (!Number.isInteger(topK) || topK < 1 || topK > 1000) {
    throw new ValidationError('topK must be an integer between 1 and 1000.');
  }
}

function corpusToPayload(corpus) {
  return corpus.map(([id, vector]) => ({ id, vector }));
}

class SimilaritySearchClient {
  constructor(options) {
    this._apiKey = resolveApiKey(options);
    this._http = buildAxiosInstance(
      this._apiKey,
      options && options.timeoutMs
    );
  }

  async search(queryVector, corpus, options) {
    if (queryVector === undefined || queryVector === null) {
      throw new ValidationError('queryVector is required and must not be null or undefined.');
    }
    validateVector(queryVector, 'queryVector');
    validateCorpusItems(corpus);

    const queryId = (options && options.queryId) || 'query';
    const topK = (options && options.topK) !== undefined ? options.topK : 10;
    const nmiBins = options && options.nmiBins;
    const alphaOverride = options && options.alphaOverride;
    validateTopK(topK);
    validateAlpha(alphaOverride);

    const payload = {
      query: { id: queryId, vector: queryVector },
      corpus: corpusToPayload(corpus),
      top_k: topK
    };
    if (nmiBins !== undefined && nmiBins !== null) {
      payload.nmi_bins = nmiBins;
    }
    if (alphaOverride !== undefined && alphaOverride !== null) {
      payload.alpha_override = alphaOverride;
    }

    try {
      const response = await this._http.post('/similarity/search', payload);
      return response.data;
    } catch (err) {
      throw wrapAxiosError(err);
    }
  }

  // Calls /similarity/calibrate-alpha/v1 -- the unversioned
  // /similarity/calibrate-alpha always returns 501 on the real server
  // (see core/similarity_search_api_api.py), the /v1 path is the only
  // usable one.
  async computeCalibratedAlpha(corpus, options) {
    validateCorpusItems(corpus);
    const nmiBins = options && options.nmiBins;

    const payload = {
      corpus: corpusToPayload(corpus)
    };
    if (nmiBins !== undefined && nmiBins !== null) {
      payload.nmi_bins = nmiBins;
    }

    try {
      const response = await this._http.post('/similarity/calibrate-alpha/v1', payload);
      return response.data;
    } catch (err) {
      throw wrapAxiosError(err);
    }
  }

  // The real server has no single-pair scoring endpoint -- this calls
  // /similarity/batch-score with a single pair and unwraps the one
  // score, which is the only real equivalent available.
  async scorePair(vectorA, vectorB, options) {
    validateVector(vectorA, 'vectorA');
    validateVector(vectorB, 'vectorB');
    const alpha = (options && options.alpha) !== undefined ? options.alpha : 0.5;
    const nmiBins = options && options.nmiBins;
    validateAlpha(alpha);

    const payload = {
      pairs: [[vectorA, vectorB]],
      alpha
    };
    if (nmiBins !== undefined && nmiBins !== null) {
      payload.nmi_bins = nmiBins;
    }

    try {
      const response = await this._http.post('/similarity/batch-score', payload);
      const scores = response.data && response.data.scores;
      if (!Array.isArray(scores) || scores.length === 0) {
        throw new SimilaritySearchError(
          `Unexpected response shape from /similarity/batch-score: missing 'scores' array.`,
          response.status,
          response.data
        );
      }
      return {
        score: scores[0],
        alphaUsed: response.data.alpha_used,
        latencyMs: response.data.latency_ms
      };
    } catch (err) {
      if (err instanceof SimilaritySearchError) throw err;
      throw wrapAxiosError(err);
    }
  }
}

function createClient(options) {
  return new SimilaritySearchClient(options);
}

const _defaultClient = {
  _instance: null,
  _getInstance() {
    if (!this._instance) {
      this._instance = new SimilaritySearchClient({});
    }
    return this._instance;
  }
};

async function mainMethod(data) {
  if (data === undefined || data === null || typeof data !== 'object' || Array.isArray(data)) {
    throw new ValidationError('data must be a plain object with queryVector and corpus fields.');
  }
  const client = _defaultClient._getInstance();
  return client.search(data.queryVector, data.corpus, data.options);
}

module.exports = {
  createClient,
  mainMethod,
  SimilaritySearchClient,
  SimilaritySearchError,
  AuthenticationError,
  ValidationError,
  RateLimitError
};
