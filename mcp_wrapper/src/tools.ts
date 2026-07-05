/**
 * Tool registration — one entry per capability exposed by this MCP server.
 * Business logic lives in the core service; these handlers only validate,
 * call, and shape the response for agent consumption.
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { callCore } from "./coreClient.js";
import { CoreServiceError } from "./types.js";
import { CHARACTER_LIMIT } from "./constants.js";
import {
  RankEmbeddingsByNmiCosineInputSchema,   EstimatePairwiseNmiMatrixInputSchema,   ScoreCandidatePairSignificanceInputSchema,   DetectEmbeddingDimensionRedundancyInputSchema,   CalibrateNmiCosineWeightForCorpusInputSchema,
} from "./schemas.js";
import type {
  RankEmbeddingsByNmiCosineInput,   EstimatePairwiseNmiMatrixInput,   ScoreCandidatePairSignificanceInput,   DetectEmbeddingDimensionRedundancyInput,   CalibrateNmiCosineWeightForCorpusInput,
} from "./types.js";

export function registerTools(server: McpServer): void {

  server.registerTool(
    "nexus_similarity_search_api_rank_embeddings_by_nmi_cosine",
    {
      title: "NMI-Cosine Hybrid Ranking",
      description: `Ranks a corpus of embeddings against a query vector using a hybrid score that combines cosine similarity with Normalized Mutual Information (NMI) computed via joint-entropy estimation over discretized embedding dimensions. Returns per-pair p-values from bootstrap confidence intervals. Use when you need statistically validated similarity rankings where you must distinguish real dependency patterns from random correlation. Do NOT use for real-time latency-sensitive paths (>500 vectors adds bootstrap overhead), for pure nearest-neighbor ANN tasks where p-values are irrelevant, or when corpus vectors are fewer than 10 (bootstrap intervals become unreliable).

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: RankEmbeddingsByNmiCosineInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_vector, corpus_vectors, corpus_ids, discretization_bins, bootstrap_iterations, top_k, alpha, filter_insignificant, nmi_cosine_weight }: RankEmbeddingsByNmiCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-nmi-cosine",
          "POST",
          { query_vector, corpus_vectors, corpus_ids, discretization_bins, bootstrap_iterations, top_k, alpha, filter_insignificant, nmi_cosine_weight },
        );

        const text = JSON.stringify(output, null, 2);
        const truncated = text.length > CHARACTER_LIMIT;
        const display = truncated
          ? text.slice(0, CHARACTER_LIMIT) +
            `\n\n... [truncated, ${text.length - CHARACTER_LIMIT} more characters. ` +
            `Narrow your query or request fewer results.]`
          : text;

        return {
          content: [{ type: "text" as const, text: display }],
          structuredContent: output,
        };
      } catch (err) {
        if (err instanceof CoreServiceError) {
          return {
            content: [{
              type: "text" as const,
              text: `Error calling nexus_similarity_search_api_rank_embeddings_by_nmi_cosine (${err.code}): ${err.message}` +
                (err.requestId ? ` [request_id=${err.requestId}]` : ""),
            }],
            isError: true,
          };
        }
        throw err;
      }
    },
  );
  server.registerTool(
    "nexus_similarity_search_api_estimate_pairwise_nmi_matrix",
    {
      title: "Pairwise NMI Matrix Estimation",
      description: `Computes the full N×N Normalized Mutual Information matrix for a set of embeddings, returning each cell's NMI score along with a bootstrap-derived p-value under the null hypothesis H0: NMI=0 (independence). Use for clustering pre-analysis, redundancy detection across a document set, or graph-of-similarity construction where edge weights must be statistically grounded. Do NOT use when N > 80 — O(N^2 * bootstrap_iterations) cost makes it prohibitive; use rank_embeddings_by_nmi_cosine in batches instead. Not suitable as a real-time retrieval path.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: EstimatePairwiseNmiMatrixInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ vectors, vector_ids, discretization_bins, bootstrap_iterations, alpha }: EstimatePairwiseNmiMatrixInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/pairwise-nmi-matrix",
          "POST",
          { vectors, vector_ids, discretization_bins, bootstrap_iterations, alpha },
        );

        const text = JSON.stringify(output, null, 2);
        const truncated = text.length > CHARACTER_LIMIT;
        const display = truncated
          ? text.slice(0, CHARACTER_LIMIT) +
            `\n\n... [truncated, ${text.length - CHARACTER_LIMIT} more characters. ` +
            `Narrow your query or request fewer results.]`
          : text;

        return {
          content: [{ type: "text" as const, text: display }],
          structuredContent: output,
        };
      } catch (err) {
        if (err instanceof CoreServiceError) {
          return {
            content: [{
              type: "text" as const,
              text: `Error calling nexus_similarity_search_api_estimate_pairwise_nmi_matrix (${err.code}): ${err.message}` +
                (err.requestId ? ` [request_id=${err.requestId}]` : ""),
            }],
            isError: true,
          };
        }
        throw err;
      }
    },
  );
  server.registerTool(
    "nexus_similarity_search_api_score_candidate_pair_significance",
    {
      title: "Single-Pair NMI Significance Score",
      description: `Computes the hybrid NMI-cosine score and bootstrap p-value for exactly one (query, candidate) embedding pair. Use when you already have a candidate from an external ANN index and need to validate whether the cosine similarity reflects a real statistical dependency — i.e., post-retrieval significance gating. Do NOT use to rank a corpus (use rank_embeddings_by_nmi_cosine instead); calling this in a loop over hundreds of candidates is wasteful because it cannot amortize discretization costs across the corpus.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ScoreCandidatePairSignificanceInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ vector_a, vector_b, discretization_bins, bootstrap_iterations, nmi_cosine_weight }: ScoreCandidatePairSignificanceInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/pair-significance",
          "POST",
          { vector_a, vector_b, discretization_bins, bootstrap_iterations, nmi_cosine_weight },
        );

        const text = JSON.stringify(output, null, 2);
        const truncated = text.length > CHARACTER_LIMIT;
        const display = truncated
          ? text.slice(0, CHARACTER_LIMIT) +
            `\n\n... [truncated, ${text.length - CHARACTER_LIMIT} more characters. ` +
            `Narrow your query or request fewer results.]`
          : text;

        return {
          content: [{ type: "text" as const, text: display }],
          structuredContent: output,
        };
      } catch (err) {
        if (err instanceof CoreServiceError) {
          return {
            content: [{
              type: "text" as const,
              text: `Error calling nexus_similarity_search_api_score_candidate_pair_significance (${err.code}): ${err.message}` +
                (err.requestId ? ` [request_id=${err.requestId}]` : ""),
            }],
            isError: true,
          };
        }
        throw err;
      }
    },
  );
  server.registerTool(
    "nexus_similarity_search_api_detect_embedding_dimension_redundancy",
    {
      title: "Embedding Dimension Redundancy Detector",
      description: `Identifies redundant dimensions within a single embedding space by computing pairwise NMI across all D*(D-1)/2 dimension pairs of the provided sample vectors, returning clusters of highly dependent dimensions (NMI above threshold) and a suggested reduced dimensionality. Use before building a similarity pipeline to prune embedding dimensions that carry no additional information — reduces downstream NMI estimation variance and cosine noise. Do NOT use for embeddings with D > 256 (quadratic in D); not intended for runtime retrieval calls, only for offline embedding space analysis.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: DetectEmbeddingDimensionRedundancyInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ sample_vectors, redundancy_nmi_threshold, discretization_bins, alpha }: DetectEmbeddingDimensionRedundancyInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/dimension-redundancy",
          "POST",
          { sample_vectors, redundancy_nmi_threshold, discretization_bins, alpha },
        );

        const text = JSON.stringify(output, null, 2);
        const truncated = text.length > CHARACTER_LIMIT;
        const display = truncated
          ? text.slice(0, CHARACTER_LIMIT) +
            `\n\n... [truncated, ${text.length - CHARACTER_LIMIT} more characters. ` +
            `Narrow your query or request fewer results.]`
          : text;

        return {
          content: [{ type: "text" as const, text: display }],
          structuredContent: output,
        };
      } catch (err) {
        if (err instanceof CoreServiceError) {
          return {
            content: [{
              type: "text" as const,
              text: `Error calling nexus_similarity_search_api_detect_embedding_dimension_redundancy (${err.code}): ${err.message}` +
                (err.requestId ? ` [request_id=${err.requestId}]` : ""),
            }],
            isError: true,
          };
        }
        throw err;
      }
    },
  );
  server.registerTool(
    "nexus_similarity_search_api_calibrate_nmi_cosine_weight_for_corpus",
    {
      title: "NMI-Cosine Weight Calibrator",
      description: `Given a labeled calibration set of (query, positive_candidate, negative_candidate) triplets and their embeddings, finds the optimal nmi_cosine_weight w that maximizes separation between positive and negative pairs under the hybrid scoring function, reporting the optimal w with its bootstrap confidence interval and the resulting AUC-ROC. Use once before deploying rank_embeddings_by_nmi_cosine on a specific embedding model and domain to select the best w rather than using the default 0.5. Do NOT use as a runtime call per request — this is a one-time offline calibration step. Requires labeled triplets; if no labels are available, skip and use the default weight.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: CalibrateNmiCosineWeightForCorpusInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ triplets, weight_search_grid_size, discretization_bins, bootstrap_iterations }: CalibrateNmiCosineWeightForCorpusInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/calibrate-hybrid-weight",
          "POST",
          { triplets, weight_search_grid_size, discretization_bins, bootstrap_iterations },
        );

        const text = JSON.stringify(output, null, 2);
        const truncated = text.length > CHARACTER_LIMIT;
        const display = truncated
          ? text.slice(0, CHARACTER_LIMIT) +
            `\n\n... [truncated, ${text.length - CHARACTER_LIMIT} more characters. ` +
            `Narrow your query or request fewer results.]`
          : text;

        return {
          content: [{ type: "text" as const, text: display }],
          structuredContent: output,
        };
      } catch (err) {
        if (err instanceof CoreServiceError) {
          return {
            content: [{
              type: "text" as const,
              text: `Error calling nexus_similarity_search_api_calibrate_nmi_cosine_weight_for_corpus (${err.code}): ${err.message}` +
                (err.requestId ? ` [request_id=${err.requestId}]` : ""),
            }],
            isError: true,
          };
        }
        throw err;
      }
    },
  );
}
