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
  RankByNmiCosineHybridInputSchema,   ComputePairwiseNmiMatrixInputSchema,   EstimateCorpusEntropyProfileInputSchema,   ScoreCandidatePairNmiCosineInputSchema,   BatchRankMultiqueryNmiCosineInputSchema,
} from "./schemas.js";
import type {
  RankByNmiCosineHybridInput,   ComputePairwiseNmiMatrixInput,   EstimateCorpusEntropyProfileInput,   ScoreCandidatePairNmiCosineInput,   BatchRankMultiqueryNmiCosineInput,
} from "./types.js";

export function registerTools(server: McpServer): void {

  server.registerTool(
    "nexus_similarity_search_api_rank_by_nmi_cosine_hybrid",
    {
      title: "NMI-Cosine Hybrid Ranking",
      description: `Ranks a corpus of raw items (text, discrete categories, or integer time series) against a query using a weighted hybrid of Normalized Mutual Information and cosine similarity, where weights are determined adaptively by the marginal entropy of the corpus. Use when your data has non-linear statistical dependencies that cosine alone would miss, or when working with categorical/discrete distributions without a prebuilt embedding index. Do NOT use for continuous high-dimensional float vectors already embedded — cosine-only is faster and sufficient in that case; do NOT use if corpus exceeds 10,000 items per call (use batch endpoint instead).

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: RankByNmiCosineHybridInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query, corpus, data_modality, top_k, nmi_weight_override }: RankByNmiCosineHybridInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-hybrid",
          "POST",
          { query, corpus, data_modality, top_k, nmi_weight_override },
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
              text: `Error calling nexus_similarity_search_api_rank_by_nmi_cosine_hybrid (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_compute_pairwise_nmi_matrix",
    {
      title: "Pairwise NMI Matrix",
      description: `Computes the full N x N Normalized Mutual Information matrix for a set of raw items, returning both the matrix and per-item marginal entropy values. Use for clustering preparation, feature selection diagnostics, or any downstream task that needs the full dependency structure of a corpus rather than a ranked list against a single query. Do NOT use as a ranking primitive — this is O(N^2) and intended for offline analysis, not per-request retrieval. Do NOT call with more than 500 items; use the batch endpoint for larger corpora.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ComputePairwiseNmiMatrixInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ items, data_modality, return_marginal_entropies }: ComputePairwiseNmiMatrixInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/pairwise-nmi-matrix",
          "POST",
          { items, data_modality, return_marginal_entropies },
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
              text: `Error calling nexus_similarity_search_api_compute_pairwise_nmi_matrix (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_estimate_corpus_entropy_profile",
    {
      title: "Corpus Entropy Profile",
      description: `Computes the marginal entropy of each item in a corpus and the joint entropy distribution across the full set, returning the entropy-based NMI weight that rank_by_nmi_cosine_hybrid would apply automatically. Use to audit or preview the adaptive weighting before committing to a ranking call, or to detect degenerate corpora where all items have near-zero entropy (making NMI unreliable). Do NOT use as a substitute for ranking — this endpoint only returns entropy diagnostics, not similarity scores.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: EstimateCorpusEntropyProfileInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ corpus, data_modality }: EstimateCorpusEntropyProfileInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/corpus-entropy-profile",
          "POST",
          { corpus, data_modality },
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
              text: `Error calling nexus_similarity_search_api_estimate_corpus_entropy_profile (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_score_candidate_pair_nmi_cosine",
    {
      title: "Single Pair NMI-Cosine Score",
      description: `Computes the NMI score, cosine score, and hybrid weighted score for exactly one (query, candidate) pair without needing a full corpus. Entropy-based weight is estimated from the pair's joint distribution alone (no corpus marginal). Use for spot-checks, unit-level debugging of scoring logic, or integration tests where you need a deterministic score for a known pair. Do NOT use in production ranking loops — call rank_by_nmi_cosine_hybrid instead, which benefits from corpus-level entropy calibration that this endpoint cannot provide for isolated pairs.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ScoreCandidatePairNmiCosineInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ item_a, item_b, data_modality, nmi_weight_override }: ScoreCandidatePairNmiCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/score-pair",
          "POST",
          { item_a, item_b, data_modality, nmi_weight_override },
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
              text: `Error calling nexus_similarity_search_api_score_candidate_pair_nmi_cosine (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_batch_rank_multiquery_nmi_cosine",
    {
      title: "Multi-Query Batch Hybrid Ranking",
      description: `Executes NMI-Cosine hybrid ranking for multiple queries against a shared corpus in a single request, sharing corpus entropy computation across all queries to reduce total cost. Use when you need to rank the same corpus against 2 or more queries simultaneously (e.g., multi-faceted retrieval, ensemble query expansion). Do NOT use for a single query — rank_by_nmi_cosine_hybrid is cheaper. Do NOT use when each query has a different corpus — this endpoint assumes one shared corpus across all queries.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: BatchRankMultiqueryNmiCosineInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ queries, corpus, data_modality, top_k, nmi_weight_override }: BatchRankMultiqueryNmiCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/batch-rank-multiquery",
          "POST",
          { queries, corpus, data_modality, top_k, nmi_weight_override },
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
              text: `Error calling nexus_similarity_search_api_batch_rank_multiquery_nmi_cosine (${err.code}): ${err.message}` +
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
