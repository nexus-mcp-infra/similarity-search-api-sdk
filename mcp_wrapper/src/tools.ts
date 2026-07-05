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
  RankByNmiWeightedCosineInputSchema,   ComputePairwiseNmiMatrixInputSchema,   ScoreHeterogeneousPairInputSchema,   FilterCandidatesByNmiThresholdInputSchema,   BenchmarkNmiVsCosineDeltaInputSchema,
} from "./schemas.js";
import type {
  RankByNmiWeightedCosineInput,   ComputePairwiseNmiMatrixInput,   ScoreHeterogeneousPairInput,   FilterCandidatesByNmiThresholdInput,   BenchmarkNmiVsCosineDeltaInput,
} from "./types.js";

export function registerTools(server: McpServer): void {

  server.registerTool(
    "nexus_similarity_search_api_rank_by_nmi_weighted_cosine",
    {
      title: "NMI-Weighted Cosine Ranking",
      description: `Computes NMI-weighted cosine similarity between a query vector and a candidate collection, returning ranked results in a single stateless HTTP call. Use when the feature space is heterogeneous (mixed numeric, categorical, or multimodal) and standard cosine underperforms due to non-linear dependencies. Do NOT use for pure text embeddings where cosine is sufficient, or when you need approximate nearest-neighbor at >1M candidates per call (latency will degrade beyond practical limits).

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: RankByNmiWeightedCosineInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_vector, candidate_vectors, candidate_ids, nmi_bins, top_k }: RankByNmiWeightedCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-nmi-cosine",
          "POST",
          { query_vector, candidate_vectors, candidate_ids, nmi_bins, top_k },
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
              text: `Error calling nexus_similarity_search_api_rank_by_nmi_weighted_cosine (${err.code}): ${err.message}` +
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
      title: "Pairwise NMI Feature Matrix",
      description: `Computes the full NMI matrix across feature dimensions for a given collection, exposing per-feature mutual information weights without performing a search. Use this to inspect which feature dimensions carry the most discriminative mutual information before invoking rank_by_nmi_weighted_cosine, or to debug why certain features dominate weighting. Do NOT use as a replacement for the ranked search — this returns a DxD weight matrix, not similarity scores between items.

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
    async ({ vectors, nmi_bins, normalize_weights }: ComputePairwiseNmiMatrixInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/nmi-feature-matrix",
          "POST",
          { vectors, nmi_bins, normalize_weights },
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
    "nexus_similarity_search_api_score_heterogeneous_pair",
    {
      title: "Single-Pair NMI-Cosine Score",
      description: `Computes the NMI-weighted cosine similarity score for exactly one query-candidate pair with full per-dimension weight breakdown. Use for explainability: when you need to understand why two specific items score high or low, or to validate that NMI weighting is behaving correctly on a known pair. Do NOT use in batch loops to simulate collection ranking — rank_by_nmi_weighted_cosine is vectorized and orders of magnitude faster for that purpose.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ScoreHeterogeneousPairInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ vector_a, vector_b, nmi_bins, return_dimension_weights }: ScoreHeterogeneousPairInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/score-pair",
          "POST",
          { vector_a, vector_b, nmi_bins, return_dimension_weights },
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
              text: `Error calling nexus_similarity_search_api_score_heterogeneous_pair (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_filter_candidates_by_nmi_threshold",
    {
      title: "NMI-Threshold Candidate Filter",
      description: `Returns only candidates whose NMI-weighted cosine score meets or exceeds a minimum threshold, without a fixed top_k cutoff. Use when downstream logic requires a quality floor rather than a fixed result count — e.g., deduplication pipelines, semantic clustering seeding, or anomaly filtering where you want all sufficiently similar items. Do NOT use when you need exactly K results; use rank_by_nmi_weighted_cosine with top_k instead.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: FilterCandidatesByNmiThresholdInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_vector, candidate_vectors, candidate_ids, min_score_threshold, nmi_bins }: FilterCandidatesByNmiThresholdInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/filter-by-threshold",
          "POST",
          { query_vector, candidate_vectors, candidate_ids, min_score_threshold, nmi_bins },
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
              text: `Error calling nexus_similarity_search_api_filter_candidates_by_nmi_threshold (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_benchmark_nmi_vs_cosine_delta",
    {
      title: "NMI vs Cosine Ranking Delta",
      description: `Runs both pure cosine and NMI-weighted cosine ranking on the same query and candidates, returning side-by-side rank positions and score deltas per candidate. Use to quantify how much NMI weighting changes ranking order for a specific dataset — critical for justifying adoption to stakeholders or detecting feature spaces where cosine already suffices. Do NOT use in production ranking pipelines — this endpoint is designed for evaluation and costs roughly 2x the compute of a single ranking call.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: BenchmarkNmiVsCosineDeltaInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_vector, candidate_vectors, candidate_ids, nmi_bins, top_k }: BenchmarkNmiVsCosineDeltaInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/benchmark-nmi-cosine-delta",
          "POST",
          { query_vector, candidate_vectors, candidate_ids, nmi_bins, top_k },
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
              text: `Error calling nexus_similarity_search_api_benchmark_nmi_vs_cosine_delta (${err.code}): ${err.message}` +
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
