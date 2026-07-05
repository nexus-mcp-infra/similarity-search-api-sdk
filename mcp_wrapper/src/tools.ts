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
  RankHybridSimilarityInputSchema,   ComputePairwiseHybridMatrixInputSchema,   ExplainFeatureWeightCalibrationInputSchema,   FilterByHybridThresholdInputSchema,   DetectFeatureTypeSchemaInputSchema,
} from "./schemas.js";
import type {
  RankHybridSimilarityInput,   ComputePairwiseHybridMatrixInput,   ExplainFeatureWeightCalibrationInput,   FilterByHybridThresholdInput,   DetectFeatureTypeSchemaInput,
} from "./types.js";

export function registerTools(server: McpServer): void {

  server.registerTool(
    "nexus_similarity_search_api_rank_hybrid_similarity",
    {
      title: "Hybrid NMI+Cosine Ranking",
      description: `Ranks a corpus of records by hybrid similarity to a query record, fusing NMI for categorical/discrete features and cosine similarity for continuous features into a single adaptive-weighted score. Use when your payload contains a mix of categorical and numeric fields and you need ranked results with per-component score explanation. Do NOT use for pure text embedding search, for corpora larger than 50,000 records per call, or when you need persistent index storage between calls.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: RankHybridSimilarityInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query, corpus, top_k, score_breakdown }: RankHybridSimilarityInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank",
          "POST",
          { query, corpus, top_k, score_breakdown },
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
              text: `Error calling nexus_similarity_search_api_rank_hybrid_similarity (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_compute_pairwise_hybrid_matrix",
    {
      title: "Pairwise Hybrid Similarity Matrix",
      description: `Computes the full N×N hybrid similarity matrix for a set of records using the same NMI+cosine adaptive fusion. Use for clustering preprocessing, graph construction, or any workflow that needs all pairwise distances in one call. Do NOT use when N > 2000 (O(N^2) complexity becomes prohibitive) or when you only need a single query ranked against a corpus — use rank_hybrid_similarity instead.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ComputePairwiseHybridMatrixInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ records, include_diagonal }: ComputePairwiseHybridMatrixInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/matrix",
          "POST",
          { records, include_diagonal },
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
              text: `Error calling nexus_similarity_search_api_compute_pairwise_hybrid_matrix (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_explain_feature_weight_calibration",
    {
      title: "Feature Weight Calibration Explanation",
      description: `Given a sample of records, returns the adaptive weight assigned to each feature under the NMI+cosine fusion model: marginal entropy for categoricals, L2 norm variance for numerics, and the resulting nmi_weight/cosine_weight split. Use for auditing or understanding why the model weights features as it does before running a large ranking job. Do NOT use as a substitute for actual similarity computation — weights here are descriptive, not prescriptive overrides.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ExplainFeatureWeightCalibrationInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ sample_records }: ExplainFeatureWeightCalibrationInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/calibration",
          "POST",
          { sample_records },
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
              text: `Error calling nexus_similarity_search_api_explain_feature_weight_calibration (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_filter_by_hybrid_threshold",
    {
      title: "Threshold-Based Hybrid Similarity Filter",
      description: `Returns all corpus records whose hybrid similarity to the query exceeds a minimum threshold, without ranking. Use when you need a membership decision (similar/not-similar) rather than a ranked list, e.g., deduplication, near-duplicate detection, or candidate set construction. Do NOT use when you need a ranked ordering — rank_hybrid_similarity is more appropriate. Do NOT use with thresholds below 0.05 on high-cardinality categorical corpora, as recall will be near-total and the result set will be unmanageably large.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: FilterByHybridThresholdInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query, corpus, min_hybrid_score, score_breakdown }: FilterByHybridThresholdInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/filter",
          "POST",
          { query, corpus, min_hybrid_score, score_breakdown },
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
              text: `Error calling nexus_similarity_search_api_filter_by_hybrid_threshold (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_detect_feature_type_schema",
    {
      title: "Automatic Feature Type Detection",
      description: `Analyzes a sample of records and returns the inferred type (categorical or continuous) and entropy/variance statistics for each feature key, exactly as the similarity engine would classify them internally. Use before a ranking or filtering call to validate that the engine will treat your features as intended, especially for ambiguous fields (e.g., integer codes that should be categorical). Do NOT use as a general-purpose schema inference tool — it only classifies features into the two types the similarity model supports.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: DetectFeatureTypeSchemaInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ sample_records, override_types }: DetectFeatureTypeSchemaInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/schema",
          "POST",
          { sample_records, override_types },
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
              text: `Error calling nexus_similarity_search_api_detect_feature_type_schema (${err.code}): ${err.message}` +
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
