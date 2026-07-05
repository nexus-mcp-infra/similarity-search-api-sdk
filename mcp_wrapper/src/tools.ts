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
  RankVectorsByNmiCosineInputSchema,   ComputeTokenizedCorpusSimilarityInputSchema,   ExtractNmiFeatureWeightsInputSchema,   CompareTabularRowSimilarityInputSchema,   EstimateSimilarityConfidenceBandInputSchema,
} from "./schemas.js";
import type {
  RankVectorsByNmiCosineInput,   ComputeTokenizedCorpusSimilarityInput,   ExtractNmiFeatureWeightsInput,   CompareTabularRowSimilarityInput,   EstimateSimilarityConfidenceBandInput,
} from "./types.js";

export function registerTools(server: McpServer): void {

  server.registerTool(
    "nexus_similarity_search_api_rank_vectors_by_nmi_cosine",
    {
      title: "NMI-Weighted Vector Ranking",
      description: `Ranks a corpus of vectors against a query vector using NMI-filtered cosine similarity. NMI is computed per-feature across the corpus to suppress noisy dimensions before distance calculation. Use this when you have raw numerical feature vectors and want ranked similarity with confidence intervals. Do NOT use for text inputs (use tokenized_corpus_similarity instead), and do NOT use when corpus size exceeds 50,000 vectors per call — batch instead.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: RankVectorsByNmiCosineInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_vector, corpus_vectors, nmi_threshold, top_k, confidence_level }: RankVectorsByNmiCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-vectors",
          "POST",
          { query_vector, corpus_vectors, nmi_threshold, top_k, confidence_level },
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
              text: `Error calling nexus_similarity_search_api_rank_vectors_by_nmi_cosine (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_compute_tokenized_corpus_similarity",
    {
      title: "Tokenized Text NMI-Cosine Similarity",
      description: `Accepts pre-tokenized text sequences (lists of token IDs or term-frequency feature arrays) and ranks them against a query sequence using NMI-weighted cosine. NMI is computed over the token co-occurrence feature space to prune uninformative vocabulary dimensions. Use this for sparse text feature vectors or BoW/TF-IDF representations. Do NOT use with dense embedding vectors (use rank_vectors_by_nmi_cosine) and do NOT pass raw strings — tokenize first.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ComputeTokenizedCorpusSimilarityInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_token_features, corpus_token_features, nmi_bins, top_k, confidence_level }: ComputeTokenizedCorpusSimilarityInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-tokenized-corpus",
          "POST",
          { query_token_features, corpus_token_features, nmi_bins, top_k, confidence_level },
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
              text: `Error calling nexus_similarity_search_api_compute_tokenized_corpus_similarity (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_extract_nmi_feature_weights",
    {
      title: "NMI Feature Relevance Extractor",
      description: `Computes per-dimension NMI scores between a query vector and a corpus, returning the weight assigned to each feature dimension without performing similarity ranking. Use this to audit which features drive the NMI-cosine score, to tune nmi_threshold before a full ranking call, or to pre-validate corpus quality. Do NOT use as a substitute for ranking — this endpoint returns weights only, not similarity scores.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ExtractNmiFeatureWeightsInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_vector, corpus_vectors, nmi_bins, return_top_n_dimensions }: ExtractNmiFeatureWeightsInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/nmi-feature-weights",
          "POST",
          { query_vector, corpus_vectors, nmi_bins, return_top_n_dimensions },
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
              text: `Error calling nexus_similarity_search_api_extract_nmi_feature_weights (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_compare_tabular_row_similarity",
    {
      title: "Tabular Row NMI-Cosine Comparator",
      description: `Ranks rows in a tabular dataset (mixed numerical features, pre-encoded) against a query row using NMI-weighted cosine. Designed for structured tabular data where feature columns have heterogeneous scales. Handles per-column z-score normalization internally before NMI and cosine steps. Use for structured ML datasets, relational feature tables, or any input that originates as a dataframe row. Do NOT use for raw text or image embeddings — those have different distribution assumptions.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: CompareTabularRowSimilarityInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_row, corpus_rows, column_names, nmi_threshold, top_k, confidence_level }: CompareTabularRowSimilarityInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-tabular-rows",
          "POST",
          { query_row, corpus_rows, column_names, nmi_threshold, top_k, confidence_level },
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
              text: `Error calling nexus_similarity_search_api_compare_tabular_row_similarity (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_estimate_similarity_confidence_band",
    {
      title: "Similarity Score Confidence Band Estimator",
      description: `Given a set of pre-computed NMI-cosine similarity scores and the underlying NMI weight distribution, returns bootstrap-derived confidence bands for each score. Use this when you already have scores from a prior ranking call and want to widen or narrow confidence intervals at a different confidence level, or to validate stability of results under resampling. Do NOT use as a primary similarity computation — this endpoint takes scores as input, not raw vectors.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: EstimateSimilarityConfidenceBandInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ similarity_scores, nmi_weight_distribution, confidence_level, bootstrap_iterations }: EstimateSimilarityConfidenceBandInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/confidence-band",
          "POST",
          { similarity_scores, nmi_weight_distribution, confidence_level, bootstrap_iterations },
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
              text: `Error calling nexus_similarity_search_api_estimate_similarity_confidence_band (${err.code}): ${err.message}` +
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
