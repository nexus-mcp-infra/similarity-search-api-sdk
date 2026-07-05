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
  RankEmbeddingsByNmiCosineInputSchema,   ComputePairwiseNmiCosineMatrixInputSchema,   ScoreEmbeddingPairNmiCosineInputSchema,   CalibrateDomainNmiCosineWeightsInputSchema,   ExplainNmiCosineRankDivergenceInputSchema,
} from "./schemas.js";
import type {
  RankEmbeddingsByNmiCosineInput,   ComputePairwiseNmiCosineMatrixInput,   ScoreEmbeddingPairNmiCosineInput,   CalibrateDomainNmiCosineWeightsInput,   ExplainNmiCosineRankDivergenceInput,
} from "./types.js";

export function registerTools(server: McpServer): void {

  server.registerTool(
    "nexus_similarity_search_api_rank_embeddings_by_nmi_cosine",
    {
      title: "NMI+Cosine Composite Ranking",
      description: `Ranks a set of candidate embeddings against a query embedding using a weighted NMI+Cosine composite score calibrated per domain. Use when you need stateless semantic similarity ranking without a vector index or upsert step, especially when candidate correlations are non-linear. Do NOT use if you have a persistent vector store already indexed — latency will be higher than ANN retrieval for corpora above 50k vectors.

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
    async ({ query_embedding, candidate_embeddings, domain, top_k, return_scores }: RankEmbeddingsByNmiCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank",
          "POST",
          { query_embedding, candidate_embeddings, domain, top_k, return_scores },
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
    "nexus_similarity_search_api_compute_pairwise_nmi_cosine_matrix",
    {
      title: "Pairwise NMI+Cosine Matrix",
      description: `Computes the full N×N composite similarity matrix for a set of embeddings. Use for clustering preprocessing, graph construction, or reranking pipelines where every pair needs a score. Do NOT use for query-vs-corpus ranking (use rank_embeddings_by_nmi_cosine instead) — cost is O(N²) and grows quadratically.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ComputePairwiseNmiCosineMatrixInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ embeddings, domain, normalize_output }: ComputePairwiseNmiCosineMatrixInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/pairwise-matrix",
          "POST",
          { embeddings, domain, normalize_output },
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
              text: `Error calling nexus_similarity_search_api_compute_pairwise_nmi_cosine_matrix (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_score_embedding_pair_nmi_cosine",
    {
      title: "Single-Pair NMI+Cosine Score",
      description: `Returns the decomposed composite similarity score (NMI component, cosine component, weighted composite) for exactly one pair of embeddings. Use for debugging, threshold calibration, or audit trails where you need interpretable component breakdown. Do NOT use in batch loops — use rank_embeddings_by_nmi_cosine or compute_pairwise_nmi_cosine_matrix for multiple pairs; per-call overhead makes looping expensive.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ScoreEmbeddingPairNmiCosineInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ embedding_a, embedding_b, domain }: ScoreEmbeddingPairNmiCosineInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/pair-score",
          "POST",
          { embedding_a, embedding_b, domain },
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
              text: `Error calling nexus_similarity_search_api_score_embedding_pair_nmi_cosine (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_calibrate_domain_nmi_cosine_weights",
    {
      title: "Domain Weight Calibration",
      description: `Derives optimal alpha_nmi and alpha_cosine weights for a custom embedding domain by fitting the composite scorer to a labeled relevance dataset you supply. Use when 'text', 'image', and 'tabular' presets underperform on your specific embedding model or corpus distribution. Do NOT use at inference time — run once offline and cache the returned weight profile; recalibrate only when the embedding model changes.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: CalibrateDomainNmiCosineWeightsInputSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: false,
      },
    },
    async ({ anchor_embeddings, positive_embeddings, negative_embeddings, domain_label }: CalibrateDomainNmiCosineWeightsInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/calibrate-weights",
          "POST",
          { anchor_embeddings, positive_embeddings, negative_embeddings, domain_label },
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
              text: `Error calling nexus_similarity_search_api_calibrate_domain_nmi_cosine_weights (${err.code}): ${err.message}` +
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
    "nexus_similarity_search_api_explain_nmi_cosine_rank_divergence",
    {
      title: "NMI vs Cosine Rank Divergence Report",
      description: `Given a query and candidates, returns a divergence report showing where NMI-informed ranking differs from pure-cosine ranking and why — quantifying non-linear dependency contribution per candidate. Use when auditing model behavior, justifying ranking decisions to stakeholders, or diagnosing unexpected rank positions. Do NOT use in latency-sensitive inference paths — this runs both rankers plus divergence attribution and is 2-3x slower than rank_embeddings_by_nmi_cosine alone.

Returns (JSON):
{
  "status": "ok",
  "results": []
}

Error handling:
  - Throws with a clear message if the core service is unreachable, times out,
    or returns a non-2xx status. The message includes the upstream request_id
    when available, for support correlation.`,
      inputSchema: ExplainNmiCosineRankDivergenceInputSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ query_embedding, candidate_embeddings, domain, top_k }: ExplainNmiCosineRankDivergenceInput) => {
      try {
        const output = await callCore<Record<string, unknown>>(
          "/v1/similarity/rank-divergence",
          "POST",
          { query_embedding, candidate_embeddings, domain, top_k },
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
              text: `Error calling nexus_similarity_search_api_explain_nmi_cosine_rank_divergence (${err.code}): ${err.message}` +
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
