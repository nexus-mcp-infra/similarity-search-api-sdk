/**
 * Type definitions -- shared across tools and the core client.
 *
 * --- NEXUS PATCH mcp_tool_grounding_similarity_search ---
 * Only RankEmbeddingsByNmiCosineInput survives grounding; see tools.ts
 * and schemas.ts for context.
 */

export interface CoreErrorBody {
  error?: {
    code?: string;
    message?: string;
    request_id?: string;
  };
}

export class CoreServiceError extends Error {
  constructor(
    message: string,
    public readonly status: number | null,
    public readonly code: string,
    public readonly requestId?: string,
  ) {
    super(message);
    this.name = "CoreServiceError";
  }
}

export interface RankEmbeddingsByNmiCosineInput {
  query_vector: number[];
  corpus_vectors: number[][];
  corpus_ids: string[];
  top_k?: number | undefined;
  nmi_bins?: number | undefined;
  alpha_override?: number | undefined;
}
