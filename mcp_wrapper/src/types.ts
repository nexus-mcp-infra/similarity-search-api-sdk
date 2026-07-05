/**
 * Type definitions — shared across tools and the core client.
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
  query_embedding: number[];
  candidate_embeddings: number[][];
  domain: string;
  top_k?: number | undefined;
  return_scores?: boolean | undefined;
}

export interface ComputePairwiseNmiCosineMatrixInput {
  embeddings: number[][];
  domain: string;
  normalize_output?: boolean | undefined;
}

export interface ScoreEmbeddingPairNmiCosineInput {
  embedding_a: number[];
  embedding_b: number[];
  domain: string;
}

export interface CalibrateDomainNmiCosineWeightsInput {
  anchor_embeddings: number[][];
  positive_embeddings: number[][];
  negative_embeddings: number[][];
  domain_label: string;
}

export interface ExplainNmiCosineRankDivergenceInput {
  query_embedding: number[];
  candidate_embeddings: number[][];
  domain: string;
  top_k?: number | undefined;
}
