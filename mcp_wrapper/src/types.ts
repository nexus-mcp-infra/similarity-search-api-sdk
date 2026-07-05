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

export interface RankVectorsByNmiCosineInput {
  query_vector: number[];
  corpus_vectors: number[][];
  nmi_threshold?: number | undefined;
  top_k?: number | undefined;
  confidence_level?: number | undefined;
}

export interface ComputeTokenizedCorpusSimilarityInput {
  query_token_features: number[];
  corpus_token_features: number[][];
  nmi_bins?: number | undefined;
  top_k?: number | undefined;
  confidence_level?: number | undefined;
}

export interface ExtractNmiFeatureWeightsInput {
  query_vector: number[];
  corpus_vectors: number[][];
  nmi_bins?: number | undefined;
  return_top_n_dimensions?: number | undefined;
}

export interface CompareTabularRowSimilarityInput {
  query_row: number[];
  corpus_rows: number[][];
  column_names?: string[] | undefined;
  nmi_threshold?: number | undefined;
  top_k?: number | undefined;
  confidence_level?: number | undefined;
}

export interface EstimateSimilarityConfidenceBandInput {
  similarity_scores: number[];
  nmi_weight_distribution: number[];
  confidence_level?: number | undefined;
  bootstrap_iterations?: number | undefined;
}
