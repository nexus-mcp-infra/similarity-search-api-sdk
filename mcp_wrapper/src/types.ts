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
  query_vector: number[];
  corpus_vectors: number[][];
  corpus_ids: string[];
  discretization_bins?: number | undefined;
  bootstrap_iterations?: number | undefined;
  top_k?: number | undefined;
  alpha?: number | undefined;
  filter_insignificant?: boolean | undefined;
  nmi_cosine_weight?: number | undefined;
}

export interface EstimatePairwiseNmiMatrixInput {
  vectors: number[][];
  vector_ids: string[];
  discretization_bins?: number | undefined;
  bootstrap_iterations?: number | undefined;
  alpha?: number | undefined;
}

export interface ScoreCandidatePairSignificanceInput {
  vector_a: number[];
  vector_b: number[];
  discretization_bins?: number | undefined;
  bootstrap_iterations?: number | undefined;
  nmi_cosine_weight?: number | undefined;
}

export interface DetectEmbeddingDimensionRedundancyInput {
  sample_vectors: number[][];
  redundancy_nmi_threshold?: number | undefined;
  discretization_bins?: number | undefined;
  alpha?: number | undefined;
}

export interface CalibrateNmiCosineWeightForCorpusInput {
  triplets: number[][];
  weight_search_grid_size?: number | undefined;
  discretization_bins?: number | undefined;
  bootstrap_iterations?: number | undefined;
}
