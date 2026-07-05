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

export interface RankByNmiWeightedCosineInput {
  query_vector: number[];
  candidate_vectors: number[][];
  candidate_ids: string[];
  nmi_bins?: number | undefined;
  top_k?: number | undefined;
}

export interface ComputePairwiseNmiMatrixInput {
  vectors: number[][];
  nmi_bins?: number | undefined;
  normalize_weights?: boolean | undefined;
}

export interface ScoreHeterogeneousPairInput {
  vector_a: number[];
  vector_b: number[];
  nmi_bins?: number | undefined;
  return_dimension_weights?: boolean | undefined;
}

export interface FilterCandidatesByNmiThresholdInput {
  query_vector: number[];
  candidate_vectors: number[][];
  candidate_ids: string[];
  min_score_threshold: number;
  nmi_bins?: number | undefined;
}

export interface BenchmarkNmiVsCosineDeltaInput {
  query_vector: number[];
  candidate_vectors: number[][];
  candidate_ids: string[];
  nmi_bins?: number | undefined;
  top_k?: number | undefined;
}
