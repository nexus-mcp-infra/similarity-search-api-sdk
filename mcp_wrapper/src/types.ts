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

export interface RankHybridSimilarityInput {
  query: string;
  corpus: string;
  top_k?: number | undefined;
  score_breakdown?: boolean | undefined;
}

export interface ComputePairwiseHybridMatrixInput {
  records: string;
  include_diagonal?: boolean | undefined;
}

export interface ExplainFeatureWeightCalibrationInput {
  sample_records: string;
}

export interface FilterByHybridThresholdInput {
  query: string;
  corpus: string;
  min_hybrid_score: number;
  score_breakdown?: boolean | undefined;
}

export interface DetectFeatureTypeSchemaInput {
  sample_records: string;
  override_types?: string | undefined;
}
