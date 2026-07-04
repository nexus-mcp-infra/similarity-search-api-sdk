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

export interface RankByNmiCosineHybridInput {
  query: string;
  corpus: string[];
  data_modality: string;
  top_k?: number | undefined;
  nmi_weight_override?: number | undefined;
}

export interface ComputePairwiseNmiMatrixInput {
  items: string[];
  data_modality: string;
  return_marginal_entropies?: boolean | undefined;
}

export interface EstimateCorpusEntropyProfileInput {
  corpus: string[];
  data_modality: string;
}

export interface ScoreCandidatePairNmiCosineInput {
  item_a: string;
  item_b: string;
  data_modality: string;
  nmi_weight_override?: number | undefined;
}

export interface BatchRankMultiqueryNmiCosineInput {
  queries: string[];
  corpus: string[];
  data_modality: string;
  top_k?: number | undefined;
  nmi_weight_override?: number | undefined;
}
