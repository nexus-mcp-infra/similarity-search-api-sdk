import { z } from "zod";

export const RankByNmiCosineHybridInputSchema = z.object({
  query: z.string().min(1).max(8192).describe("Raw query item: plain text, a comma-separated categorical sequence, or a JSON-encoded integer array representing a discrete time series. Must be non-empty and consistent in type with corpus items."),
  corpus: z.array(z.string()).describe("Array of raw corpus items to rank. Each item must be the same type as query. Minimum 2 items required to compute marginal entropy weighting. Maximum 10,000 items per call."),
  data_modality: z.string().min(4).max(20).describe("Declares the semantic type of query and corpus items for tokenization and joint distribution estimation. One of: 'text' (whitespace-tokenized), 'categorical' (comma-separated discrete labels), 'discrete_timeseries' (JSON int array, binned by Sturges rule). Required \u2014 do not guess."),
  top_k: z.number().min(1).max(500).default(10).describe("Number of top-ranked results to return, sorted descending by hybrid score. Must be between 1 and min(corpus length, 500)."),
  nmi_weight_override: z.number().min(0.0).max(1.0).optional().describe("Fixed weight for the NMI component in [0.0, 1.0]; cosine weight becomes 1 - nmi_weight_override. If null, weight is computed adaptively from corpus marginal entropy: high-entropy corpora increase NMI weight. Provide a value only when you have a domain-specific reason to override entropy-based calibration."),
}).strict();

export const ComputePairwiseNmiMatrixInputSchema = z.object({
  items: z.array(z.string()).describe("Array of raw items for which to compute pairwise NMI. All items must share the same modality. Minimum 2, maximum 500."),
  data_modality: z.string().min(4).max(20).describe("Tokenization and binning strategy: 'text', 'categorical', or 'discrete_timeseries'. Must match the actual format of items."),
  return_marginal_entropies: z.boolean().default(true).describe("If true, includes per-item marginal entropy H(X_i) in the response alongside the NMI matrix. Useful for identifying low-entropy items that may distort similarity scores."),
}).strict();

export const EstimateCorpusEntropyProfileInputSchema = z.object({
  corpus: z.array(z.string()).describe("Raw corpus items to profile. Same format constraints as rank_by_nmi_cosine_hybrid. Minimum 2, maximum 10,000."),
  data_modality: z.string().min(4).max(20).describe("Tokenization strategy: 'text', 'categorical', or 'discrete_timeseries'."),
}).strict();

export const ScoreCandidatePairNmiCosineInputSchema = z.object({
  item_a: z.string().min(1).max(8192).describe("First raw item of the pair. Non-empty, same modality as item_b."),
  item_b: z.string().min(1).max(8192).describe("Second raw item of the pair. Non-empty, same modality as item_a."),
  data_modality: z.string().min(4).max(20).describe("Tokenization strategy: 'text', 'categorical', or 'discrete_timeseries'."),
  nmi_weight_override: z.number().min(0.0).max(1.0).optional().describe("Fixed NMI weight in [0.0, 1.0]. If null, inferred from pair-level joint entropy (less accurate than corpus-level; prefer explicit override for isolated pair scoring)."),
}).strict();

export const BatchRankMultiqueryNmiCosineInputSchema = z.object({
  queries: z.array(z.string()).describe("Array of raw query items, all of the same modality. Minimum 2, maximum 50 queries per batch call."),
  corpus: z.array(z.string()).describe("Shared corpus to rank against all queries. Minimum 2, maximum 10,000 items."),
  data_modality: z.string().min(4).max(20).describe("Tokenization strategy applied uniformly to all queries and corpus items: 'text', 'categorical', or 'discrete_timeseries'."),
  top_k: z.number().min(1).max(500).default(10).describe("Number of top-ranked corpus items returned per query. Applied uniformly across all queries. Between 1 and min(corpus length, 500)."),
  nmi_weight_override: z.number().min(0.0).max(1.0).optional().describe("Fixed NMI weight in [0.0, 1.0] applied to all queries. If null, corpus marginal entropy determines the weight once and reuses it across all queries in the batch."),
}).strict();
