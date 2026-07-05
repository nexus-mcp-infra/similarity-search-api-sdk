import { z } from "zod";

export const RankByNmiWeightedCosineInputSchema = z.object({
  query_vector: z.array(z.number()).min(2).max(4096).describe("Dense numeric representation of the query item. Must match dimensionality of all candidate vectors."),
  candidate_vectors: z.array(z.array(z.number())).describe("Collection of candidate dense vectors to rank against the query. Each row must have the same length as query_vector."),
  candidate_ids: z.array(z.string()).describe("Stable identifiers for each candidate vector, returned in ranked output. Must be same length as candidate_vectors."),
  nmi_bins: z.number().min(3).max(50).default(10).describe("Number of histogram bins used when estimating marginal and joint distributions for NMI computation per feature dimension. Higher values increase precision for continuous features but raise compute cost. Recommended range 5-20."),
  top_k: z.number().min(1).max(500).default(10).describe("Maximum number of ranked results to return. Results are sorted descending by NMI-weighted cosine score."),
}).strict();

export const ComputePairwiseNmiMatrixInputSchema = z.object({
  vectors: z.array(z.array(z.number())).describe("Collection of vectors from which feature-level NMI weights are derived. Each row is one sample; each column is one feature dimension."),
  nmi_bins: z.number().min(3).max(50).default(10).describe("Histogram bins per feature dimension used to estimate joint and marginal distributions. Increase for continuous features with wide range, decrease for near-binary features."),
  normalize_weights: z.boolean().default(true).describe("If true, returns NMI values normalized to [0,1] per dimension pair. If false, returns raw joint entropy ratios. Set true when you intend to feed output directly into rank_by_nmi_weighted_cosine as custom weights."),
}).strict();

export const ScoreHeterogeneousPairInputSchema = z.object({
  vector_a: z.array(z.number()).min(2).max(4096).describe("First dense vector of the pair. Treated as query in the NMI weight derivation."),
  vector_b: z.array(z.number()).min(2).max(4096).describe("Second dense vector of the pair. Must match dimensionality of vector_a."),
  nmi_bins: z.number().min(3).max(50).default(10).describe("Histogram bins used to estimate per-dimension marginal distributions from the two vectors. Lower values smooth distributions; higher values respect fine structure."),
  return_dimension_weights: z.boolean().default(false).describe("If true, the response includes the NMI weight assigned to each feature dimension, enabling full score decomposition."),
}).strict();

export const FilterCandidatesByNmiThresholdInputSchema = z.object({
  query_vector: z.array(z.number()).min(2).max(4096).describe("Dense query vector. Must match dimensionality of all candidate vectors."),
  candidate_vectors: z.array(z.array(z.number())).describe("Collection of candidate dense vectors to evaluate against the threshold."),
  candidate_ids: z.array(z.string()).describe("Stable identifiers for each candidate vector. Must be same length as candidate_vectors."),
  min_score_threshold: z.number().min(0.0).max(1.0).describe("Minimum NMI-weighted cosine score [0.0, 1.0] a candidate must reach to be included in the response. Values below 0.3 may return very large result sets on heterogeneous collections."),
  nmi_bins: z.number().min(3).max(50).default(10).describe("Histogram bins for NMI estimation per feature dimension."),
}).strict();

export const BenchmarkNmiVsCosineDeltaInputSchema = z.object({
  query_vector: z.array(z.number()).min(2).max(4096).describe("Dense query vector used in both cosine and NMI-weighted cosine evaluations."),
  candidate_vectors: z.array(z.array(z.number())).describe("Collection of candidate dense vectors to rank under both metrics."),
  candidate_ids: z.array(z.string()).describe("Stable identifiers for each candidate. Must be same length as candidate_vectors."),
  nmi_bins: z.number().min(3).max(50).default(10).describe("Histogram bins for NMI estimation. Applies only to the NMI-weighted cosine branch of the benchmark."),
  top_k: z.number().min(2).max(200).default(20).describe("Number of top candidates to include in the delta comparison output. Full collection is still ranked internally."),
}).strict();
