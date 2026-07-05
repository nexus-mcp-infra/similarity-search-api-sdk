import { z } from "zod";

export const RankVectorsByNmiCosineInputSchema = z.object({
  query_vector: z.array(z.number()).min(2).max(8192).describe("Query vector as a flat list of floats. Must match dimensionality of all corpus vectors."),
  corpus_vectors: z.array(z.array(z.number())).describe("List of candidate vectors to rank. Each inner list must have the same length as query_vector. Maximum 50,000 rows."),
  nmi_threshold: z.number().min(0.0).max(1.0).default(0.1).describe("Minimum NMI score [0.0\u20131.0] a feature dimension must achieve to be included in cosine computation. Higher values are more aggressive at dropping noisy features. Recommended range 0.05\u20130.30."),
  top_k: z.number().min(1).max(50000).default(10).describe("Number of top-ranked results to return. Must be between 1 and the corpus size."),
  confidence_level: z.number().min(0.8).max(0.99).default(0.95).describe("Confidence level for the returned similarity score interval, derived from empirical NMI distribution. Typical values: 0.90, 0.95, 0.99."),
}).strict();

export const ComputeTokenizedCorpusSimilarityInputSchema = z.object({
  query_token_features: z.array(z.number()).min(2).max(131072).describe("Query document represented as a feature vector over vocabulary (e.g., TF-IDF weights or binary term presence). Length defines vocabulary size."),
  corpus_token_features: z.array(z.array(z.number())).describe("Corpus documents as feature vectors over the same vocabulary as query_token_features. All rows must share vocabulary length."),
  nmi_bins: z.number().min(5).max(50).default(10).describe("Number of histogram bins used when discretizing continuous feature values to compute NMI. Higher bins increase NMI resolution but add compute cost. Meaningful range: 5\u201350."),
  top_k: z.number().min(1).max(20000).default(10).describe("Number of top-ranked results to return."),
  confidence_level: z.number().min(0.8).max(0.99).default(0.95).describe("Confidence level for the returned per-result similarity confidence interval."),
}).strict();

export const ExtractNmiFeatureWeightsInputSchema = z.object({
  query_vector: z.array(z.number()).min(2).max(8192).describe("Query vector defining the reference distribution for NMI computation across corpus dimensions."),
  corpus_vectors: z.array(z.array(z.number())).describe("Corpus used to estimate the empirical feature distributions needed for NMI. Same shape constraints as rank_vectors_by_nmi_cosine."),
  nmi_bins: z.number().min(5).max(50).default(10).describe("Number of histogram bins for NMI discretization. Consistent with the bins used in downstream ranking calls."),
  return_top_n_dimensions: z.number().min(1).max(8192).optional().describe("Return only the top-N highest-NMI feature indices and their scores, sorted descending. Returns all dimensions if omitted."),
}).strict();

export const CompareTabularRowSimilarityInputSchema = z.object({
  query_row: z.array(z.number()).min(2).max(4096).describe("Single query row as a list of numerically encoded feature values. Length must equal number of columns in corpus_rows."),
  corpus_rows: z.array(z.array(z.number())).describe("Tabular corpus where each row is a candidate record with the same feature schema as query_row."),
  column_names: z.array(z.string()).optional().describe("Optional column labels for each feature position. Used only for interpretability in the response payload \u2014 does not affect computation. Must match query_row length if provided."),
  nmi_threshold: z.number().min(0.0).max(1.0).default(0.05).describe("Minimum NMI score for a column to participate in cosine similarity. For tabular data with many correlated columns, values of 0.05\u20130.15 are typical."),
  top_k: z.number().min(1).max(50000).default(10).describe("Number of most-similar rows to return, ranked by NMI-weighted cosine score."),
  confidence_level: z.number().min(0.8).max(0.99).default(0.95).describe("Confidence level for similarity score intervals."),
}).strict();

export const EstimateSimilarityConfidenceBandInputSchema = z.object({
  similarity_scores: z.array(z.number()).min(1).max(50000).describe("List of raw NMI-weighted cosine similarity scores from a prior ranking call, in [\u22121.0, 1.0]."),
  nmi_weight_distribution: z.array(z.number()).min(2).max(131072).describe("Per-dimension NMI weights used to produce the scores. Returned by extract_nmi_feature_weights or included in rank_vectors_by_nmi_cosine response."),
  confidence_level: z.number().min(0.8).max(0.99).default(0.95).describe("Target confidence level for the output bands. Can differ from the level used in the original ranking call."),
  bootstrap_iterations: z.number().min(100).max(5000).default(500).describe("Number of bootstrap resampling iterations for band estimation. Higher values reduce Monte Carlo variance at compute cost. Range 100\u20135000."),
}).strict();
