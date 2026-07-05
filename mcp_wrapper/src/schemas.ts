import { z } from "zod";

export const RankEmbeddingsByNmiCosineInputSchema = z.object({
  query_vector: z.array(z.number()).min(8).max(4096).describe("Dense embedding vector to rank against the corpus. Must be the same dimensionality as all corpus_vectors. Values should be raw floats, not pre-normalized."),
  corpus_vectors: z.array(z.array(z.number())).describe("List of candidate embedding vectors. Each inner array must share the same dimensionality as query_vector. Minimum 10 vectors required for valid bootstrap estimation."),
  corpus_ids: z.array(z.string()).describe("Opaque identifiers for each vector in corpus_vectors, returned in the ranking response. Length must exactly match corpus_vectors. Use stable external IDs (document IDs, chunk hashes, etc.)."),
  discretization_bins: z.number().min(2).max(30).default(8).describe("Number of equal-width bins used to discretize continuous embedding dimensions before computing joint entropy. Higher values increase NMI resolution but also estimation variance. Recommended range 4-20; values above 30 degrade NMI reliability on typical embedding sizes."),
  bootstrap_iterations: z.number().min(100).max(1000).default(200).describe("Number of bootstrap resampling iterations per pair used to compute the 95% confidence interval on the hybrid score and the p-value under the null of no mutual information. Higher values narrow confidence intervals at linear compute cost. Minimum 100 for valid p-values."),
  top_k: z.number().min(0).max(500).default(10).describe("Number of top-ranked results to return, ordered by descending hybrid score. Must be <= len(corpus_vectors). Set to 0 to return all ranked results."),
  alpha: z.number().min(0.001).max(0.2).default(0.05).describe("Significance level threshold for p-value filtering. Results with p_value > alpha are flagged as statistically insignificant in the response but are still returned unless filter_insignificant is true. Standard values: 0.05 or 0.01."),
  filter_insignificant: z.boolean().default(false).describe("If true, results where p_value > alpha are excluded from the returned ranking entirely. If false, they are included but flagged. Set to false when you need ranked output regardless of significance for downstream reranking."),
  nmi_cosine_weight: z.number().min(0.0).max(1.0).default(0.5).describe("Convex combination weight w in hybrid_score = w * nmi + (1-w) * cosine_similarity. Set closer to 1.0 to emphasize statistical dependency over geometric proximity. Set to 0.5 for balanced scoring. Do not set below 0.1 if the goal is statistical validation \u2014 the p-value computation is always NMI-based regardless of this weight."),
}).strict();

export const EstimatePairwiseNmiMatrixInputSchema = z.object({
  vectors: z.array(z.array(z.number())).describe("Set of embedding vectors for which to compute the pairwise NMI matrix. All vectors must share identical dimensionality. N must be between 2 and 80."),
  vector_ids: z.array(z.string()).describe("Identifiers for each vector in vectors, used to label the matrix rows and columns in the response. Length must exactly match vectors."),
  discretization_bins: z.number().min(2).max(30).default(8).describe("Number of bins for discretizing continuous dimensions before joint-entropy estimation. Shared across all pairs. Same semantics as in rank_embeddings_by_nmi_cosine."),
  bootstrap_iterations: z.number().min(100).max(500).default(150).describe("Bootstrap resampling iterations applied to each pair. Because this endpoint computes N*(N-1)/2 pairs, keep this lower than in single-pair calls to control latency. Minimum 100 required for valid p-values."),
  alpha: z.number().min(0.001).max(0.2).default(0.05).describe("Significance threshold. Pairs with p_value > alpha are flagged as statistically independent in the matrix response."),
}).strict();

export const ScoreCandidatePairSignificanceInputSchema = z.object({
  vector_a: z.array(z.number()).min(8).max(4096).describe("First embedding vector of the pair. Must share dimensionality with vector_b."),
  vector_b: z.array(z.number()).min(8).max(4096).describe("Second embedding vector of the pair. Must share dimensionality with vector_a."),
  discretization_bins: z.number().min(2).max(30).default(8).describe("Bins for joint-entropy discretization of the two vectors' dimensions."),
  bootstrap_iterations: z.number().min(100).max(2000).default(500).describe("Resampling iterations for confidence interval and p-value on the NMI estimate. For a single pair, 500+ iterations are affordable and recommended for tight intervals."),
  nmi_cosine_weight: z.number().min(0.0).max(1.0).default(0.5).describe("Weight w for hybrid_score = w * nmi + (1-w) * cosine. Same semantics as in rank_embeddings_by_nmi_cosine."),
}).strict();

export const DetectEmbeddingDimensionRedundancyInputSchema = z.object({
  sample_vectors: z.array(z.array(z.number())).describe("Representative sample of embedding vectors from the target space. Used to estimate marginal and joint entropy across dimensions. Minimum 50 samples recommended for stable entropy estimates; maximum 1000."),
  redundancy_nmi_threshold: z.number().min(0.1).max(0.99).default(0.7).describe("NMI value above which two dimensions are considered redundant with each other and grouped into the same cluster. NMI=1.0 means perfect dependency; 0.0 means independence. Values between 0.6 and 0.85 are typical for flagging actionable redundancy."),
  discretization_bins: z.number().min(2).max(20).default(8).describe("Bins for discretizing each scalar dimension value across the sample before joint-entropy computation."),
  alpha: z.number().min(0.001).max(0.2).default(0.05).describe("Significance level. Dimension pairs whose NMI p-value > alpha are excluded from redundancy clusters even if their NMI point estimate exceeds redundancy_nmi_threshold."),
}).strict();

export const CalibrateNmiCosineWeightForCorpusInputSchema = z.object({
  triplets: z.array(z.array(z.number())).describe("Flattened calibration triplets. Each triplet is encoded as three consecutive rows: [query_vector, positive_vector, negative_vector]. Total rows must be a multiple of 3. Minimum 15 triplets (45 rows); maximum 300 triplets (900 rows)."),
  weight_search_grid_size: z.number().min(5).max(100).default(20).describe("Number of evenly spaced w values in [0.0, 1.0] to evaluate. A grid of 20 evaluates w in {0.0, 0.05, 0.10, ..., 1.0}. Higher values yield finer-grained optima at linear cost."),
  discretization_bins: z.number().min(2).max(30).default(8).describe("Bins for joint-entropy estimation used during calibration scoring. Should match the value you plan to use in production rank_embeddings_by_nmi_cosine calls."),
  bootstrap_iterations: z.number().min(50).max(500).default(150).describe("Bootstrap iterations per NMI computation during calibration grid search. Kept lower than single-pair calls because it is applied across all triplets and all grid points."),
}).strict();
