import { z } from "zod";

export const RankHybridSimilarityInputSchema = z.object({
  query: z.string().min(2).max(102400).describe("JSON-encoded object representing the query record. Keys are feature names; values are strings (categorical) or numbers (continuous). Must share at least one key with corpus records."),
  corpus: z.string().min(2).max(10485760).describe("JSON-encoded array of objects. Each object is a candidate record with the same schema as query. Minimum 2 records, maximum 50000 records per call."),
  top_k: z.number().min(1).max(1000).default(10).describe("Number of top-ranked results to return. Must be between 1 and the corpus size. Defaults to 10."),
  score_breakdown: z.boolean().default(false).describe("If true, each result includes per-component scores: nmi_score, cosine_score, nmi_weight, cosine_weight, and per-feature NMI contributions. Adds latency proportional to feature count. Defaults to false."),
}).strict();

export const ComputePairwiseHybridMatrixInputSchema = z.object({
  records: z.string().min(2).max(5242880).describe("JSON-encoded array of objects. Each object is a record with string (categorical) or numeric (continuous) values. All records must share at least 2 common keys. Maximum 2000 records."),
  include_diagonal: z.boolean().default(true).describe("If true, diagonal entries (self-similarity = 1.0) are included in the output matrix. Set to false when passing the matrix directly to clustering algorithms that assume zero diagonal. Defaults to true."),
}).strict();

export const ExplainFeatureWeightCalibrationInputSchema = z.object({
  sample_records: z.string().min(2).max(2097152).describe("JSON-encoded array of at least 10 representative records from your dataset. Used to estimate marginal entropies and variance. Maximum 5000 records."),
}).strict();

export const FilterByHybridThresholdInputSchema = z.object({
  query: z.string().min(2).max(102400).describe("JSON-encoded object representing the query record. Keys are feature names; values are strings (categorical) or numbers (continuous)."),
  corpus: z.string().min(2).max(10485760).describe("JSON-encoded array of candidate records. Maximum 50000 records per call."),
  min_hybrid_score: z.number().min(0.0).max(1.0).describe("Minimum hybrid similarity score [0.0, 1.0] a record must exceed to be included in the result. Values below 0.05 risk returning the full corpus."),
  score_breakdown: z.boolean().default(false).describe("If true, each returned record includes nmi_score, cosine_score, and component weights alongside the hybrid_score. Defaults to false."),
}).strict();

export const DetectFeatureTypeSchemaInputSchema = z.object({
  sample_records: z.string().min(2).max(2097152).describe("JSON-encoded array of at least 5 representative records. More records yield more reliable type detection for low-frequency categorical values. Maximum 5000 records."),
  override_types: z.string().min(2).max(4096).optional().describe("Optional JSON-encoded object mapping feature names to forced types: 'categorical' or 'continuous'. Use to correct misclassifications on ambiguous integer-coded fields. Example: {\"zip_code\": \"categorical\"}. Pass null to skip."),
}).strict();
