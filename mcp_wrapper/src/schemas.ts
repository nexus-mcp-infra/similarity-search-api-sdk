import { z } from "zod";

export const RankEmbeddingsByNmiCosineInputSchema = z.object({
  query_embedding: z.array(z.number()).min(32).max(4096).describe("Dense float vector representing the query. Must match dimensionality of all candidate_embeddings."),
  candidate_embeddings: z.array(z.array(z.number())).describe("List of dense float vectors to rank against the query. Each inner array must have the same length as query_embedding."),
  domain: z.string().min(4).max(8).describe("Embedding domain that controls the learned NMI/cosine weight mix. Accepted values: 'text', 'image', 'tabular'. Determines alpha_nmi and alpha_cosine from pretrained domain calibration."),
  top_k: z.number().min(1).max(2048).default(10).describe("Number of top-ranked candidates to return. Must be <= len(candidate_embeddings)."),
  return_scores: z.boolean().default(true).describe("If true, includes composite score, nmi_component, and cosine_component for each result. Set false to reduce payload size when only rank order matters."),
}).strict();

export const ComputePairwiseNmiCosineMatrixInputSchema = z.object({
  embeddings: z.array(z.array(z.number())).describe("Set of dense float vectors for which to compute all pairwise composite scores. All vectors must share the same dimensionality."),
  domain: z.string().min(4).max(8).describe("Embedding domain controlling the NMI/cosine weight calibration. Accepted values: 'text', 'image', 'tabular'."),
  normalize_output: z.boolean().default(false).describe("If true, min-max normalizes the composite matrix to [0, 1] per row. Set false to preserve raw composite scores for downstream calibration."),
}).strict();

export const ScoreEmbeddingPairNmiCosineInputSchema = z.object({
  embedding_a: z.array(z.number()).min(32).max(4096).describe("First dense float vector of the pair."),
  embedding_b: z.array(z.number()).min(32).max(4096).describe("Second dense float vector of the pair. Must match dimensionality of embedding_a."),
  domain: z.string().min(4).max(8).describe("Embedding domain controlling calibrated weight mix. Accepted values: 'text', 'image', 'tabular'."),
}).strict();

export const CalibrateDomainNmiCosineWeightsInputSchema = z.object({
  anchor_embeddings: z.array(z.array(z.number())).describe("Query-side embeddings for each relevance pair. Index-aligned with positive_embeddings and negative_embeddings."),
  positive_embeddings: z.array(z.array(z.number())).describe("Embeddings of items labeled as relevant/similar to the corresponding anchor. Same length as anchor_embeddings."),
  negative_embeddings: z.array(z.array(z.number())).describe("Embeddings of items labeled as non-relevant to the corresponding anchor. Same length as anchor_embeddings."),
  domain_label: z.string().min(3).max(64).describe("Identifier for the custom domain profile to be created. Used to reference this calibration in subsequent ranking calls via the domain parameter."),
}).strict();

export const ExplainNmiCosineRankDivergenceInputSchema = z.object({
  query_embedding: z.array(z.number()).min(32).max(4096).describe("Dense float vector representing the query."),
  candidate_embeddings: z.array(z.array(z.number())).describe("Candidate dense float vectors to compare under both ranking methods."),
  domain: z.string().min(3).max(64).describe("Embedding domain for composite weight calibration. Accepted values: 'text', 'image', 'tabular', or a custom domain_label from calibrate_domain_nmi_cosine_weights."),
  top_k: z.number().min(2).max(256).default(10).describe("Number of candidates to include in the divergence report, taken from the top of the composite ranking."),
}).strict();
