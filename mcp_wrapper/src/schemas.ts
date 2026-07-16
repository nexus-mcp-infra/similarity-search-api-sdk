import { z } from "zod";

// --- NEXUS PATCH mcp_tool_grounding_similarity_search ---
// Only RankEmbeddingsByNmiCosineInputSchema survives grounding against the
// real core service; the other 4 schemas described endpoints that were
// never implemented and have been removed. Params below match the real
// /similarity/search request body (query, corpus, top_k, nmi_bins,
// alpha_override) instead of the fictional bootstrap/significance params.

export const RankEmbeddingsByNmiCosineInputSchema = z.object({
  query_vector: z.array(z.number()).min(2).max(4096).describe("Dense embedding vector to rank the corpus against. Must share dimensionality with every vector in corpus_vectors."),
  corpus_vectors: z.array(z.array(z.number())).min(1).describe("Candidate embedding vectors to rank. Each inner array must share query_vector's dimensionality."),
  corpus_ids: z.array(z.string()).describe("Identifiers for each vector in corpus_vectors, returned in the ranking response. Length must exactly match corpus_vectors."),
  top_k: z.number().min(1).max(1000).default(10).describe("Number of top-ranked results to return, ordered by descending composite score. Must be <= len(corpus_vectors)."),
  nmi_bins: z.number().min(3).max(50).default(10).describe("Number of bins used to discretize embedding dimensions before computing Normalized Mutual Information. Higher values increase resolution but also estimation variance."),
  alpha_override: z.number().min(0.0).max(1.0).optional().describe("Pin the cosine/NMI blend weight manually in [0,1]. If omitted, an entropy-calibrated alpha is computed automatically from the corpus."),
}).strict();
