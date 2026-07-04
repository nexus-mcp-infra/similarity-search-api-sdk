# Similarity Search API

The only search API that computes NMI-weighted similarity on raw data — no embeddings, no vector index, no infrastructure.

```bash
pip install similarity-search-client
```

```python
from similarity_search import SimilarityClient
client = SimilarityClient(api_key="sk_live_...")
results = client.search(query="payment fraud detection", corpus=my_texts, top_k=10)
```

That's it. No Pinecone. No embedding model. No index to maintain.

---

## Why this exists

Every vector search provider — Pinecone, Weaviate, Qdrant, pgvector — assumes you've already solved the hard part: converting your raw data into embeddings and building a persistent index. That pipeline costs you:

- **3–5 engineering days** to wire an embedding model, index pipeline, and sync job
- **$0.0001–$0.0004 per token** in embedding API costs before you search anything
- **Cosine similarity only** — which is linear and misses non-linear statistical dependencies between tokens and categories

This API solves a different problem: similarity over raw data, on-the-fly, using information theory.

---

## The math that makes it different

The score for a query `q` against document `d` is:

```
H(q, d) = alpha(C) * cosine(q, d) + (1 - alpha(C)) * NMI(q, d)

where alpha(C) = H_marginal(corpus) / log2(|V|)
```

`alpha(C)` is not a hyperparameter you tune. It's derived from the marginal entropy of your specific corpus on every ingest. When your corpus has high token diversity (high `H_marginal`), cosine gets more weight because the vocabulary is already expressive. When your corpus is semantically dense and repetitive, NMI gets more weight because co-occurrence structure carries the signal that cosine misses.

**What NMI catches that cosine doesn't:**

| Data type | Cosine limitation | NMI advantage |
|---|---|---|
| Short categorical text | Sparse vectors, near-zero dot products | Captures co-occurrence dependency regardless of vector length |
| Discrete time series | No natural embedding | Treats symbol sequences as joint distributions |
| Mixed numeric + text | Requires separate embedding strategies | Unified information-theoretic treatment |
| Repetitive domain corpora | High false-positive similarity | Entropy weighting suppresses uninformative shared tokens |

You cannot replicate this weighting with `sklearn.metrics.normalized_mutual_info_score` or FAISS. Both require you to know `alpha(C)` in advance — which requires knowing the corpus marginal entropy — which is exactly what this API computes and stores per ingest.

---

## Endpoints

| Operation | What it does |
|---|---|
| `POST /corpus` | Ingest raw documents, compute marginal entropy, store corpus ID |
| `POST /search` | Run hybrid H(q,d) search against a corpus ID |
| `GET /corpus/{id}/entropy` | Retrieve `H_marginal` and `alpha` for a corpus |
| `DELETE /corpus/{id}` | Remove corpus and all associated state |

No index files. No embedding storage. No persistent vector dimensions to lock into.

---

## When to use this

**Use this API when:**
- Your collection is under 100k documents and standing up a vector database is disproportionate overhead
- Your data is raw text, categories, or discrete sequences — not pre-embedded vectors
- You need a defensible similarity metric for compliance or explainability (NMI has a closed-form information-theoretic interpretation; cosine does not)
- You're building a one-off script, internal tool, or prototype and cannot justify an embedding pipeline

**Do not use this API when:**
- You have pre-computed embeddings you want to reuse across systems
- Your corpus exceeds 500k documents (use a dedicated ANN index)
- You need sub-10ms latency at p99 (on-the-fly NMI computation has a floor around 40ms per query at corpus size 10k)

---

## Quickstart

```python
from similarity_search import SimilarityClient

client = SimilarityClient(api_key="sk_live_...")

# Step 1: ingest your raw documents (returns corpus_id + entropy metadata)
corpus = client.ingest(
    documents=["credit card fraud", "payment anomaly", "transaction declined", ...],
    label="fraud-detection-v1"
)

# Step 2: search — no embedding step, no index warmup
results = client.search(
    query="suspicious transaction pattern",
    corpus_id=corpus.id,
    top_k=5
)

for r in results:
    print(r.document, r.score, r.nmi_component, r.cosine_component)
```

Output:

```
transaction declined     0.847   nmi=0.61  cosine=0.71
payment anomaly          0.821   nmi=0.58  cosine=0.68
...
```

The `nmi_component` and `cosine_component` fields are exposed in every response. The weighting is not a black box.

---

## Pricing

No monthly seat fee. No storage tier. Pay per operation.

| Operation | Price |
|---|---|
| Corpus ingest | $0.004 per document |
| Search query | $0.002 per query |
| Entropy metadata read | Free |

A corpus of 10,000 documents costs $40 to ingest. Running 1,000 searches against it costs $2.00. Compare that to: embedding pipeline ($1–4 per 10k docs at standard rates) + vector DB hosting ($70+/month minimum on managed services) + engineering time to wire it.

---

## Install

```bash
pip install similarity-search-client
```

Requires Python 3.9+. No system dependencies. The client handles auth, retries with exponential backoff, and corpus ID management locally.

API keys at [dashboard.similarity-search.io](https://dashboard.similarity-search.io).

---

## Self-hosting

The server is `Python 3.12 + FastAPI + Gunicorn`. If you need to run on-premise:

```bash
docker pull similaritysearch/api:latest
docker run -p 8000:8000 -e SECRET_KEY=... similaritysearch/api:latest
```

The corpus entropy state is stored in-process by default. Mount a volume at `/data` for persistence across restarts.

---

## License

MIT for the client SDK. The hybrid scoring engine (server-side) is proprietary.