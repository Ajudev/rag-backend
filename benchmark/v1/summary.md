# Retrieval benchmark v1

- Run date (UTC): 2026-09-23T13:55:48.945801+00:00
- Corpus: aurora_protocol.md, nimbus_lab.txt, willow_garden.md
- Isolated BM25/dense sources: aurora_protocol.md, nimbus_lab.txt, willow_garden.md
- Embedding model: fake-hash
- Reranker model: FakeReranker
- Python: 3.13.5
- Device: cpu
- Backends: hash FakeEmbedder (not semantically meaningful); FakeReranker term-overlap scores

These numbers are measured by `benchmark.v1.run`. Hybrid RRF and rerank are not assumed to win.

## Test split (held-out; not used for tuning)

| config | mode | R@5 | R@10 | MRR@10 | nDCG@10 | lat p50 | lat p95 | rerank p50 | rerank p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dense | 1.0000 | 1.0000 | 0.4583 | 0.5982 | 0.26 | 1.18 | — | — |
| B | bm25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.12 | 0.16 | — | — |
| C | hybrid | 1.0000 | 1.0000 | 0.8750 | 0.9077 | 0.32 | 0.33 | — | — |
| D | hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.33 | 0.34 | 0.03 | 0.03 |

## Validation grid (hybrid only)

Selected by validation MRR@10 (ties: R@10, then R@5): dense_weight=1.0, bm25_weight=2.0, candidates=20.

| dense_w | bm25_w | candidates | MRR@10 | R@10 | R@5 |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 1.0 | 10 | 0.7857 | 1.0000 | 1.0000 |
| 1.0 | 1.0 | 20 | 0.7857 | 1.0000 | 1.0000 |
| 2.0 | 1.0 | 10 | 0.5476 | 1.0000 | 1.0000 |
| 2.0 | 1.0 | 20 | 0.5476 | 1.0000 | 1.0000 |
| 1.0 | 2.0 | 10 | 1.0000 | 1.0000 | 1.0000 |
| 1.0 | 2.0 | 20 | 1.0000 | 1.0000 | 1.0000 |

## Selected hybrid on test (chosen on validation only)

| R@5 | R@10 | MRR@10 | nDCG@10 | lat p50 | lat p95 |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.31 | 0.34 |

## Parameters

- Test top_k: 10 (same for all four configs)
- RRF k: 60
- Default candidate pools: dense 20, BM25 20, rerank 20

## Limitations

- Tiny sample corpus; metrics are not a production quality claim.
- Each run uses a fresh temp BM25 index; leftover `.bm25.json` is not loaded.
- Fake embeddings and fake rerank (when enabled) are for CI speed, not ranking quality.
- Cross-encoder scores are uncalibrated ranking utilities, not probabilities.
