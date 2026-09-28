# Citation verification benchmark

Generated at 2026-09-24T13:26:47.865090+00:00.

Tiny labeled set. Statistical power is limited; do not treat scores as production quality.
Gold labels were written by a human, not by the verifier model.

## Fake run

- Cases: 20
- Invalid-citation cases: 2
- Invalid-citation detection rate: 1.000

Semantic cases were **not** scored for quality on the fake path.
Programmed/fake semantic labels are not a quality metric.

## Live metrics

Live metrics were **not measured** (no `--live` run). No scores are invented.

A/B/C comparison protocol (not executed in this run):
- A: grounded generation without semantic verification
- B: deterministic citation reference validation only
- C: full CitationVerificationService (`semantic_verify`)
Compare false-support rate and macro F1 on the held-out test split only.
