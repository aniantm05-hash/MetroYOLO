# Verification record

## Measured model outputs

- 2026-10-13: official YOLOv5s CPU inference smoke test on upstream bus.jpg completed.
- 2026-10-13: all 21 supplied validation images evaluated. Count MAE 1.047619 and
  the other rounded aggregates matched the report; per-image outputs are committed.
- Source archive audited: 81 train / 21 validation; no label syntax errors, corrupt
  images or identical-byte duplicates across splits found.
- A prior reduced-settings CPU training smoke check completed 1 epoch at 64 px using
  the v7.0 profile, produced best.pt and exported post-training metrics. Its metrics
  are not treated as scientific results.

## Software checks

Tests cover metric arithmetic, missing/empty/invalid labels, exact duplicate leakage,
fingerprints, CLI exports, independent pretrained initialization, original notebook
provenance and Streamlit through AppTest. Current local results are in `verification.json`.
Static lint is checked with Ruff. Per-image charts use the committed baseline CSV.
Training and validation curves use the actual 30-epoch log. The comparison chart
uses explicitly labeled report aggregates; it is not a new fine-tuning rerun.
Three report figures were extracted unchanged and checked visually. AppTest verifies
that changing inference thresholds does not fabricate new outputs for historical figures.

The CI matrix is configured for Python 3.10, 3.11 and 3.12. It has not run on GitHub
until publication. Full historical 30/10-epoch GPU training was not rerun here;
Mac execution and production latency are not benchmarked.
