# Frozen A diagnostics

These scripts were used for the 2026-09-27 research reassessment. They do not
train a model or resume any canceled experiment. See
[the report](../../docs/FROZEN_A_DIAGNOSTICS_20260927.md) for results and limitations.

1. `audit_event_label_support.py` checks the original fixed validation cohort,
   selects at most 32 anchors across scenes, and audits seven native keyframes.
   Its JSON `selection` is the input to frozen export.
2. `export_frozen_a_diagnostics.py` verifies the original A source manifest,
   loads the frozen epoch9 best model strictly, and saves unchanged semantic
   predictions plus coordinate metadata. The output directory must be new.
3. `analyze_frozen_a_changes.py` is CPU only. It validates prediction/pose hashes,
   compares static transport on matched domains, saves complete confusion
   matrices, and labels every use of future GT as a diagnostic oracle.
4. `audit_event_sequences.py` is CPU only. It audits seven-frame runs and
   intermediate-label reconstruction from true endpoints. Run `--self-test`
   for its synthetic geometry, mask, event-count and time checks.

Example CPU analysis, with paths supplied by the local installation:

```sh
python tools/diagnostics/analyze_frozen_a_changes.py \
  --manifest /path/to/A_frozen32/manifest.json \
  --occ-root /path/to/occ3d/gts \
  --endpoint-cache /path/to/endpoint_segments_v1_20260926 \
  --output /path/to/new_analysis.json

python tools/diagnostics/audit_event_sequences.py \
  --support-json /path/to/event_label_support.json > new_sequence_audit.json
```

`run_frozen_a_diagnostic.py` and `inspect_frozen_a_diagnostic.py` are deployment
receipts/tools for the specific H200 installation, not portable launch commands.
The runner holds the existing GPU1 lock, requires 132000 MiB free continuously
for 60 seconds, and refuses an existing submission receipt. **Inspect existing
receipts instead of launching it again.** No other process is stopped.

The published analyzer contains the final v3 additive sensitivity check. Earlier
v2 evidence is retained locally and remotely; every original v2 confusion matrix
and metric was verified unchanged in v3. The stricter 3D-neighborhood domain
proved nearly all-free and must not be used to reject the scientific hypothesis.

These are previously used validation scenes. Occupancy labels are reconstructed,
GT endpoint reconstruction is not sensor forecasting, and semantic query indices
do not establish object identity. Prediction archives stay outside Git; the
export manifest records each sample's SHA256 for reproducibility.
