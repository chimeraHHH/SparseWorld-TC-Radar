# Offline causal radar cache

`tools/precompute_causal_radar.py` moves deterministic radar PCD parsing,
causal sweep accumulation and point/velocity/LOS transformations out of the
training dataloader. It calls the same online loader. It does not change the
network, augmentation, labels, batch size, learning rate or the radar protocol.

The M0 protocol remains five radar sensors, at most five sweeps per sensor,
LiDAR-timestamp ego coordinates, age in [0, 0.5] seconds, xy limit 44 m and
4096 points maximum with stable youngest-first truncation. Each point has ten
float32 features. Learned query sampling, encoding and gating remain online.

## Build and validate

Run from the repository root in the existing training environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/precompute_causal_radar.py \
  --config configs/sw-radar-m0-single-bs8.py \
  --output /home/huayiming/Workspace/SparseWorld-cache/radar_m0_v2_20260918 \
  --workers 4
```

The builder creates a `.building` directory under an exclusive lock, verifies
every serialized array, exercises the production reader across the full
manifest, and independently recomputes 128 samples. Only then does it publish
the final directory with an atomic rename. It refuses to overwrite a published
cache. A partial build can be rerun with the same protocol; files are regenerated.
For a fully generated but unpublished staging cache, `--verify-existing` checks
its protocol, annotation hash and full coverage, then repeats every read and
independent recomputation before publishing. It does not skip validation.
Keep the code checkout unchanged while a builder or trainer is running; stage
subsequent changes in another checkout.

`protocol.json` fingerprints the parameters, fields, sensor channels, nuScenes
radar filters, online algorithm/PCD parser source and five relevant nuScenes
metadata tables. `manifest.json` records per-token reference time, count, file
hash and array hash. `COMPLETE.json` records coverage, annotation hash and
validation evidence. Raw PCD bytes are not rehashed on every training read;
the cache represents the source dataset snapshot used during generation.
Regenerate if the raw dataset changes even if its metadata is unchanged.

The reader rejects unsealed caches, changed protocol/source/metadata, missing
samples, timestamp mismatch, malformed/nonfinite/out-of-range points and
checksum mismatch. There is no silent fallback. Online and cached paths emit
the same `radar_points` and `radar_reference_timestamp_us` keys. The historical
v1 probe cache is intentionally not accepted by the production v2 reader.

## Single-H200 continuation

`configs/sw-radar-m0-single-bs8-cache.py` changes only the training radar source,
output directory and resume anchor. It retains GPU 0, batch size 8, accumulation
1, 12 workers and the original optimizer/model/input settings. Validation/test
pipelines still use the equivalent online loader; this cache covers the training
split only. The initial cutover resumes from completed epoch 4, with the same
2992 steps per epoch. Model, AdamW state and FP16 scaler are restored.

Epoch checkpoints in the upstream runner do not capture all sampler/worker RNG
states, so a restart is not a bitwise continuation of the sample/augmentation
sequence. No incomplete epoch is intentionally discarded by this cutover.

Keep the preceding online checkout and work directory for rollback. For a later
restart, inspect the latest **complete** checkpoint, set a fresh work directory,
and use the corresponding epoch-boundary resume parameters. Do not relaunch
an existing output directory blindly. `tools/run_h200_single.sh` verifies code
hashes, checkpoint state, batch size, GPU ownership and the shared training lock.

Report radar-only CPU timing separately from complete input-pipeline timing and
sustained GPU training throughput. The earlier 31x isolated radar result is not
an end-to-end acceleration claim.
