# A + velocity-consistency association + temporal reliability

User-authorized September 22, 2026. This is one combined pilot compared with the
completed A transport run, not a reproduction of Sparse4D-Radar or a completed
component ablation. Existing A, belief and camera source snapshots are unchanged;
the canceled B arm must never restart.

## Method

Retain A's historical motion compensation and constant-velocity forecast transport,
deformable query samples, spatial neighborhoods, age weights and residual fusion.
Two optional flags default to false, preserving the old execution path.

**Velocity-consistency soft association.** Within each query sample's existing
eight-return neighborhood, project each peer's compensated planar velocity onto
the tested return's cached LOS. Compare this with that return's cached radial
proxy. Exclude the diagonal, average Gaussian agreement using the peers' existing
spatial/age weights, then multiply A's association weights by `0.25 + 0.75*agreement`.
Single returns fall back to spatial association. The learned Gaussian width is
bounded to 1--10 m/s, initially 3 m/s. It receives the unchanged occupancy loss;
no new labels or auxiliary loss are introduced.

This is a **neighborhood consensus adaptation**, not the paper's supervised
anchor-velocity/query-to-radar VCS. It avoids inventing an unsupervised object
velocity head or comparing a return against a mean containing itself. Mixed-object
neighborhoods can still penalize a valid minority motion; the nonzero floor only
mitigates this. Multiple correlated sweeps are not independent measurements.
The cache contains compensated velocity and derived radial proxies, not retained
raw Doppler, sensor identities or complete quality fields. No double ego-speed
subtraction or claim of full planar object-velocity recovery is made.

**Temporal reliability.** Pool six quantities for each query: mean return age,
neighbor support fraction, velocity agreement, velocity spread, squared spatial
distance and forecast horizon. A 6->32->1 MLP predicts `delta`; use
`exp(-t * exp(tanh(delta)) / 3)` as the residual's temporal factor. Its final layer
starts at zero, exactly recovering A's `exp(-t/3)` time factor. Rates stay positive
and bounded, and the current-time factor remains one. This learns how much future
radar evidence to retain; it is not a calibrated probability or a direct copy of
the paper's feature-based adaptive modality gate. There are 1,548 added parameters
(258 per decoder layer), including six learned velocity widths.

All original A module parameters and random generator progression are preserved
when constructing the extra gate. Geometry/consistency computations use FP32.
Official initialization zeros all six radar residual outputs and exactly loads
all 669 camera tensors. Both new and original parameters train jointly.

## Protocol and automatic queue

- `configs/sw-radar-forecast-transport-reliable.py`: fresh complete official model,
  fresh AdamW, single H200 GPU1, BS8/accumulation1, seed0, ten epochs/29,920 steps.
  Same cache, data, losses, augmentations, learning rates and schedule as A.
- Independent immutable committed deployment. CPU regression/config/initialization
  audits precede submission; then GPU tests, four real-input output-equivalence
  checks, 24-step BS8 smoke and finite model/optimizer checks. All six smoke audit
  windows must show real gradients and changes in velocity scale, reliability
  gate, radar readout and pretrained backbone/neck/head. Smoke weights are never
  formal initialization.
- Wait for A and belief queues to complete, hold the existing GPU1 lock, and require
  at least 132,000 MiB free continuously for 60 seconds before every GPU stage.
  Resource waits are not failures. Never stop another account's tasks. No automatic
  retry or hyperparameter fallback on failure.
- Fixed256 initialization/epoch validation, selected-best and final full5119 with
  scene confusions; best256 normal/drop/zero/shuffle velocity interventions.
  Automatically compare best-to-best and final-to-final against A using 2,000
  paired scene-bootstrap replicates, reporting horizons, classes and future mean.
  Existing matched camera/belief/reference results remain part of the final report.
- Selection256 is inside full5119, one seed, combined changes and no trained
  component ablations: gains cannot separately establish either module's effect.
  Runtime, actual optimizer steps and AMP events must accompany quality results.

Campaign receipts: `analysis/transport_reliable_campaign_20260922`. Controller:
`tools/run_belief_experiment.py --arm transport-reliable --gpu 1`; exclusive CPU
preflight/submission: `tools/launch_transport_reliable_campaign.py`. Neither entry
point should be invoked again when an existing receipt/process already exists.

## Inspiration and limits

- [Blog reviewed in Safari](https://blog.51cto.com/u_16099347/14838286).
- [Sparse4D-Radar primary paper](https://arxiv.org/html/2607.04098v1), sections III-C/D:
  velocity similarity and adaptive modality gating motivate this adaptation.
- [Author repository](https://github.com/Aiuan/Sparse4D-Radar): the inspected version
  did not release the complete implementation. Its OmniHD detection setting and
  velocity supervision differ from this nuScenes occupancy forecast setting.

Tests, successful launch and trainability establish implementation behavior, not
prediction improvement. The completed A best future mIoU 26.2881632681% is the
primary full-validation comparator; this new experiment has no result yet.

## Verified admission, September 22, 2026

Immutable source **2fc2feaf5edbf0e1bc839b6b34c694be7e85af83** passed
[the complete preflight](evidence/transport_reliable_preflight_20260922.json):
57 CPU regressions, three subsequent CUDA contracts, exact loading of669 official
tensors, fresh optimizer and four real anchors with13 output tensors each exactly
equal to the disabled-radar camera model (maximum difference0). The BS8 smoke
completed24 optimizer updates with801 finite model tensors and a finite optimizer;
all six audit windows had positive finite gradients and parameter changes in
velocity scale, reliability gate, radar readout and pretrained backbone/neck/head.
The smoke checkpoint's iteration field is23 because MMCV saves the zero-based
iteration before incrementing it; AdamW's actual update counter is24.

At09:10:43 Beijing controller1297329 had completed all pretraining checks and was
waiting for its final60-second capacity window before the fresh formal run. The
trial used about98,557MiB device memory. Short smoke steps around3.1 seconds are
not a sustained formal-training speed estimate. Training results remain pending.

An earlier CPU-only audit failure at5361783 was caused by comparing configs after
MMDetection inserted train_cfg/test_cfg into the candidate dictionary. A before/
after diagnostic verified those were the only differences; the comparison was
moved before model construction. Failed receipts were preserved, and the revised
immutable source repeated all checks. No GPU training used that failed revision,
and no scientific hyperparameter changed.
