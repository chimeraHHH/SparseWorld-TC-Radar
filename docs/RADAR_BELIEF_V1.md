# Radar belief v1: reference-time motion information for future occupancy

## Problem and scope

M0 improved current occupancy but barely changed future predictions under radar-removal and velocity-zeroing interventions. A routes deterministic transported radar directly into every horizon and has promising fixed-subset results; it is still running and is not modified. B was stopped by the user and will not resume.

The new version implements the referenced September 20 research discussion's structured motion-posterior hypothesis. This is a pilot, not a claim that Bayesian updates or radar fusion are novel. It does not implement calibrated occupancy probabilities, explicit object tracking, sensor-aware association, or the entire proposed experiment matrix.

## Computation

1. First-layer current-time camera sampling/mixing produces 600 spatial query features before radar and cross-horizon attention. A small MLP predicts planar velocity mean and a positive-definite 2x2 covariance (bounded standard deviations and correlation).
2. Radar returns are associated at their actual historical timestamp: compare their positions to `anchor - prior_velocity * age`. Exact duplicate rows are removed. Nearest neighbors receive spatial, detached robust-residual, and bounded observation-noise weights. Per-anchor total information weight is capped at one effective observation; this is a conservative correlation heuristic, not an independence proof.
3. Each return contributes `w/variance * n n^T` to precision and `w/variance * n * radial` to the information vector. A single LOS does not directly measure tangential velocity. Prior cross-covariance can nevertheless couple the axes. No evidence gives the exact prior.
4. Construct this reference posterior **once per model forward**. Every decoder layer and horizon reads the same state. Future queries use spatial lookup of propagated anchors, without assuming matching query indices identify the same object. Spread is `P_position + t^2 P_velocity + t^4 P_acceleration/4`; kernel density and information reduction downweight diffuse evidence. This is a feature-readout covariance, not occupancy calibration.
5. Zero-initialized residual outputs preserve official camera outputs exactly at initialization. The original model and new branch jointly train with fresh AdamW state.
6. A separate held-out radar objective splits spatial/time groups using only positions and age. Its posterior excludes all held-out returns, and held-out radial values enter only the NLL target. Full forecasting uses all valid radar. The auxiliary loss (weight 0.05) is averaged once across scenes, outside the existing samplewise occupancy reduction. Held-out groups are an approximate return split, not independent objects or sensors.

All matrix algebra is FP32 under AMP; analytic 2x2 inverses avoid repeatedly launching a general matrix solver. The prior, observation-noise model, point/state encoders and residual readouts are audited separately for actual gradients and parameter updates during smoke training.

## Input limits and causal boundary

Use exactly the existing causal 10-column radar cache: xyz, compensated vx/vy, RCS, age, compensated radial proxy and planar LOS. The radial value is derived from the SDK's compensated vector velocity and rotated LOS, **not an independently retained raw Doppler measurement**. Planar LOS is not renormalized. The discarded vertical component, quality fields and sensor identities remain limitations. No new sensor-quality fields are invented; no ego-velocity is subtracted twice. Targets/future labels enter losses only, not belief construction. Official SDK defaults filter returns upstream.

The first pilot holds cache, train/validation samples and data processing fixed against A/M0. A later sensor-origin/quality-aware cache needs a separate protocol and retraining; it must not silently replace this run's data.

## Matched experiment and scheduling

- New belief: `configs/sw-radar-forecast-belief.py`, GPU1 single H200.
- Camera control: `configs/sw-radar-forecast-camera.py`, GPU0 only after A's entire existing pipeline finishes and its missing old-M0 reference evaluation is completed. It derives from the actual official-ft configuration, has `radar_cfg=None`, and removes radar-specific audit hooks. Radar loading is retained for identical data ordering/I/O, but is not consumed by the model.
- Both: official checkpoint SHA256 `871c4da344fbb71f9e0a8067c7f10f6ba55edc6cedd4955878bd9076c6920a6a`; all 669 original tensors exact; BS8, accumulation1, seed0, 23930 train anchors, 2992 steps/epoch, 10 epochs. Shared parameter LR/budget/optimizer/augmentation are identical. Radar LR2e-4, world2e-5, backbone/sampling2e-6.
- Each pipeline: CPU contracts/regressions; real-input zero-residual parity at four anchors; BS8 24-step smoke with finite model/optimizer and positive component updates; **fresh official initialization** for formal training. No smoke weights enter formal training.
- Each GPU stage requires >=132000 MiB free continuously for 60s and the existing per-GPU lock. No other user processes are stopped. Immutable snapshots are separate from active A.
- Validation: same fixed256 seed20260918 at initialization and each epoch; trained-best and final full5119 with scene confusion matrices. Belief best gets normal/drop/zero_velocity/shuffle_velocity fixed256 interventions. Selected and final results must both be reported.
- Primary outcome: full-validation future1/2/3s mean mIoU versus matched camera, alongside current and per-horizon/per-class metrics and paired scene bootstrap. Compare A as a different method, not a clean component ablation. The fixed256 selection set lies within5119: report selection bias and single-seed limits. No performance gain is established by smoke/parity.

The minimal pilot does not yet identify geometry and feature velocity paths separately, establish calibrated uncertainties, or test actual-motion/radial-tangential strata. Those claims require additional interventions, targets, controls and seeds. An equal-capacity unconstrained velocity model and isotropic-covariance model remain future ablations, not completed evidence.

## Sources

- Research design input: ChatGPT conversation `6aafc368-dc94-83ee-ad4c-6701758dd6ba`, “修改训练配置”; suggestions were reviewed as research proposals, not accepted as verified literature claims.
- [nuScenes SDK radar schema and filtering](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py).
- [Long et al., Full-Velocity Radar Returns by Radar-Camera Fusion, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Long_Full-Velocity_Radar_Returns_by_Radar-Camera_Fusion_ICCV_2021_paper.html).
