# Future forecasting: diagnosis and two controlled improvement arms

## What the first run establishes

The matched 256-anchor diagnostic improved current-frame semantic mIoU from
30.1578 to 31.9764%, but future 1–3 s mean changed from 24.4398 to 24.4542%.
The best future score occurred at epoch 1, not epoch 10. This supports a plateau
in this run; it does not establish that radar is intrinsically unhelpful.

Follow-up inference interventions on the same 256 anchors strengthen the routing
diagnosis: normal radar gives current/future-mean mIoU 31.9767/24.45424; removing
radar gives 30.1292/24.45256; zeroing its velocity gives 31.7147/24.44859. Thus
the current prediction depends materially on the branch, while future-mean
differences are tiny. These interventions use the already jointly trained
model, and do not replace a separately trained camera-only control.

A deterministic sample of 1,024 training anchors contains 491,824 radar returns;
5.84% have compensated speed over 2 m/s. For those returns, historical motion
displacement has median 0.89 m and 95th percentile 2.97 m. On 64 sampled training
anchors, movable semantic categories account for 7.02% of occupied voxels and
5.66% of camera-visible occupied voxels. These are input/target statistics,
not measured fractions of the optimizer gradient.

Inspection of the executed implementation identifies three mechanisms worth
testing:

1. Each decoder layer injects radar into only `query_feat[:batch_size]`, the
   current horizon. Future queries receive radar indirectly through attention
   over all horizon/query tokens. Nonzero radar gradients do not show that this
   indirect route supplies useful future motion evidence.
2. Historical positions are transformed into the current ego coordinate system,
   but are not advanced by `velocity * age`. Velocity, radial velocity and age
   enter an MLP as features. The network has no explicit radar motion transport
   operation. Ego compensation and object motion compensation are different.
3. Supervision combines geometric point losses and classification across
   horizons and classes. Large static structures and the easier current-frame
   task can provide a useful shortcut. This is a hypothesis, not a measured
   gradient attribution. Class-frequency diagnostics and radar/velocity
   counterfactual evaluation are collected separately.

The original model uses horizon and trajectory conditioning plus attention and
point/classification losses. Its paper reports random-horizon training for
long-term forecasting, but the present matched M0 experiment uses fixed
0/1/2/3-second horizons. Changing that sampling at the same time would introduce
another variable, so both arms retain the existing horizon/data protocol.
See the [SparseWorld-TC paper](https://arxiv.org/html/2511.22039v1).

The [nuScenes SDK](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)
documents compensated radar velocity in metres per second. Compensated returns
are still noisy measurements, not reliable object tracks or future labels.
Constant velocity is therefore an inductive bias with a limited horizon, not
ground-truth object motion.

## Arm A: direct motion-compensated future radar fusion

Config: `configs/sw-radar-forecast-transport.py`.

For a radar return observed `age` seconds before the current timestamp, propose
its position at future horizon `h` in the fixed current-ego frame:

`p(h) = p_observed + v_compensated * (age + h)`.

Transform each future-ego query center into that same frame using the original
future-to-current ego transform, then apply the existing local radar association
and residual feature fusion directly at every horizon. This avoids translating
velocity by ego displacement or mixing coordinates from different timestamps.
No future radar or future object labels enter the forward pass. The future ego
conditioning is the same conditioning already used by the baseline.

Keep all existing learned tensor shapes. Apply a 35 m/s return-validity bound,
age-weighted association (0.5 s scale), and a residual confidence decay
`exp(-h / 3 s)` to reduce reliance on noisy long extrapolations. All six radar
residual output matrices start at zero, preserving the complete official
camera model's initial function.

Expected signature: stronger future sensitivity to radar velocity and improved
future movable-category IoU. Risks: turns/acceleration, poor tangential velocity,
ghost returns and incorrect object associations can make extrapolation harmful.
A negative result should trigger motion uncertainty/correspondence analysis,
not a claim that more epochs alone will fix it.

## Arm B: future- and category-balanced supervision

Config: `configs/sw-radar-forecast-balanced.py`.

Keep the original current-only M0 fusion architecture. Reduce each scene and
each horizon separately, then take the normalized weighted mean with weights
`[0.25, 1.0, 1.25, 1.5]` for `[0, 1, 2, 3]` seconds. This assigns 6.25% of the
horizon weight to reconstruction and 93.75% to forecasting, while retaining
current-frame supervision and avoiding a global loss multiplier.

For future horizons, give movable semantic categories a factor of two in
classification and both directions of point matching. Normalize those factors
to mean one within the relevant target/prediction set. The categories are
bicycle, bus, car, construction vehicle, motorcycle, pedestrian, trailer and
truck. They are **movable categories**, not a claim that every labelled object
is moving. Original rare-class and visibility weighting remains in place.

Expected signature: improved future category accuracy without changing the
information path. Risks: sacrificing current/static accuracy or amplifying
label noise. If B helps but A does not, the optimization budget may be the more
immediate limitation; if A helps but B does not, direct evidence routing is
better supported. These interpretations remain conditional on this one seed.

## Shared protocol and evaluation

- Start each arm from the full official model with fresh AdamW/scaler state.
  Never initialize one arm from the other or from the M0 smoke checkpoints.
- One H200 per arm, run independently in parallel; BS8, accumulation1, seed0,
  ten epochs, the same data/augmentations/cache and learning-rate schedule.
- CPU geometry/objective tests, exact 669-tensor pretrained coverage, four real
  input initialization-parity checks, then a separate 24-step BS8 smoke per arm.
- Same fixed 256-anchor validation before training and after every epoch.
  Keep the best trained epoch even if it underperforms initialization; report
  the negative comparison explicitly. Run all 5,119 anchors at epoch10 and for
  the selected best checkpoint.
- Evaluate the selected checkpoint with normal radar, no radar and zero radar
  velocity on the same 256 anchors. These are inference interventions, not a
  separately trained camera-only baseline.
- Report current/1/2/3 s semantic mIoU, binary IoU and per-class IoU. Record scene
  confusion matrices for paired scene-level bootstrap comparisons. Checkpoint
  selection still uses validation data, and one seed cannot establish robustness.
- Predeclared engineering target: at least +0.3 percentage points in full-set
  future mean mIoU over the previous M0 final checkpoint, with no more than
  0.3-point current-frame regression. Also inspect movable-category changes,
  paired uncertainty, best-vs-final behavior and runtime; this threshold alone
  is not a statistical significance claim.
- Stop an arm on a failed preflight, OOM, nonfinite model/loss, or process error.
  Do not silently reduce batch size, change input resolution, restart from
  scratch or occupy another user's GPU.

Both changes are initially hypotheses. Runtime success, gradients and successful
checkpoint writes will be reported separately from evaluated model benefit.
