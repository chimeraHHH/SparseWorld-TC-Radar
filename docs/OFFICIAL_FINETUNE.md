# Official checkpoint initialization and joint radar fine-tuning

This experiment loads `sw-tc-small.pth` from the author's Hugging Face repository,
revision `1d6ce8f35a2d963612f52989119a3fd4686cefa5`, SHA256
`871c4da344fbb71f9e0a8067c7f10f6ba55edc6cedd4955878bd9076c6920a6a`.
The 952,230,050-byte checkpoint records epoch70/iteration246190. Only model
weights are used: its optimizer, scaler and iteration counters are not resumed.

`official_init.py` requires exact coverage and shapes for every non-radar tensor,
rejects unexpected/missing/nonfinite state, and verifies exact loaded values.
The six new radar residual output matrices are zeroed, leaving the pretrained
camera function initially unchanged. Other new radar parameters retain their
fresh initialization and learn as the output matrices move away from zero.

`sw-radar-m0-official-ft.py` uses one H200 GPU0, actual/effective BS8, accumulation1,
the complete architecture, input/label protocol and verified local radar cache.
The initial fine-tuning budget is10 epochs, selected as a bounded engineering
trial, not an experimentally established optimum. AdamW peak learning rates are
2e-5 for the pretrained world model, 2e-6 for the image backbone/sampling offsets,
and 2e-4 for the new radar branch. Linear warmup lasts300 updates; epoch-based
cosine scheduling decays to10% of each group's peak LR. Original frozen early
backbone stages and BN policy are retained. All other original trainable layers
and the new radar branch train jointly from the first update.

Validation runs before training and after each epoch on the same256 anchors
selected with seed20260918, with their original full-dataset image history and
token-based future targets. Each report identifies every anchor and separates
0/1/2/3-second semantic/binary IoU and per-class IoU. These subset metrics are
diagnostics, not full benchmark results or a matched fine-tuned camera control.
After epoch10 the full5119-anchor validation runs too. The best subset future
mean mIoU checkpoint is retained when it exceeds the initial model's score.

Validation uses offline image feature extraction, preserves FP16 flags and RNG,
and restores training mode. The original single-GPU online inference path changes
FP16 flags and caches image features; those must not persist across training
updates. `tools/check_official_gpu.py` checks real-input raw-head parity and the
validation state round-trip before launch. Separate radar and pretrained-group
gradient/parameter audits establish that both parts actually update.

The previous backbone-initialized experiment and checkpoints are retained in
their existing directories. This is a new experiment in `m0_official_ft_seed0`,
not continuation from the old experiment's partially completed epoch. Smoke
weights are isolated in `m0_official_smoke_seed0` and never initialize the main run.

Launch after CPU/GPU audits and an exclusive GPU0/ownership check:
`bash tools/run_h200_official.sh configs/sw-radar-m0-official-ft.py`.
