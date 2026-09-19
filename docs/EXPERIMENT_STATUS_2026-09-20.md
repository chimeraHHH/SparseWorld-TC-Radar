# Official-model radar joint fine-tuning: completed run

Verified on 2026-09-20 at 00:15 Asia/Shanghai. Training source revision:
`35081f9863c6ed9fb4dde7c235cc8a67a1482548`.

## Configuration and completion

- Complete official `sw-tc-small.pth` model initialization; fresh optimizer and
  scaler. See [initialization details](OFFICIAL_FINETUNE.md).
- New radar branch and originally trainable camera/world-model parameters jointly
  updated; original early-backbone freezing and batch-normalization policy kept.
- One H200, batch size 8, accumulation 1, 10 epochs and 29,920 iterations.
- Verified causal radar cache covers all 23,930 training anchors.
- Initial and every-epoch validation on the same 256 anchors; final validation on
  all 5,119 anchors. All subset indices and tokens were checked for equality.
- Training completed on September 19 at 22:30:58; full validation completed at
  23:13:28. Times are Asia/Shanghai. Total runtime from formal log start was
  approximately 26 hours 17 minutes; full validation took 41.25 minutes.
- Training exited and released GPU0. The last 1,000 logged training steps averaged
  3.0467 seconds per step and 2.626 samples per second.

## Matched diagnostic subset

Semantic mIoU is in percent. Changes are percentage points, measured on the same
256 anchors, comparing official initialization against epoch 10.

| Horizon | Initial | Epoch 10 | Change |
|---|---:|---:|---:|
| Current | 30.1578 | 31.9764 | +1.8186 |
| 1 second | 26.9001 | 26.8478 | -0.0523 |
| 2 seconds | 24.4200 | 24.4299 | +0.0100 |
| 3 seconds | 21.9994 | 22.0848 | +0.0854 |
| Mean over 1–3 seconds | 24.4398 | 24.4542 | +0.0144 |

The best subset future mean was at **epoch 1**, 24.5260%, an improvement of
0.0862 percentage points over initialization. `best_future.pth` retains that
epoch; `latest.pth` points to `epoch_10.pth`.

## Full validation of epoch 10

These results use all 5,119 anchors. They must not be subtracted from the initial
256-anchor results to estimate improvement.

| Horizon | Semantic mIoU (%) | Binary IoU (%) |
|---|---:|---:|
| Current | 32.7914 | 52.7268 |
| 1 second | 27.6575 | 50.2104 |
| 2 seconds | 25.2713 | 48.8463 |
| 3 seconds | 23.1480 | 47.1590 |

The full-set mean semantic mIoU over 1–3 seconds is **25.3589%**.

## Numerical audit and interpretation

Five logged windows in epochs 6–8 contained infinite gradient norms, while
losses stayed finite. The final optimizer recorded 29,915 updates across 29,920
iterations, consistent with five skipped mixed-precision updates. Both the final
and best checkpoints were reloaded on CPU and all 771 model tensors were finite.
No crash or out-of-memory traceback was found, and final validation completed.

The current-frame diagnostic improves, while future prediction is essentially
unchanged. Joint parameter updates prove the two branches trained; they do not
establish an independent radar benefit. A matched camera-only fine-tuning
control has not been run. Full-set evaluation of the official initialization and
epoch-1 checkpoint, followed by a matched camera-only control, is needed for
stronger conclusions. No additional training was launched during this audit.

Checkpoints and raw run receipts remain in the original experiment storage and
are intentionally not distributed with this source repository.
