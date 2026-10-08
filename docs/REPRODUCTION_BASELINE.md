# Pure DAT reproduction baseline

This repository snapshot is the single-channel hologram x4 adaptation of the
DAT network copied from the local `对比模型/DAT` source tree. The `main` branch
is the pure DAT baseline: it uses the DAT spatial mixer, L1 loss only, and no
FFT loss or LKA module.

## Fixed protocol

- Input/output: one-channel intensity, x4 super-resolution (`1x64x64 ->
  1x256x256`).
- Dataset root: `/root/autodl-tmp/real_hologram_x4` (kept outside Git).
- Splits: 1,700 train / 150 validation / 150 test; split seed 2026.
- Training: 25,500 optimizer iterations, seed 2026, Adam `1e-4`, betas
  `[0.9, 0.99]`, MultiStepLR milestones `[17000, 21250, 23800]`, EMA 0.999,
  AMP enabled, batch size 2, GT patch 256.
- Validation: every 850 iterations, crop border 4, PSNR/SSIM/MAE/RMSE plus
  the hologram spectral metrics.
- Initialization: `pretrain_network_g: ~` and `resume_state: ~`.

The manifest and per-sample metadata must be checked before a run. The
reference dataset records 2,000 files but only four unique HR contents; file
names are split-disjoint while content is not. Results from that dataset must
therefore be reported as exploratory and not as an independent-generalization
test.

## Pure DAT command

```bash
cd /path/to/pure-dat
PYTHONPATH=. python basicsr/train.py \
  -opt options/Train/train_DAT_holo_x4_matched_25500.yml \
  --launcher none
```

The startup log must contain `Network [DAT] is created`, only `Loss [L1Loss] is
created`, and `Start training from epoch: 0, iter: 0`.

## Scope

This branch is intentionally a network/loss baseline. It does not contain
LKA code and does not load checkpoints or datasets from Git. Keep checkpoints,
logs, manifests, and environment captures in the experiment archive rather
than committing them here.
