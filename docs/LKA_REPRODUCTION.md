# DAT+FFT+LKA reproduction

The `lka` branch starts from the `FFT` branch. It keeps the DAT backbone,
dataset protocol, loss weights, schedule, validation metrics, and split fixed.
The architectural ablation is limited to the spatial attention path:

- `spatial_mixer: large_kernel` replaces the DSTB in residual groups `[4, 5]`
  (zero-based).
- Each of those groups has three even-indexed spatial blocks, so exactly six
  `LargeKernelSpatialMixer` modules are created.
- All 18 DCTBs and all spatial DSTBs in groups `[0, 1, 2, 3]` remain unchanged.
- The LKA mixer uses a depthwise `5x5` local convolution, a depthwise `7x7`
  convolution with dilation `3`, and two pointwise projections. Its effective
  receptive field is `23x23`.

FFT is only a training objective (`HologramSpectrumLoss`, weight `0.01`); it
is not inserted into the network and is not used to alter validation metrics.

## Smoke test

From the repository root, run:

```bash
PYTHONPATH=. python scripts/smoke_test_lka.py
```

The check must report six replaced DSTBs, 18 preserved DCTBs, a valid `1x8x8`
to `1x32x32` forward pass, and finite backward gradients.

## Training and test

```bash
PYTHONPATH=. python basicsr/train.py \
  -opt options/Train/train_DAT_FFT_LKA_holo_x4_matched_25500.yml \
  --launcher none

PYTHONPATH=. python basicsr/test.py \
  -opt options/Test/test_DAT_FFT_LKA_holo_x4.yml
```

The dataset is intentionally external to Git and must be present at
`/root/autodl-tmp/real_hologram_x4`. The protocol is 1,700 train / 150
validation / 150 test, seed 2026, batch size 2, and 25,500 optimizer
iterations. The known manifest contains only four unique HR contents repeated
across files, so results on this dataset must be reported as exploratory and
not as an independent-generalization claim.
