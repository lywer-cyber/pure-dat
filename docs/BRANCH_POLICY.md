# Branch policy

- `main`: pure DAT + L1 baseline copied from the local DAT source tree.
- `FFT`: the same DAT source and protocol, with only the registered
  `HologramSpectrumLoss` objective and its matched training/test configurations
  added. It does not add LKA or replace any DAT spatial mixer.
- `lka`: starts from `FFT` and replaces only the six spatial DSTBs in residual
  groups `[4, 5]` with `LargeKernelSpatialMixer`; see
  `docs/LKA_REPRODUCTION.md`. It keeps the FFT objective and the matched
  training/test protocol fixed.

For a fair ablation, hold the dataset, split assignment, seed, optimizer,
augmentation, iteration budget, validation frequency, and metric definitions
fixed. Compare the FFT branch as a complete training-objective variant; do not
attribute a difference to an architectural LKA change.
