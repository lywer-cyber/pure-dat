# Branch policy

- `main`: pure DAT + L1 baseline copied from the local DAT source tree.
- `FFT`: the same DAT source and protocol, with only the registered
  `HologramSpectrumLoss` objective and its matched training configuration
  added. It does not add LKA or replace any DAT spatial mixer.

For a fair ablation, hold the dataset, split assignment, seed, optimizer,
augmentation, iteration budget, validation frequency, and metric definitions
fixed. Compare the FFT branch as a complete training-objective variant; do not
attribute a difference to an architectural LKA change.
