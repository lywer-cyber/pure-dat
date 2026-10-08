"""Dataset adapter for manifest-validated single-channel hologram SR pairs."""

import json
import os
import random

import numpy as np
import torch
from torch.utils import data

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class HologramIntensityDataset(data.Dataset):
    """Read normalized ``.npy`` LR/HR intensity pairs from a prepared dataset.

    The dataset root must contain ``manifest.json`` and ``train``, ``val`` and
    ``test`` splits. Each split has ``input``, a configured target directory,
    and ``metadata`` subdirectories with matching sample stems.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.root = os.path.realpath(os.path.abspath(os.path.expanduser(opt['dataroot'])))
        self.phase = str(opt['phase']).lower()
        self.scale = int(opt['scale'])
        self.patch_size = opt.get('gt_size')
        self.patch_size = int(self.patch_size) if self.patch_size else None
        self.use_hflip = bool(opt.get('use_hflip', False))
        self.use_rot = bool(opt.get('use_rot', False))
        self.target_dir = str(opt.get('target_dir', 'hologram_clean'))
        self.expected_protocol = opt.get('dataset_protocol')
        self.samples = self._find_samples()

    def _find_samples(self):
        manifest_path = os.path.join(self.root, 'manifest.json')
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f'Hologram dataset manifest not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)

        if manifest.get('schema_version') != 'holo-intensity-sr-v2':
            raise RuntimeError('Unsupported hologram dataset schema; expected holo-intensity-sr-v2.')
        protocol = str(manifest.get('protocol', ''))
        if self.expected_protocol is not None and protocol != str(self.expected_protocol):
            raise RuntimeError(
                f'Dataset protocol mismatch: expected {self.expected_protocol}, found {protocol}.'
            )
        task_names = {
            'clean_pair': 'off_axis_hologram_intensity_sr_clean_pair',
            'lab_like': 'off_axis_hologram_intensity_sr_lab_like',
            'complex': 'off_axis_hologram_intensity_restoration_sr',
        }
        task = manifest.get('task', {})
        if task.get('name') != task_names.get(protocol):
            raise RuntimeError('Manifest task is not a supported hologram-intensity SR task.')
        if int(task.get('scale', -1)) != self.scale:
            raise RuntimeError('Dataset scale does not match the configured DAT scale.')
        if self.target_dir not in manifest.get('layout', {}):
            raise RuntimeError(f'Target directory {self.target_dir!r} is not declared in the manifest.')

        split_root = os.path.join(self.root, self.phase)
        input_dir = os.path.join(split_root, 'input')
        target_dir = os.path.join(split_root, self.target_dir)
        metadata_dir = os.path.join(split_root, 'metadata')
        if not all(os.path.isdir(path) for path in (input_dir, target_dir, metadata_dir)):
            raise FileNotFoundError(f'Incomplete {self.phase} split under {split_root}.')

        def stems(directory, suffix):
            return {
                os.path.splitext(name)[0]
                for name in os.listdir(directory)
                if name.lower().endswith(suffix)
            }

        input_stems = stems(input_dir, '.npy')
        target_stems = stems(target_dir, '.npy')
        metadata_stems = stems(metadata_dir, '.json')
        if not input_stems:
            raise RuntimeError(f'No hologram samples found in {input_dir}.')
        if input_stems != target_stems or input_stems != metadata_stems:
            raise RuntimeError(f'{self.phase} input, target, and metadata stems do not match.')
        expected_count = manifest.get('splits', {}).get(self.phase)
        if expected_count is None or int(expected_count) != len(input_stems):
            raise RuntimeError(f'{self.phase} split count does not match the manifest.')

        return [
            (
                stem,
                os.path.join(input_dir, stem + '.npy'),
                os.path.join(target_dir, stem + '.npy'),
            )
            for stem in sorted(input_stems)
        ]

    @staticmethod
    def _read(path):
        array = np.load(path).astype(np.float32, copy=False)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f'{path} must contain a finite [1,H,W] or [H,W] array.')
        if float(array.min()) < -1e-6 or float(array.max()) > 1.000001:
            raise ValueError(f'{path} is outside the stored [0,1] range.')
        return np.ascontiguousarray(np.clip(array, 0.0, 1.0)[..., None])

    def _crop(self, lq, gt):
        if self.phase != 'train' or self.patch_size is None:
            return lq, gt
        if self.patch_size % self.scale != 0:
            raise ValueError('gt_size must be divisible by scale.')
        lr_size = self.patch_size // self.scale
        if lq.shape[0] < lr_size or lq.shape[1] < lr_size:
            raise ValueError('LR image is smaller than gt_size/scale.')

        top = random.randrange(0, lq.shape[0] - lr_size + 1)
        left = random.randrange(0, lq.shape[1] - lr_size + 1)
        lq = lq[top:top + lr_size, left:left + lr_size]
        gt_top, gt_left = top * self.scale, left * self.scale
        gt = gt[gt_top:gt_top + self.patch_size, gt_left:gt_left + self.patch_size]
        if self.use_hflip and random.random() < 0.5:
            lq, gt = lq[:, ::-1], gt[:, ::-1]
        if self.use_rot and random.random() < 0.5:
            lq, gt = lq[::-1], gt[::-1]
        if self.use_rot and random.random() < 0.5:
            lq, gt = lq.transpose(1, 0, 2), gt.transpose(1, 0, 2)
        return np.ascontiguousarray(lq), np.ascontiguousarray(gt)

    def __getitem__(self, index):
        _, lq_path, gt_path = self.samples[index]
        lq, gt = self._read(lq_path), self._read(gt_path)
        expected_gt_shape = (lq.shape[0] * self.scale, lq.shape[1] * self.scale)
        if gt.shape[:2] != expected_gt_shape:
            raise ValueError(f'HR/LR dimensions in {lq_path} and {gt_path} do not match x{self.scale}.')
        lq, gt = self._crop(lq, gt)
        return {
            'lq': torch.from_numpy(lq.transpose(2, 0, 1)).float(),
            'gt': torch.from_numpy(gt.transpose(2, 0, 1)).float(),
            'lq_path': lq_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.samples)
