"""Held-out SID real pairs, with the same preprocessing as SIDEvalDataset."""
from pathlib import Path
import pickle
import random

import numpy as np
import rawpy

from datasets.real_dataset import SIDEvalDataset


class SIDRealValidationDataset(SIDEvalDataset):
    def __init__(self, pair_list, sid_long_dir, resource_dir, ratios=(100, 250, 300), max_items=None, seed=1):
        self.wl, self.bl = 16383, 512
        self.clip_low, self.clip_high = float('-inf'), 1
        resource_dir = Path(resource_dir)
        with (resource_dir / 'darkshading_BLE.pkl').open('rb') as handle:
            self.pmn_ble = pickle.load(handle)
        for attr, filename in (
            ('pmn_dsk_high', 'darkshading_highISO_k.npy'),
            ('pmn_dsk_low', 'darkshading_lowISO_k.npy'),
            ('pmn_dsb_high', 'darkshading_highISO_b.npy'),
            ('pmn_dsb_low', 'darkshading_lowISO_b.npy'),
        ):
            setattr(self, attr, np.load(resource_dir / filename, mmap_mode='r'))
        self.data_info = []
        long_dir = Path(sid_long_dir)
        for line in Path(pair_list).read_text().splitlines():
            if not line.strip():
                continue
            short, long, iso, *_ = line.split()
            short, long = Path(short).name, Path(long).name
            if not long.startswith('2') or short[:5] != long[:5]:
                raise ValueError(f'Validation requires paired prefix-2 SID scenes: {line}')
            exposure = lambda name: float(Path(name).stem.rsplit('_', 1)[1][:-1])
            ratio = min(exposure(long) / exposure(short), 300.0)
            if not any(np.isclose(ratio, value) for value in ratios):
                continue
            short_path, long_path = long_dir.parent / 'short' / short, long_dir / long
            for path in (short_path, long_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            self.data_info.append(dict(name=long, long=str(long_path), short=[str(short_path)],
                                       ratio=[ratio], ISO=int(iso[3:]),
                                       wb=np.ones(4), ccm=np.eye(3)))
        if max_items is not None:
            # A local RNG keeps the subset reproducible without changing training randomness.
            self.data_info = random.Random(seed).sample(
                self.data_info, min(max_items, len(self.data_info))
            )
        if not self.data_info:
            raise ValueError('No real SID validation pairs match the requested ratios')

    def load_raw_pair(self, idx):
        record = self.data_info[idx]
        with rawpy.imread(record['long']) as raw:
            hr = raw.raw_image_visible.astype(np.float32)
        with rawpy.imread(record['short'][0]) as raw:
            lr = raw.raw_image_visible.astype(np.float32)
        return hr, lr, 0
