import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _build_stem_map(folder):
    stem_map = {}
    for fname in os.listdir(folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in VALID_EXTS:
            stem_map[stem] = os.path.join(folder, fname)
    return stem_map


class SegDataAugmentation:
    def __init__(self, img_size=224, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

    def transform(self, image, mask, dist):
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        dist = TF.resize(dist, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                dist = TF.hflip(dist)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                dist = TF.vflip(dist)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                dist = TF.rotate(dist, angle, interpolation=InterpolationMode.NEAREST)

        return image, mask, dist


class BCCDMaskDataset(data.Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        edt_dir=None,
        stems=None,
        img_size=224,
        is_train=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.edt_dir = edt_dir
        self.mean = mean
        self.std = std
        self.augm = SegDataAugmentation(img_size=img_size, is_train=is_train)

        image_map = _build_stem_map(image_dir)
        mask_map = _build_stem_map(mask_dir)

        common_stems = sorted(set(image_map.keys()) & set(mask_map.keys()))
        if stems is not None:
            requested = set(stems)
            common_stems = [s for s in common_stems if s in requested]

        if len(common_stems) == 0:
            raise RuntimeError(f"No matching image/mask pairs found in {image_dir} and {mask_dir}")

        self.samples = []
        for stem in common_stems:
            edt_path = None
            if edt_dir is not None:
                candidate = os.path.join(edt_dir, os.path.basename(mask_map[stem]))
                if os.path.exists(candidate):
                    edt_path = candidate
            self.samples.append(
                {
                    "stem": stem,
                    "image_path": image_map[stem],
                    "mask_path": mask_map[stem],
                    "edt_path": edt_path,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        has_dist = 0.0
        if sample["edt_path"] is not None:
            dist = Image.open(sample["edt_path"]).convert("L")
            has_dist = 1.0
        else:
            dist = Image.new("L", size=mask.size, color=0)

        image, mask, dist = self.augm.transform(image, mask, dist)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, mean=self.mean, std=self.std)

        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.int64)
        mask_t = torch.from_numpy(mask_np)

        dist_np = np.array(dist, dtype=np.float32) / 255.0
        dist_t = torch.from_numpy(dist_np).unsqueeze(0)

        return {
            "I": image_t,  # image
            "M": mask_t,   # segmentation target in {0,1}
            "D": dist_t,   # normalized EDT target in [0,1]
            "has_dist": torch.tensor(has_dist, dtype=torch.float32),
            "stem": sample["stem"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
        }
