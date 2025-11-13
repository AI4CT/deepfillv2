"""Dataset utilities for bubble occlusion inpainting.

This module provides dataset classes that read pre-generated triplets of
blocked (occluded) images, corresponding binary masks (where 255 denotes
visible pixels), and ground-truth origin images. The datasets output
normalized single-channel tensors suitable for the grayscale DeepFill v2
pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

_SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class BubbleSample:
    """Container describing a single bubble inpainting sample."""

    blocked_path: Path
    mask_path: Path
    origin_path: Path
    category: str


def _strip_suffix(name: str, suffix: str) -> str:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _resolve_partner(directory: Path, stem: str, suffix: str) -> Optional[Path]:
    """Locate a file with a specific suffix and supported extension."""

    for ext in _SUPPORTED_EXTS:
        candidate = directory / f"{stem}{suffix}{ext}"
        if candidate.exists():
            return candidate
    matches = list(directory.glob(f"{stem}{suffix}.*"))
    return matches[0] if matches else None


def _collect_triplets(base_dir: Path, category: Optional[str] = None) -> List[BubbleSample]:
    blocked_dir = base_dir / "Blocked"
    mask_dir = base_dir / "Mask"
    origin_dir = base_dir / "Origin"

    if not blocked_dir.is_dir() or not mask_dir.is_dir() or not origin_dir.is_dir():
        LOGGER.debug("Skipping directory %s because required sub-folders are missing", base_dir)
        return []

    category = category or base_dir.name
    samples: List[BubbleSample] = []

    for blocked_path in sorted(blocked_dir.glob("*")):
        if not blocked_path.is_file() or blocked_path.suffix.lower() not in _SUPPORTED_EXTS:
            continue

        stem_with_suffix = blocked_path.stem
        base_stem = _strip_suffix(stem_with_suffix, "_blocked")

        mask_path = _resolve_partner(mask_dir, base_stem, "_mask")
        origin_path = _resolve_partner(origin_dir, base_stem, "_original")

        if mask_path is None or origin_path is None:
            LOGGER.warning(
                "Incomplete triplet for %s (mask: %s, origin: %s)",
                blocked_path,
                mask_path,
                origin_path,
            )
            continue

        samples.append(
            BubbleSample(
                blocked_path=blocked_path,
                mask_path=mask_path,
                origin_path=origin_path,
                category=category,
            )
        )

    return samples


def _discover_samples(root: Path) -> List[BubbleSample]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    # Two supported layouts:
    # 1. root contains category folders (e.g., 0-10/) each with Blocked/Mask/Origin.
    # 2. root directly contains Blocked/Mask/Origin sub-folders.
    samples: List[BubbleSample] = []

    direct_samples = _collect_triplets(root, category=root.name)
    if direct_samples:
        samples.extend(direct_samples)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"Blocked", "Mask", "Origin", "Segmented"}:
            # Already handled by direct layout.
            continue
        samples.extend(_collect_triplets(child, category=child.name))

    return samples


def _sort_categories(categories: Iterable[str]) -> List[str]:
    def _key(cat: str) -> Tuple[int, str]:
        try:
            prefix = cat.split('-')[0]
            return int(prefix), cat
        except ValueError:
            return (1 << 30, cat)

    return sorted(set(categories), key=_key)


class BubbleInpaintDataset(Dataset):
    """PyTorch dataset for bubble occlusion inpainting."""

    def __init__(
        self,
        root: str,
        opt,
        phase: str = "train",
        return_paths: bool = False,
        strict: bool = True,
        allowed_categories: Optional[Sequence[str]] = None,
        samples: Optional[List[BubbleSample]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.opt = opt
        self.phase = phase
        self.return_paths = return_paths
        self.imgsize: int = getattr(opt, "imgsize", 128)
        self._allowed_categories = set(allowed_categories) if allowed_categories else None

        all_samples = list(samples) if samples is not None else _discover_samples(self.root)
        self._all_samples: List[BubbleSample] = all_samples

        if self._allowed_categories is not None:
            working_samples = [s for s in all_samples if s.category in self._allowed_categories]
        else:
            working_samples = list(all_samples)

        if not working_samples:
            message = f"No samples discovered under {self.root}"
            if self._allowed_categories:
                message += f" with categories {sorted(self._allowed_categories)}"
            if strict:
                raise RuntimeError(message)
            LOGGER.warning(message)

        self._samples: List[BubbleSample] = working_samples
        self.samples = self._samples  # backward compatibility

        self.available_categories: List[str] = _sort_categories(s.category for s in self._all_samples)
        self.active_categories: List[str] = _sort_categories(s.category for s in self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]

        blocked = cv2.imread(str(sample.blocked_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        origin = cv2.imread(str(sample.origin_path), cv2.IMREAD_GRAYSCALE)

        if blocked is None or mask is None or origin is None:
            raise RuntimeError(f"Failed to read triplet: {sample}")

        blocked = self._resize_if_needed(blocked)
        mask = self._resize_if_needed(mask)
        origin = self._resize_if_needed(origin)

        blocked_norm = blocked.astype(np.float32) / 255.0
        origin_norm = origin.astype(np.float32) / 255.0

        mask_norm = mask.astype(np.float32) / 255.0
        edit_mask = 1.0 - mask_norm  # 1 -> occluded region
        edit_mask = np.clip(edit_mask, 0.0, 1.0)

        # Remove occlusion content to mimic classic inpainting input.
        blocked_clean = blocked_norm * (1.0 - edit_mask)

        blocked_tensor = torch.from_numpy(blocked_clean).unsqueeze(0)
        origin_tensor = torch.from_numpy(origin_norm).unsqueeze(0)
        mask_tensor = torch.from_numpy(edit_mask).unsqueeze(0)

        if self.return_paths:
            return blocked_tensor, mask_tensor, origin_tensor, str(sample.blocked_path)
        return blocked_tensor, mask_tensor, origin_tensor

    def _resize_if_needed(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == self.imgsize and arr.shape[1] == self.imgsize:
            return arr
        return cv2.resize(arr, (self.imgsize, self.imgsize), interpolation=cv2.INTER_AREA)

    def filtered_dataset(self, categories: Sequence[str], strict: bool = True) -> "BubbleInpaintDataset":
        allowed = set(categories)
        filtered = [s for s in self._all_samples if s.category in allowed]
        return BubbleInpaintDataset(
            root=str(self.root),
            opt=self.opt,
            phase=self.phase,
            return_paths=self.return_paths,
            strict=strict,
            allowed_categories=None,
            samples=filtered,
        )


__all__ = ["BubbleInpaintDataset", "BubbleSample"]
