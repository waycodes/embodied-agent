"""Image standardization transforms."""

from __future__ import annotations

from typing import Any

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.schema.step import Step
from embodied_datakit.transforms.base import BaseTransform


class ImageStandardizeTransform(BaseTransform):
    """Standardize images: resize, enforce channel order, store original shapes."""

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        channel_order: str = "HWC",
        dtype: np.dtype | str = np.uint8,
        keys: list[str] | None = None,
        store_original_shape: bool = True,
    ) -> None:
        """Initialize image standardization transform.

        Args:
            target_size: Target (height, width).
            channel_order: Output channel order ('HWC' or 'CHW').
            dtype: Output dtype.
            keys: Specific image keys. If None, processes all images.
            store_original_shape: Store original shape in step metadata.
        """
        super().__init__("image_standardize")
        self.target_size = target_size
        self.channel_order = channel_order
        self.dtype = np.dtype(dtype)
        self.keys = keys
        self.store_original_shape = store_original_shape

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform episode by standardizing images."""
        new_steps = [self._transform_step(step) for step in episode.steps]
        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            invalid=episode.invalid,
            episode_metadata=episode.episode_metadata,
        )

    def _transform_step(self, step: Step) -> Step:
        """Standardize images in step."""
        new_obs = dict(step.observation)
        original_shapes: dict[str, tuple[int, ...]] = {}

        for key, value in step.observation.items():
            if not self._is_image(key, value):
                continue
            if self.keys is not None and key not in self.keys:
                continue

            if self.store_original_shape:
                original_shapes[key] = value.shape

            new_obs[key] = self._standardize(value)

        new_meta = dict(step.step_metadata) if step.step_metadata else {}
        if original_shapes:
            new_meta["original_image_shapes"] = original_shapes

        return Step(
            is_first=step.is_first,
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            observation=new_obs,
            action=step.action,
            reward=step.reward,
            discount=step.discount,
            timestamp=step.timestamp,
            step_metadata=new_meta if new_meta else None,
        )

    def _is_image(self, key: str, value: Any) -> bool:
        """Check if value is an image."""
        return "image" in key.lower() and isinstance(value, np.ndarray) and value.ndim == 3

    def _standardize(self, image: np.ndarray) -> np.ndarray:
        """Resize, reorder channels, cast dtype."""
        # Detect input channel order
        if image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[1]:
            # CHW format
            image = np.transpose(image, (1, 2, 0))  # -> HWC

        # Resize
        image = self._resize(image)

        # Convert to target channel order
        if self.channel_order == "CHW":
            image = np.transpose(image, (2, 0, 1))

        return image.astype(self.dtype)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        h, w = image.shape[:2]
        th, tw = self.target_size
        if (h, w) == (th, tw):
            return image

        try:
            from PIL import Image
            pil = Image.fromarray(image)
            resized = pil.resize((tw, th), Image.BILINEAR)
            return np.array(resized)
        except ImportError:
            # Nearest neighbor fallback
            row_idx = (np.arange(th) * h / th).astype(int)
            col_idx = (np.arange(tw) * w / tw).astype(int)
            return image[row_idx[:, None], col_idx]
