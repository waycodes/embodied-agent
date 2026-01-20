"""Image integrity and alignment validators."""

from __future__ import annotations

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.keys import is_image_key
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.validators.base import BaseValidator, Finding, Severity


class ImageIntegrityValidator(BaseValidator):
    """Validate image data integrity.
    
    Checks:
    - Image is numpy array
    - Correct dtype (uint8 for images)
    - Finite values (no NaN/Inf)
    - Valid shape (H, W, C) or (H, W)
    - Non-empty dimensions
    """
    
    def __init__(self, expected_dtype: str = "uint8") -> None:
        """Initialize image integrity validator.
        
        Args:
            expected_dtype: Expected image dtype.
        """
        super().__init__("image_integrity")
        self.expected_dtype = expected_dtype
    
    def validate_episode(self, episode: Episode, spec: DatasetSpec) -> list[Finding]:
        """Validate image integrity for all steps."""
        findings: list[Finding] = []
        
        for step_idx, step in enumerate(episode.steps):
            for key, value in step.observation.items():
                if not is_image_key(key):
                    continue
                
                findings.extend(self._validate_image(
                    value, key, episode.episode_id, step_idx
                ))
        
        return findings
    
    def _validate_image(
        self, value: object, key: str, episode_id: str, step_idx: int
    ) -> list[Finding]:
        """Validate a single image."""
        findings: list[Finding] = []
        
        # Check type
        if not isinstance(value, np.ndarray):
            findings.append(Finding(
                code=self.name,
                severity=Severity.ERROR,
                message=f"Image {key} is not numpy array",
                episode_id=episode_id,
                step_index=step_idx,
                field=key,
            ))
            return findings
        
        # Check dtype
        if value.dtype.name != self.expected_dtype:
            findings.append(Finding(
                code=self.name,
                severity=Severity.WARN,
                message=f"Image {key} dtype {value.dtype} != {self.expected_dtype}",
                episode_id=episode_id,
                step_index=step_idx,
                field=key,
            ))
        
        # Check shape
        if value.ndim not in (2, 3):
            findings.append(Finding(
                code=self.name,
                severity=Severity.ERROR,
                message=f"Image {key} has invalid ndim={value.ndim}",
                episode_id=episode_id,
                step_index=step_idx,
                field=key,
            ))
        elif any(d == 0 for d in value.shape):
            findings.append(Finding(
                code=self.name,
                severity=Severity.ERROR,
                message=f"Image {key} has zero dimension: {value.shape}",
                episode_id=episode_id,
                step_index=step_idx,
                field=key,
            ))
        
        # Check finite values (for float images)
        if np.issubdtype(value.dtype, np.floating):
            if not np.all(np.isfinite(value)):
                findings.append(Finding(
                    code=self.name,
                    severity=Severity.ERROR,
                    message=f"Image {key} contains NaN/Inf",
                    episode_id=episode_id,
                    step_index=step_idx,
                    field=key,
                ))
        
        return findings


class ImageAlignmentValidator(BaseValidator):
    """Validate image alignment across steps.
    
    Checks:
    - Consistent image keys across all steps
    - Consistent shapes for each camera
    - No missing frames
    """
    
    def __init__(self) -> None:
        """Initialize image alignment validator."""
        super().__init__("image_alignment")
    
    def validate_episode(self, episode: Episode, spec: DatasetSpec) -> list[Finding]:
        """Validate image alignment across steps."""
        findings: list[Finding] = []
        
        if not episode.steps:
            return findings
        
        # Get image keys from first step
        first_step = episode.steps[0]
        image_keys = {k for k in first_step.observation if is_image_key(k)}
        
        if not image_keys:
            return findings
        
        # Get reference shapes
        ref_shapes: dict[str, tuple[int, ...]] = {}
        for key in image_keys:
            val = first_step.observation.get(key)
            if isinstance(val, np.ndarray):
                ref_shapes[key] = val.shape
        
        # Check all steps
        for step_idx, step in enumerate(episode.steps):
            step_image_keys = {k for k in step.observation if is_image_key(k)}
            
            # Check for missing keys
            missing = image_keys - step_image_keys
            for key in missing:
                findings.append(Finding(
                    code=self.name,
                    severity=Severity.ERROR,
                    message=f"Missing image {key} at step {step_idx}",
                    episode_id=episode.episode_id,
                    step_index=step_idx,
                    field=key,
                ))
            
            # Check for extra keys
            extra = step_image_keys - image_keys
            for key in extra:
                findings.append(Finding(
                    code=self.name,
                    severity=Severity.WARN,
                    message=f"Extra image {key} at step {step_idx}",
                    episode_id=episode.episode_id,
                    step_index=step_idx,
                    field=key,
                ))
            
            # Check shapes
            for key in image_keys & step_image_keys:
                val = step.observation.get(key)
                if isinstance(val, np.ndarray) and key in ref_shapes:
                    if val.shape != ref_shapes[key]:
                        findings.append(Finding(
                            code=self.name,
                            severity=Severity.ERROR,
                            message=f"Shape mismatch for {key}: {val.shape} != {ref_shapes[key]}",
                            episode_id=episode.episode_id,
                            step_index=step_idx,
                            field=key,
                        ))
        
        return findings
