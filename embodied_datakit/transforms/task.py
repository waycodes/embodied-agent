"""Task text extraction and normalization transform."""

from __future__ import annotations

import re
import unicodedata

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.transforms.base import BaseTransform


def normalize_task_text(text: str | bytes | None) -> str:
    """Normalize task text to canonical form.
    
    - Decode bytes to string
    - Normalize unicode
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TaskTextTransform(BaseTransform):
    """Extract and normalize task text from episodes.
    
    Sources (in priority order):
    1. Episode task_text field
    2. observation.language from first step
    3. episode_metadata["task"] or ["instruction"]
    4. Default fallback text
    """
    
    def __init__(
        self,
        default_text: str = "",
        allow_empty: bool = False,
        language_key: str = "observation.language",
    ) -> None:
        """Initialize task text transform.
        
        Args:
            default_text: Fallback text if no task found.
            allow_empty: If False, raise on empty task text.
            language_key: Observation key for language instruction.
        """
        super().__init__("task_text")
        self.default_text = default_text
        self.allow_empty = allow_empty
        self.language_key = language_key
    
    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Extract and normalize task text."""
        task_text = self._extract_task_text(episode)
        task_text = normalize_task_text(task_text)
        
        if not task_text:
            task_text = self.default_text
        
        if not task_text and not self.allow_empty:
            raise ValueError(f"Empty task_text for episode {episode.episode_id}")
        
        # Register task in catalog
        task_id = spec.task_catalog.get_or_add(task_text) if task_text else 0
        
        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=episode.steps,
            task_id=task_id,
            task_text=task_text,
            invalid=episode.invalid,
            episode_metadata=episode.episode_metadata,
        )
    
    def _extract_task_text(self, episode: Episode) -> str | bytes | None:
        """Extract task text from available sources."""
        # 1. Episode task_text
        if episode.task_text:
            return episode.task_text
        
        # 2. observation.language from first step
        if episode.steps:
            lang = episode.steps[0].observation.get(self.language_key)
            if lang:
                return lang
        
        # 3. episode_metadata
        for key in ["task", "instruction", "task_text", "language_instruction"]:
            if key in episode.episode_metadata:
                return episode.episode_metadata[key]
        
        return None
