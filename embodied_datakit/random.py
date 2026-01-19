"""Deterministic randomness policy for EmbodiedDataKit."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class DeterministicRNG:
    """Deterministic random number generator with episode-specific seeding."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize with base seed."""
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def base_seed(self) -> int:
        """Get base seed."""
        return self._base_seed

    def get_rng(self) -> np.random.Generator:
        """Get the numpy random generator."""
        return self._rng

    def derive_seed(self, *args: Any) -> int:
        """Derive a deterministic seed from arguments."""
        seed_str = f"{self._base_seed}:" + ":".join(str(arg) for arg in args)
        hash_bytes = hashlib.sha256(seed_str.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big")

    def create_episode_rng(self, episode_id: str) -> np.random.Generator:
        """Create an RNG specific to an episode."""
        episode_seed = self.derive_seed(episode_id)
        return np.random.default_rng(episode_seed)

    def shuffle(self, array: np.ndarray) -> np.ndarray:
        """Shuffle array in place."""
        self._rng.shuffle(array)
        return array

    def permutation(self, n: int) -> np.ndarray:
        """Return permutation of indices."""
        return self._rng.permutation(n)

    def choice(
        self,
        a: int | np.ndarray,
        size: int | None = None,
        replace: bool = True,
        p: np.ndarray | None = None,
    ) -> np.ndarray:
        """Random choice from array or range."""
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | float:
        """Generate uniform random values."""
        return self._rng.uniform(low, high, size)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | int:
        """Generate random integers."""
        return self._rng.integers(low, high, size=size)


# Global RNG instance
_global_rng: DeterministicRNG | None = None


def get_rng() -> DeterministicRNG:
    """Get or create global RNG."""
    global _global_rng
    if _global_rng is None:
        _global_rng = DeterministicRNG()
    return _global_rng


def set_seed(seed: int) -> DeterministicRNG:
    """Set global seed and return RNG."""
    global _global_rng
    _global_rng = DeterministicRNG(seed)
    return _global_rng


def create_rng(seed: int, episode_id: str | None = None) -> np.random.Generator:
    """Create a deterministic RNG, optionally episode-specific."""
    rng = DeterministicRNG(seed)
    if episode_id:
        return rng.create_episode_rng(episode_id)
    return rng.get_rng()


def compute_split_assignment(
    episode_id: str,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> str:
    """Compute deterministic train/val/test split assignment for an episode.

    Uses hash-based splitting for stability across rebuilds.
    """
    # Derive a deterministic hash
    hash_input = f"{seed}:{episode_id}"
    hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)
    normalized = hash_value / 0xFFFFFFFF

    if normalized < train_ratio:
        return "train"
    elif normalized < train_ratio + val_ratio:
        return "val"
    else:
        return "test"
