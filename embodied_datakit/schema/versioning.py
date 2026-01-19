"""Schema versioning and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

CURRENT_SCHEMA_VERSION = "1.0.0"


@dataclass
class Version:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string."""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
        )

    def __str__(self) -> str:
        """Convert to string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        """Compare versions."""
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)


def can_read(dataset_version: str, reader_version: str | None = None) -> Tuple[bool, str]:
    """Check if reader can read dataset.

    Args:
        dataset_version: Schema version of the dataset.
        reader_version: Schema version of the reader (defaults to current).

    Returns:
        Tuple of (can_read, reason).
    """
    if reader_version is None:
        reader_version = CURRENT_SCHEMA_VERSION

    try:
        d_ver = Version.parse(dataset_version)
        r_ver = Version.parse(reader_version)
    except ValueError as e:
        return False, str(e)

    # Major version must match
    if d_ver.major != r_ver.major:
        return False, f"Major version mismatch: dataset={d_ver.major}, reader={r_ver.major}"

    # Dataset minor version must not be newer than reader
    if d_ver.minor > r_ver.minor:
        return False, f"Dataset has newer features: dataset={d_ver}, reader={r_ver}"

    return True, "Compatible"


def get_current_version() -> str:
    """Get current schema version."""
    return CURRENT_SCHEMA_VERSION


def check_compatibility(dataset_version: str) -> None:
    """Check compatibility and raise if incompatible.

    Raises:
        ValueError: If versions are incompatible.
    """
    can, reason = can_read(dataset_version)
    if not can:
        raise ValueError(f"Incompatible schema version: {reason}")
