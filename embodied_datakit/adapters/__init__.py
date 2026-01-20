"""Adapters subpackage for data source ingestion."""

from embodied_datakit.adapters.base import Adapter, BaseAdapter

# Conditional imports for optional dependencies
try:
    from embodied_datakit.adapters.tfds import DirectoryAdapter, TFDSAdapter
except ImportError:
    TFDSAdapter = None  # type: ignore
    DirectoryAdapter = None  # type: ignore

__all__ = ["Adapter", "BaseAdapter", "TFDSAdapter", "DirectoryAdapter"]

