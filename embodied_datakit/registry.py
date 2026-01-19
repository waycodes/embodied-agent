"""Plugin registry for adapters, transforms, validators, and writers."""

from __future__ import annotations

from typing import Callable, TypeVar

from embodied_datakit.adapters.base import Adapter
from embodied_datakit.transforms.base import Transform
from embodied_datakit.validators.base import Validator
from embodied_datakit.writers.base import Writer

T = TypeVar("T")


class Registry:
    """Generic plugin registry."""

    def __init__(self, name: str) -> None:
        """Initialize registry.

        Args:
            name: Registry name for error messages.
        """
        self.name = name
        self._registry: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a plugin.

        Args:
            name: Plugin name.

        Returns:
            Decorator function.
        """
        def decorator(cls: Callable[..., T]) -> Callable[..., T]:
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Callable[..., T]:
        """Get a registered plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin class/factory.

        Raises:
            KeyError: If plugin not found.
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"Unknown {self.name}: {name}. Available: {available}")
        return self._registry[name]

    def list(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if plugin is registered."""
        return name in self._registry


# Global registries
adapters = Registry[Adapter]("adapter")
transforms = Registry[Transform]("transform")
validators = Registry[Validator]("validator")
writers = Registry[Writer]("writer")


def get_adapter(name: str, **kwargs: object) -> Adapter:
    """Get and instantiate an adapter by name."""
    adapter_cls = adapters.get(name)
    return adapter_cls(**kwargs)


def get_transform(name: str, **kwargs: object) -> Transform:
    """Get and instantiate a transform by name."""
    transform_cls = transforms.get(name)
    return transform_cls(**kwargs)


def get_validator(name: str, **kwargs: object) -> Validator:
    """Get and instantiate a validator by name."""
    validator_cls = validators.get(name)
    return validator_cls(**kwargs)


def get_writer(name: str, **kwargs: object) -> Writer:
    """Get and instantiate a writer by name."""
    writer_cls = writers.get(name)
    return writer_cls(**kwargs)
