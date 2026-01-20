"""Policy API and observation/action adapters for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Protocol for policy inference."""
    
    def reset(self) -> None:
        """Reset policy state for new episode."""
        ...
    
    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Predict action from observation.
        
        Args:
            observation: Dict of observation arrays.
        
        Returns:
            Action array.
        """
        ...


class BasePolicy(ABC):
    """Abstract base class for policies."""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset policy state."""
        pass
    
    @abstractmethod
    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Predict action from observation."""
        pass


class RandomPolicy(BasePolicy):
    """Random action policy for testing."""
    
    def __init__(self, action_dim: int = 7, action_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        """Initialize random policy.
        
        Args:
            action_dim: Action dimension.
            action_range: (min, max) action values.
        """
        self.action_dim = action_dim
        self.action_range = action_range
    
    def reset(self) -> None:
        """Reset (no-op for random policy)."""
        pass
    
    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Return random action."""
        low, high = self.action_range
        return np.random.uniform(low, high, size=self.action_dim).astype(np.float32)


@dataclass
class ObservationAdapter:
    """Adapt observations between canonical and policy formats.
    
    Attributes:
        image_key: Key for image observation in policy format.
        state_key: Key for state observation in policy format.
        canonical_image_key: Canonical image key.
        canonical_state_key: Canonical state key.
        image_size: Target image size (H, W).
    """
    
    image_key: str = "image"
    state_key: str = "state"
    canonical_image_key: str = "observation.images.canonical"
    canonical_state_key: str = "observation.state"
    image_size: tuple[int, int] | None = None
    
    def to_policy(self, canonical_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert canonical observation to policy format.
        
        Args:
            canonical_obs: Canonical observation dict.
        
        Returns:
            Policy-format observation dict.
        """
        policy_obs: dict[str, np.ndarray] = {}
        
        # Image
        if self.canonical_image_key in canonical_obs:
            img = canonical_obs[self.canonical_image_key]
            if self.image_size and img.shape[:2] != self.image_size:
                # Resize if needed (simple nearest neighbor)
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((self.image_size[1], self.image_size[0]))
                img = np.array(pil_img)
            policy_obs[self.image_key] = img
        
        # State
        if self.canonical_state_key in canonical_obs:
            policy_obs[self.state_key] = canonical_obs[self.canonical_state_key]
        
        return policy_obs
    
    def from_canonical_step(self, step: "Step") -> dict[str, np.ndarray]:
        """Convert canonical Step to policy observation."""
        from embodied_datakit.schema.step import Step
        return self.to_policy(step.observation)


@dataclass
class ActionAdapter:
    """Adapt actions between canonical and environment formats.
    
    Attributes:
        action_dim: Expected action dimension.
        action_range: (min, max) action values.
        gripper_index: Index of gripper in action (-1 for last).
    """
    
    action_dim: int = 7
    action_range: tuple[float, float] = (-1.0, 1.0)
    gripper_index: int = -1
    
    def to_env(self, policy_action: np.ndarray) -> np.ndarray:
        """Convert policy action to environment format.
        
        Args:
            policy_action: Action from policy.
        
        Returns:
            Environment-format action.
        """
        # Clip to range
        low, high = self.action_range
        action = np.clip(policy_action, low, high)
        
        # Pad or truncate to expected dimension
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))
        elif len(action) > self.action_dim:
            action = action[:self.action_dim]
        
        return action.astype(np.float32)
    
    def from_env(self, env_action: np.ndarray) -> np.ndarray:
        """Convert environment action to canonical format."""
        return self.to_env(env_action)  # Same transformation
