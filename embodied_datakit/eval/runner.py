"""Evaluator runner with metrics aggregation and video recording."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

from embodied_datakit.eval.policy import ActionAdapter, ObservationAdapter, Policy


class Environment(Protocol):
    """Protocol for evaluation environments."""
    
    def reset(self, task: str) -> dict[str, np.ndarray]:
        """Reset environment for task."""
        ...
    
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Execute action and return (obs, reward, done, info)."""
        ...
    
    def get_success(self) -> bool:
        """Check if task was successful."""
        ...


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""
    
    task: str
    episode_idx: int
    success: bool
    total_reward: float
    num_steps: int
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskMetrics:
    """Aggregated metrics for a task."""
    
    task: str
    num_episodes: int
    num_successes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "num_episodes": self.num_episodes,
            "num_successes": self.num_successes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_steps": self.mean_steps,
        }


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    
    tasks: list[str]
    episodes_per_task: int = 10
    max_steps: int = 200
    record_video: bool = False
    video_dir: Path | None = None
    seed: int = 42


class Evaluator:
    """Run policy evaluation and aggregate metrics."""
    
    def __init__(
        self,
        policy: Policy,
        env: Environment,
        obs_adapter: ObservationAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
    ) -> None:
        """Initialize evaluator.
        
        Args:
            policy: Policy to evaluate.
            env: Environment to evaluate in.
            obs_adapter: Observation adapter.
            action_adapter: Action adapter.
        """
        self.policy = policy
        self.env = env
        self.obs_adapter = obs_adapter or ObservationAdapter()
        self.action_adapter = action_adapter or ActionAdapter()
        self._results: list[EpisodeResult] = []
        self._video_frames: list[np.ndarray] = []
    
    def run_episode(
        self, task: str, episode_idx: int, max_steps: int, record_video: bool = False
    ) -> EpisodeResult:
        """Run a single evaluation episode.
        
        Args:
            task: Task name.
            episode_idx: Episode index.
            max_steps: Maximum steps.
            record_video: Whether to record video.
        
        Returns:
            EpisodeResult.
        """
        self.policy.reset()
        obs = self.env.reset(task)
        
        total_reward = 0.0
        frames = []
        
        for step in range(max_steps):
            # Adapt observation
            policy_obs = self.obs_adapter.to_policy(obs)
            
            # Get action
            action = self.policy.predict(policy_obs)
            
            # Adapt action
            env_action = self.action_adapter.to_env(action)
            
            # Step environment
            obs, reward, done, info = self.env.step(env_action)
            total_reward += reward
            
            # Record frame
            if record_video and "image" in obs:
                frames.append(obs["image"].copy())
            
            if done:
                break
        
        success = self.env.get_success()
        
        result = EpisodeResult(
            task=task,
            episode_idx=episode_idx,
            success=success,
            total_reward=total_reward,
            num_steps=step + 1,
        )
        
        if record_video:
            result.info["frames"] = frames
        
        return result
    
    def run(self, config: EvalConfig) -> list[EpisodeResult]:
        """Run full evaluation.
        
        Args:
            config: Evaluation configuration.
        
        Returns:
            List of episode results.
        """
        np.random.seed(config.seed)
        self._results = []
        
        for task in config.tasks:
            for ep_idx in range(config.episodes_per_task):
                result = self.run_episode(
                    task=task,
                    episode_idx=ep_idx,
                    max_steps=config.max_steps,
                    record_video=config.record_video,
                )
                self._results.append(result)
        
        return self._results
    
    def aggregate_metrics(self) -> dict[str, TaskMetrics]:
        """Aggregate metrics by task.
        
        Returns:
            Dict mapping task name to TaskMetrics.
        """
        task_results: dict[str, list[EpisodeResult]] = {}
        for result in self._results:
            if result.task not in task_results:
                task_results[result.task] = []
            task_results[result.task].append(result)
        
        metrics = {}
        for task, results in task_results.items():
            num_episodes = len(results)
            num_successes = sum(1 for r in results if r.success)
            success_rate = num_successes / num_episodes if num_episodes > 0 else 0.0
            mean_reward = np.mean([r.total_reward for r in results])
            mean_steps = np.mean([r.num_steps for r in results])
            
            metrics[task] = TaskMetrics(
                task=task,
                num_episodes=num_episodes,
                num_successes=num_successes,
                success_rate=success_rate,
                mean_reward=float(mean_reward),
                mean_steps=float(mean_steps),
            )
        
        return metrics
    
    def save_results(self, output_dir: Path | str) -> tuple[Path, Path]:
        """Save results to CSV and JSON.
        
        Args:
            output_dir: Output directory.
        
        Returns:
            Tuple of (csv_path, json_path).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "eval_results.csv"
        json_path = output_dir / "eval_results.json"
        
        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "episode", "success", "reward", "steps"])
            writer.writeheader()
            for result in self._results:
                writer.writerow({
                    "task": result.task,
                    "episode": result.episode_idx,
                    "success": int(result.success),
                    "reward": result.total_reward,
                    "steps": result.num_steps,
                })
        
        # Write JSON
        metrics = self.aggregate_metrics()
        summary = {
            "total_episodes": len(self._results),
            "total_successes": sum(1 for r in self._results if r.success),
            "overall_success_rate": sum(1 for r in self._results if r.success) / len(self._results) if self._results else 0.0,
            "per_task": {task: m.to_dict() for task, m in metrics.items()},
        }
        
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return csv_path, json_path
