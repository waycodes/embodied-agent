# Adding a Custom Adapter

This guide shows how to add a custom adapter for a new dataset format.

## Adapter Interface

All adapters implement the `Adapter` protocol:

```python
from embodied_datakit.adapters.base import BaseAdapter
from embodied_datakit.schema import Episode, DatasetSpec
from typing import Iterator

class MyAdapter(BaseAdapter):
    """Adapter for MyDataset format."""
    
    def __init__(self, source_uri: str) -> None:
        super().__init__(source_uri)
        # Initialize your data loading
    
    def get_spec(self) -> DatasetSpec:
        """Return dataset specification."""
        return DatasetSpec(
            dataset_id="my_dataset",
            dataset_name="My Dataset",
            observation_schema={
                "observation.images.front": FeatureSpec(
                    dtype="uint8",
                    shape=(256, 256, 3),
                ),
                "observation.state": FeatureSpec(
                    dtype="float32",
                    shape=(7,),
                ),
            },
            action_schema=FeatureSpec(dtype="float32", shape=(7,)),
        )
    
    def iter_episodes(self) -> Iterator[Episode]:
        """Iterate over episodes."""
        for raw_episode in self._load_raw_episodes():
            yield self._convert_episode(raw_episode)
    
    def _convert_episode(self, raw: dict) -> Episode:
        """Convert raw episode to canonical format."""
        steps = []
        for i, raw_step in enumerate(raw["steps"]):
            step = Step(
                is_first=(i == 0),
                is_last=(i == len(raw["steps"]) - 1),
                observation={
                    "observation.images.front": raw_step["image"],
                    "observation.state": raw_step["state"],
                },
                action=raw_step.get("action"),
                reward=raw_step.get("reward", 0.0),
            )
            steps.append(step)
        
        return Episode(
            episode_id=raw["id"],
            dataset_id="my_dataset",
            steps=steps,
            task_text=raw.get("task", ""),
        )
```

## Register the Adapter

```python
from embodied_datakit.registry import register_adapter

register_adapter("my_format", MyAdapter)
```

## Use with CLI

```bash
edk compile my_format://path/to/data -o ./compiled/my_data
```

## Key Considerations

1. **RLDS Invariants**: Ensure `is_first` is True only for step 0, `is_last` only for the final step
2. **Observation Keys**: Use canonical dotted keys (`observation.images.{camera}`, `observation.state`)
3. **Action Format**: Last step should have `action=None` per RLDS convention
4. **Task Text**: Extract and normalize task descriptions
