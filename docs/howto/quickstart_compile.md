# Quickstart: Compile a Dataset

This guide shows how to compile a robot trajectory dataset to LeRobot v3 format.

## Installation

```bash
pip install embodied-datakit
```

## Basic Compilation

```python
from embodied_datakit.compiler import Compiler
from embodied_datakit.writers import LeRobotV3Writer
from embodied_datakit.validators import RLDSInvariantValidator

# Create compiler
compiler = Compiler()
compiler.add_validator(RLDSInvariantValidator())
compiler.set_writer(LeRobotV3Writer())

# Compile from source
compiler.compile(
    source="oxe://berkeley_autolab_ur5",
    output_dir="./compiled/ur5",
    split="train[:100]",
)
```

## Using Configuration Files

Create a pipeline config `config.yaml`:

```yaml
transforms:
  - name: select_camera
    params:
      camera: front
  - name: resize_images
    params:
      size: [256, 256]
  - name: task_text
    params:
      allow_empty: false
```

Load and apply:

```python
from embodied_datakit.transforms import load_pipeline_config

pipeline = load_pipeline_config("config.yaml")
compiler.set_transform_pipeline(pipeline)
```

## Output Structure

After compilation, you'll have:

```
./compiled/ur5/
├── meta/
│   ├── info.json           # Dataset metadata
│   ├── tasks.jsonl         # Task catalog
│   ├── episodes.parquet    # Episode index
│   └── stats.json          # Statistics
├── data/
│   └── chunk-000/
│       └── episode_*.parquet
└── videos/
    └── chunk-000/
        └── episode_*_front.mp4
```

## Inspecting Results

```python
import pyarrow.parquet as pq

# Read episode index
episodes = pq.read_table("./compiled/ur5/meta/episodes.parquet")
print(f"Total episodes: {len(episodes)}")

# Read a step parquet
steps = pq.read_table("./compiled/ur5/data/chunk-000/episode_000000.parquet")
print(f"Columns: {steps.column_names}")
```
