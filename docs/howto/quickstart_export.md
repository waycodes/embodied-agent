# Quickstart: Export to RLDS

This guide shows how to export a compiled dataset to RLDS/TFDS format.

## Prerequisites

```bash
pip install embodied-datakit[tfds]
```

## Export from Compiled Dataset

```python
from pathlib import Path
from embodied_datakit.writers.rlds_tfds import build_rlds_schema, TFRecordShardWriter
from embodied_datakit.schema import DatasetSpec
import json

# Load dataset spec
with open("./compiled/ur5/meta/info.json") as f:
    info = json.load(f)

spec = DatasetSpec(
    dataset_id=info["dataset_id"],
    dataset_name=info["dataset_name"],
)

# Create RLDS schema
schema = build_rlds_schema(spec)
print("RLDS Schema:", list(schema["episode"]["steps"].keys()))
```

## Write TFRecord Shards

```python
from embodied_datakit.writers.rlds_tfds import TFRecordShardWriter

# Initialize writer
writer = TFRecordShardWriter(
    output_dir="./rlds/ur5",
    episodes_per_shard=100,
    split="train",
)

# Write episodes (from your episode iterator)
for episode in episodes:
    writer.write_episode(episode, spec)

# Finalize and write metadata
writer.finish()
writer.write_tfds_metadata(spec)
```

## Output Structure

```
./rlds/ur5/
├── dataset_info.json
└── train/
    ├── train-00000.tfrecord
    ├── train-00001.tfrecord
    └── ...
```

## Load with TFDS

```python
import tensorflow_datasets as tfds

# Load the exported dataset
builder = tfds.builder_from_directory("./rlds/ur5")
ds = builder.as_dataset(split="train")

for episode in ds.take(1):
    print("Episode keys:", list(episode.keys()))
```
