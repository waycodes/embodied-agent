# Upstream Format Requirements

This document is the source-of-truth for RLDS, LeRobotDataset v3, and Open X-Embodiment format requirements. It defines what "compatibility" means for EmbodiedDataKit.

---

## 1. RLDS (Reinforcement Learning Datasets)

Reference: [google-research/rlds](https://github.com/google-research/rlds)

> [!NOTE]
> The `google-research/rlds` repository is archived (read-only). EmbodiedDataKit relies primarily on TFDS interfaces where possible.

### 1.1 Episode Structure

RLDS datasets are `tf.data.Dataset` objects where each element is an **episode**. Each episode contains a nested `tf.data.Dataset` of **steps**.

```python
# Episode structure
{
    "steps": tf.data.Dataset({
        "is_first": bool,
        "is_last": bool,
        "is_terminal": bool,  # optional
        "observation": {...},
        "action": Tensor,
        "reward": float,
        "discount": float,
    }),
    # Episode-level metadata (optional)
    "episode_id": str,
    "experiment_id": str,
    "invalid": bool,
}
```

### 1.2 Step Field Requirements

| Field | Requirement | Description |
|-------|-------------|-------------|
| `is_first` | **MUST** | Boolean, `True` exactly once at the first step of each episode |
| `is_last` | **MUST** | Boolean, `True` exactly once at the last step of each episode |
| `is_terminal` | SHOULD | Boolean, `True` if the episode ended due to terminal state (vs truncation) |
| `observation` | **MUST** | Dictionary of observations; fields MUST be consistent across all steps |
| `action` | SHOULD | Action taken; semantically invalid after `is_last=True` |
| `reward` | SHOULD | Reward received; semantically invalid after `is_last=True` |
| `discount` | SHOULD | Discount factor; semantically invalid after `is_last=True` |

### 1.3 RLDS Invariants

- **INV-R1**: `is_first=True` MUST appear exactly once, at step index 0
- **INV-R2**: `is_last=True` MUST appear exactly once, at the final step
- **INV-R3**: All steps within an episode MUST have identical field schemas (keys, dtypes, shapes)
- **INV-R4**: After `is_last=True`, fields `action`, `reward`, `discount` are undefined/invalid
- **INV-R5**: After `is_terminal=True`, the observation represents the terminal state

### 1.4 Episode-Level Metadata (Suggested)

| Field | Description |
|-------|-------------|
| `episode_id` | Unique identifier; SHOULD be unique across merged datasets |
| `experiment_id` | Groups related episodes (e.g., same task/robot) |
| `invalid` | Boolean flag for episodes that should be excluded from training |

---

## 2. LeRobotDataset v3

Reference: [LeRobotDataset v3 Documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)

### 2.1 Storage Pillars

LeRobotDataset v3 uses four storage components:

1. **Parquet Files** (`data/`): High-frequency tabular signals (state, action, timestamps)
2. **MP4 Video Shards** (`videos/`): Visual observations, frames from multiple episodes concatenated
3. **Metadata Tables** (`meta/`): Episode segmentation, task mapping, normalization stats
4. **Relational Indexing**: Episode views reconstructed from metadata, not file boundaries

### 2.2 Directory Structure

```
dataset/
├── meta/
│   ├── info.json           # Schema, fps, path templates, provenance
│   ├── stats.json          # Normalization statistics (mean/std/min/max)
│   ├── tasks.jsonl         # Task text → integer ID mapping
│   └── episodes/
│       └── *.parquet       # Per-episode metadata with storage offsets
├── data/
│   └── *.parquet           # Step-level tabular data shards
└── videos/
    └── {camera_name}/
        └── *.mp4           # Video shards per camera
```

### 2.3 Metadata Requirements

#### `meta/info.json` (MUST)

```json
{
    "codebase_version": "v3.0",
    "robot_type": "string",
    "fps": 30,
    "features": {
        "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state": {"dtype": "float32", "shape": [7]},
        "action": {"dtype": "float32", "shape": [7]}
    },
    "splits": {"train": "0:100"},
    "total_episodes": 100,
    "total_frames": 10000,
    "data_path": "data/{episode_chunk:04d}/episode_{episode_id:06d}.parquet",
    "video_path": "videos/{camera}/{episode_chunk:04d}.mp4"
}
```

#### `meta/tasks.jsonl` (MUST)

```jsonl
{"task_index": 0, "task": "Pick up the red block"}
{"task_index": 1, "task": "Place the block in the bin"}
```

#### `meta/stats.json` (SHOULD)

```json
{
    "observation.state": {
        "mean": [0.1, 0.2, ...],
        "std": [0.05, 0.1, ...],
        "min": [-1.0, -1.0, ...],
        "max": [1.0, 1.0, ...]
    },
    "action": {...}
}
```

### 2.4 LeRobot v3 Design Principles

| Principle | Requirement |
|-----------|-------------|
| **Fewer, larger files** | **MUST** target fewer shards to optimize streaming and reduce overhead |
| **Hub-native streaming** | **MUST** support random access without downloading entire dataset |
| **Relational metadata** | **MUST** store episode offsets to reconstruct views from shards |
| **Temporal windows** | SHOULD support delta timestamp indexing for history windows |

### 2.5 Sample Key Naming Convention

```
observation.images.{camera_name}    # e.g., observation.images.front_left
observation.state                    # Proprioceptive state vector
observation.language                 # Task instruction text
action                              # Action vector
timestamp                           # Step timestamp
```

---

## 3. Open X-Embodiment (OXE)

Reference: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)

### 3.1 Dataset Characteristics

- OXE datasets are RLDS episodes stored as TFDS datasets
- Loadable via `tfds.builder_from_directory()` or `tfds.load()`
- Datasets have heterogeneous observation/action spaces
- Coordinate frames and action semantics vary across datasets

### 3.2 Coarse Alignment Strategy

OXE defines **coarse alignment** across embodiments:

| Aspect | Alignment Approach |
|--------|-------------------|
| **Action Space** | Map to 7D end-effector: `[x, y, z, roll, pitch, yaw, gripper]` |
| **Action Semantics** | Track whether values are absolute/delta/velocity |
| **Camera Selection** | Select canonical workspace camera per dataset |
| **Image Resolution** | Resize to common resolution (e.g., 256×256) |
| **Control Rate** | Record dataset-specific control rate; do not force alignment |
| **Coordinate Frame** | Accept that frames are NOT aligned across datasets |

### 3.3 Action Representation

| Field | Description |
|-------|-------------|
| `action[:3]` | End-effector position (x, y, z) |
| `action[3:6]` | End-effector orientation (roll, pitch, yaw OR axis-angle) |
| `action[6]` | Gripper command (0=open, 1=closed, or continuous) |

> [!WARNING]
> Action semantics (absolute pose, delta pose, velocity) are dataset-specific and MUST be tracked in metadata.

### 3.4 Common OXE Fields

```python
# Typical OXE episode structure
{
    "steps": {
        "observation": {
            "image": Tensor[H, W, 3],           # Primary camera
            "wrist_image": Tensor[H, W, 3],     # Optional wrist camera
            "state": Tensor[N],                  # Robot state
        },
        "action": Tensor[7],                     # 7D end-effector action
        "reward": float,
        "is_first": bool,
        "is_last": bool,
        "is_terminal": bool,
        "language_instruction": bytes,           # Task text (often bytes)
    }
}
```

### 3.5 OXE Invariants

- **INV-O1**: Each dataset MUST provide at least one RGB image observation
- **INV-O2**: Action vectors SHOULD be mappable to 7D end-effector representation
- **INV-O3**: Language instructions MAY be bytes or string; decoder MUST handle both
- **INV-O4**: Episode structure follows RLDS invariants (INV-R1 through INV-R5)

---

## 4. Compatibility Matrix

### 4.1 EmbodiedDataKit Guarantees

| Feature | RLDS | LeRobot v3 | OXE |
|---------|------|------------|-----|
| Episode/step structure | Native | Reconstructed from offsets | Native (RLDS) |
| `is_first`/`is_last` | Native | N/A (implicit from offsets) | Native |
| Image observations | Tensor | MP4 frames | Tensor |
| Action vectors | Native | Parquet column | Native |
| Task text | Episode metadata | `tasks.jsonl` | Step field |
| Timestamps | Optional | Native | Optional |
| Streaming support | TFDS | Hub-native | TFDS |

### 4.2 Round-Trip Fidelity

| Conversion | Fidelity |
|------------|----------|
| RLDS → LeRobot v3 | Lossless (with video encoding) |
| LeRobot v3 → RLDS | Lossless (with video decoding) |
| OXE → LeRobot v3 | Lossless |
| LeRobot v3 → OXE | Requires action semantics metadata |

---

## 5. Validation Requirements Summary

### 5.1 MUST Validate

- [ ] RLDS `is_first`/`is_last` invariants
- [ ] Consistent field schemas across steps
- [ ] Non-empty episodes (at least 1 step)
- [ ] Valid tensor dtypes and shapes
- [ ] Task text present (for training consumption)

### 5.2 SHOULD Validate

- [ ] Timestamp monotonicity
- [ ] Action bounds and sanity
- [ ] Image integrity (finite values, correct channels)
- [ ] Episode length within policy bounds

### 5.3 MAY Validate

- [ ] Multi-camera frame count alignment
- [ ] Control rate consistency
- [ ] Language field encoding (bytes vs string)
