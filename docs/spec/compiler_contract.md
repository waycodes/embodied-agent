# Compiler Contract

This document defines the minimum viable dataset transformations that EmbodiedDataKit guarantees.

---

## 1. Overview

EmbodiedDataKit is a **dataset compiler** that:
1. Ingests heterogeneous robot trajectory data (RLDS, LeRobot, OXE)
2. Canonicalizes to an internal representation
3. Validates structural and semantic correctness
4. Writes to streamable output formats

---

## 2. Transformation Guarantees

### 2.1 Mandatory Transformations

Every compiled dataset MUST have these transformations applied:

| Transformation | Description |
|----------------|-------------|
| **Episode/Step Structure** | Canonical `Episode` and `Step` objects with RLDS-aligned fields |
| **Episode ID** | Globally unique `episode_id` for each episode |
| **Dataset ID** | Source `dataset_id` preserved in metadata |
| **Task Text** | Non-empty `task_text` field per episode (extracted or inferred) |
| **Timestamps** | `timestamp` field per step (synthesized from control rate if missing) |
| **Flattened Keys** | Nested observations flattened to dotted keys (e.g., `observation.images.front`) |
| **Provenance** | Build metadata recording source, transforms, and code version |

### 2.2 Configurable Transformations

These transformations are applied based on configuration:

| Transformation | Default | Description |
|----------------|---------|-------------|
| **Camera Selection** | First RGB | Select canonical camera from multi-camera datasets |
| **Image Resize** | 256×256 | Resize images to configured resolution |
| **Action Mapping** | Passthrough | Map to canonical 7D end-effector representation |
| **Action Normalization** | Disabled | Apply per-dataset normalization |
| **Invalid Marking** | Enabled | Mark truncated/incomplete episodes as invalid |

### 2.3 Optional Fields (Preserved if Present)

| Field | Description |
|-------|-------------|
| `observation.depth.*` | Depth camera observations |
| `observation.wrist_image` | Wrist-mounted camera |
| `observation.force_torque` | Force/torque sensor readings |
| `observation.proprio` | Additional proprioceptive state |
| `step_metadata.*` | Additional step-level metadata |
| `episode_metadata.*` | Additional episode-level metadata |

---

## 3. Canonical Output Schema

### 3.1 Episode Schema

```python
@dataclass
class Episode:
    # Required fields
    episode_id: str              # Globally unique identifier
    dataset_id: str              # Source dataset identifier
    steps: list[Step]            # Ordered sequence of steps
    
    # Metadata
    task_id: int                 # Integer ID for task (from TaskCatalog)
    task_text: str               # Natural language task description
    num_steps: int               # Number of steps in episode
    
    # Optional fields
    invalid: bool = False        # RLDS invalid flag
    episode_metadata: dict = {}  # Additional metadata
```

### 3.2 Step Schema

```python
@dataclass
class Step:
    # RLDS-aligned required fields
    is_first: bool               # True at step 0 only
    is_last: bool                # True at final step only
    
    # RLDS-aligned optional fields
    is_terminal: bool = False    # True if terminal state reached
    
    # Observations (flattened dict)
    observation: dict[str, np.ndarray]
    # Examples:
    #   "observation.images.front": np.ndarray[H, W, 3]
    #   "observation.state": np.ndarray[N]
    #   "observation.language": str
    
    # Action (post is_last: None or zeros)
    action: np.ndarray | None
    
    # Reward/discount signals
    reward: float | None = None
    discount: float | None = None
    
    # Temporal
    timestamp: float             # Seconds since episode start
    
    # Metadata
    step_metadata: dict = {}
```

### 3.3 DatasetSpec Schema

```python
@dataclass
class DatasetSpec:
    # Identity
    dataset_id: str
    dataset_name: str
    
    # Schema
    observation_schema: dict[str, FeatureSpec]
    action_schema: FeatureSpec
    
    # Temporal
    control_rate_hz: float
    
    # Action semantics
    action_space_type: Literal[
        "ee_delta_7",      # Delta end-effector (x,y,z,r,p,y,g)
        "ee_abs_7",        # Absolute end-effector
        "ee_velocity_7",   # Velocity end-effector
        "joint_delta_n",   # Delta joint angles
        "joint_abs_n",     # Absolute joint angles
        "custom"           # Dataset-specific
    ]
    
    # Cameras
    camera_names: list[str]
    canonical_camera: str | None
    
    # Tasks
    task_catalog: TaskCatalog
    
    # Provenance
    source_uri: str
    build_id: str
    transform_pipeline: list[str]
```

### 3.4 FeatureSpec Schema

```python
@dataclass
class FeatureSpec:
    dtype: str                   # "float32", "uint8", "int64", etc.
    shape: tuple[int, ...]       # Shape of tensor
    description: str = ""        # Human-readable description
    is_video: bool = False       # True for video-encoded features
```

---

## 4. Observation Naming Convention

### 4.1 Hierarchical Keys

All observations use dot-separated hierarchical keys:

```
observation.images.{camera_name}     # RGB images
observation.depth.{camera_name}      # Depth maps
observation.state                    # Proprioceptive state vector
observation.language                 # Task instruction text
observation.proprio.{sensor_name}    # Additional proprioception
```

### 4.2 Reserved Camera Names

| Name | Description |
|------|-------------|
| `front` | Front-facing workspace camera |
| `wrist` | Wrist-mounted camera |
| `overhead` | Top-down overhead camera |
| `left` | Left side camera |
| `right` | Right side camera |

### 4.3 State Vector Convention

The `observation.state` vector should contain:
- Joint positions (if available)
- Joint velocities (if available)
- End-effector pose (if available)
- Gripper state

Exact interpretation is dataset-specific and recorded in `DatasetSpec`.

---

## 5. Action Representation

### 5.1 Canonical 7D End-Effector Actions

For cross-embodiment training, actions are mapped to:

```
action[0:3]  = (x, y, z)           # Position
action[3:6]  = (roll, pitch, yaw)  # Orientation (Euler angles)
action[6]    = gripper             # Gripper command
```

### 5.2 Action Space Types

| Type | Description | Gripper Encoding |
|------|-------------|------------------|
| `ee_delta_7` | Delta end-effector pose | 0=open, 1=close (delta) |
| `ee_abs_7` | Absolute end-effector pose | 0=open, 1=closed (binary) |
| `ee_velocity_7` | Velocity commands | Continuous velocity |
| `joint_delta_n` | Delta joint angles | Last element if included |
| `joint_abs_n` | Absolute joint angles | Last element if included |

### 5.3 Coordinate Frame Note

> [!WARNING]
> Coordinate frames are NOT aligned across datasets. The compiler preserves dataset-specific coordinate frames and records the frame convention in metadata.

---

## 6. Validation Contract

### 6.1 Severity Levels

| Level | Action | Description |
|-------|--------|-------------|
| `ERROR` | Reject | Episode cannot be compiled; fatal structural violation |
| `WARN` | Mark Invalid | Episode compiled but marked `invalid=True`; training skip |
| `INFO` | Log | Statistical observation; no action taken |

### 6.2 Mandatory Validation (ERROR on fail)

- RLDS `is_first`/`is_last` invariants violated
- Empty episode (0 steps)
- Schema drift within episode (inconsistent fields)
- NaN/Inf in observations or actions

### 6.3 Configurable Validation (WARN on fail by default)

- Episode too short (< min_steps)
- Episode too long (> max_steps)
- Timestamp non-monotonic
- Action out of configured bounds
- Missing task text

---

## 7. Provenance Requirements

### 7.1 Build ID Computation

```python
build_id = hash(
    source_dataset_name,
    source_dataset_version,
    transform_pipeline_hash,
    edk_code_version,
    random_seed
)
```

### 7.2 Recorded Provenance Fields

| Field | Description |
|-------|-------------|
| `source_uri` | Original data location |
| `source_split` | Split and slice selector used |
| `transform_pipeline` | Ordered list of transform names |
| `transform_config` | Complete configuration used |
| `edk_version` | EmbodiedDataKit version |
| `build_timestamp` | ISO 8601 timestamp |
| `random_seed` | Seed used for any randomness |

---

## 8. Output Format Contract

### 8.1 LeRobotDataset v3 (Primary)

| Component | Guarantee |
|-----------|-----------|
| `meta/info.json` | Schema, fps, path templates, provenance |
| `meta/tasks.jsonl` | Complete task catalog |
| `meta/stats.json` | Normalization statistics for all numeric fields |
| `meta/episodes/` | Per-episode metadata with storage offsets |
| `data/*.parquet` | Step-level tabular data (sharded) |
| `videos/{camera}/*.mp4` | Video shards per camera |

### 8.2 RLDS/TFDS (Secondary)

| Component | Guarantee |
|-----------|-----------|
| TFRecord shards | Episodes as RLDS-compliant examples |
| TFDS metadata | Loadable via `tfds.builder_from_directory` |
| Feature specification | RLDS-aligned observation/action schema |

---

## 9. API Contract

### 9.1 CLI Commands

| Command | Input | Output |
|---------|-------|--------|
| `edk ingest` | Source URI | DatasetSpec + sample episodes |
| `edk validate` | Source/compiled dataset | Validation report |
| `edk compile` | Source URI + config | Compiled LeRobot v3 dataset |
| `edk index` | Compiled dataset | Episode index (Parquet) |
| `edk slice` | Index + query | New dataset or manifest |
| `edk export-rlds` | Compiled dataset | RLDS/TFDS dataset |
| `edk inspect` | Dataset | Schema + sample summary |

### 9.2 Python API

```python
from embodied_datakit import Compiler, Config

# Compile a dataset
compiler = Compiler(config=Config.from_yaml("config.yaml"))
result = compiler.compile(
    source="oxe://berkeley_autolab_ur5",
    split="train[:10]",
    output="./compiled",
)

# Access compiled data
from embodied_datakit.training import LeRobotDataset
dataset = LeRobotDataset("./compiled")
sample = dataset[0]
```
