# CLI Reference

This document defines the EmbodiedDataKit CLI surface area, subcommands, flags, and exit codes.

---

## 1. Overview

The CLI is invoked via the `edk` command (short for EmbodiedDataKit):

```bash
edk <command> [options]
```

---

## 2. Global Options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to YAML configuration file |
| `--verbose`, `-v` | Increase log verbosity (can be repeated) |
| `--quiet`, `-q` | Suppress non-error output |
| `--log-format` | Log format: `text` (default), `json` |
| `--log-file` | Write logs to file in addition to stderr |
| `--version` | Print version and exit |
| `--help`, `-h` | Show help message |

---

## 3. Commands

### 3.1 `edk ingest`

Probe a source dataset and optionally sample episodes.

```bash
edk ingest <source> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<source>` | Source URI (e.g., `oxe://dataset_name`, `tfds://name`, `/path/to/tfds_dir`) |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--split` | `train` | Split to probe |
| `--sample`, `-n` | `0` | Number of episodes to sample (0 = probe only) |
| `--output`, `-o` | stdout | Output path for probe results |
| `--format` | `yaml` | Output format: `yaml`, `json` |

**Examples:**
```bash
# Probe dataset schema
edk ingest oxe://berkeley_autolab_ur5

# Sample 5 episodes
edk ingest oxe://berkeley_autolab_ur5 --sample 5 --output samples/

# Probe specific split slice
edk ingest tfds://bridge --split "train[0:100]"
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Source not found or inaccessible |
| 2 | Parse/probe error |

---

### 3.2 `edk validate`

Run validation on a source or compiled dataset without writing output.

```bash
edk validate <dataset> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<dataset>` | Source URI or path to compiled dataset |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--split` | `train` | Split to validate (for sources) |
| `--slice` | `None` | Slice selector (e.g., `[0:100]`) |
| `--max-episodes` | `None` | Maximum episodes to validate |
| `--report`, `-r` | `None` | Output path for validation report |
| `--format` | `json` | Report format: `json`, `html`, `csv` |
| `--fail-on-warn` | `false` | Exit with error on warnings |
| `--strict` | `false` | Enable all optional validations |

**Examples:**
```bash
# Validate source dataset
edk validate oxe://berkeley_autolab_ur5 --max-episodes 100

# Validate compiled dataset with HTML report
edk validate ./compiled --report report.html --format html

# Strict validation
edk validate ./compiled --strict --fail-on-warn
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Validation passed (no errors) |
| 1 | Dataset not found |
| 2 | Validation errors found |
| 3 | Validation warnings found (with `--fail-on-warn`) |

---

### 3.3 `edk compile`

Compile a source dataset to LeRobotDataset v3 format.

```bash
edk compile <source> --output <path> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<source>` | Source URI or path |

**Required Options:**
| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output directory for compiled dataset |

**Compilation Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--split` | `train` | Split to compile |
| `--slice` | `None` | Slice selector |
| `--pipeline` | `None` | Pipeline config path (overrides default) |
| `--camera` | `None` | Canonical camera name (auto-select if None) |
| `--resolution` | `256x256` | Image resize resolution |
| `--action-mapping` | `passthrough` | Action mapping: `passthrough`, `ee7`, `config` |
| `--normalize-actions` | `false` | Apply action normalization |
| `--skip-validation` | `false` | Skip validation (not recommended) |

**Sharding Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--episodes-per-shard` | `1000` | Episodes per Parquet shard |
| `--max-video-frames` | `10000` | Max frames per MP4 shard |
| `--video-crf` | `23` | Video compression quality (0-51) |

**Execution Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--workers`, `-j` | `1` | Parallel workers for transforms |
| `--fail-fast` | `false` | Stop on first error |
| `--quarantine` | `false` | Write invalid episodes to separate partition |
| `--resume` | `false` | Resume interrupted compilation |
| `--seed` | `42` | Random seed for reproducibility |

**Examples:**
```bash
# Basic compilation
edk compile oxe://berkeley_autolab_ur5 -o ./compiled/ur5

# Compile with custom pipeline and camera
edk compile oxe://bridge --pipeline configs/bridge.yaml --camera front -o ./compiled/bridge

# Compile subset with EE7 action mapping
edk compile tfds://kuka --slice "[0:1000]" --action-mapping ee7 -o ./compiled/kuka

# Resume interrupted compilation
edk compile oxe://rt1 -o ./compiled/rt1 --resume
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Compilation successful |
| 1 | Source not found |
| 2 | Validation errors (compilation aborted) |
| 3 | Write errors |
| 4 | Partial failure (some episodes failed, see report) |

---

### 3.4 `edk index`

Build or rebuild the episode index for a compiled dataset.

```bash
edk index <dataset> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<dataset>` | Path to compiled dataset |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `<dataset>/indexes/` | Output path for index |
| `--rebuild` | `false` | Force rebuild even if index exists |

**Examples:**
```bash
# Build index
edk index ./compiled/ur5

# Rebuild index
edk index ./compiled/ur5 --rebuild
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Index built successfully |
| 1 | Dataset not found |
| 2 | Index build error |

---

### 3.5 `edk slice`

Create a dataset subset based on query predicates.

```bash
edk slice <dataset> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<dataset>` | Path to compiled dataset |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--query`, `-q` | Required | Query predicate (SQL-like) |
| `--output`, `-o` | Required | Output path |
| `--mode` | `copy` | Output mode: `copy` (new dataset), `view` (manifest) |
| `--limit` | `None` | Maximum episodes in slice |

**Query Syntax:**
```sql
-- Filter by robot
robot_id = 'ur5'

-- Filter by task (regex)
task_text LIKE '%pick%'

-- Filter by length
num_steps >= 100 AND num_steps <= 500

-- Combine filters
robot_id = 'ur5' AND task_id IN (1, 2, 3) AND invalid = false
```

**Examples:**
```bash
# Slice by task
edk slice ./compiled --query "task_text LIKE '%pick%'" -o ./slices/pick

# Slice by episode count
edk slice ./compiled --query "num_steps >= 50" --limit 1000 -o ./slices/long

# Create view manifest (no copy)
edk slice ./compiled --query "invalid = false" --mode view -o ./views/valid.json
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Slice created successfully |
| 1 | Dataset/index not found |
| 2 | Query parse error |
| 3 | No episodes match query |

---

### 3.6 `edk export-rlds`

Export compiled dataset to RLDS/TFDS format.

```bash
edk export-rlds <dataset> --output <path> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<dataset>` | Path to compiled LeRobot v3 dataset |

**Required Options:**
| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output directory for RLDS dataset |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--name` | (from source) | TFDS dataset name |
| `--episodes-per-file` | `100` | Episodes per TFRecord shard |
| `--include-video` | `true` | Decode and include video frames |

**Examples:**
```bash
# Export to RLDS
edk export-rlds ./compiled/ur5 -o ./rlds/ur5

# Export with custom name
edk export-rlds ./compiled/ur5 --name my_robot_dataset -o ./rlds/my_robot
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Export successful |
| 1 | Dataset not found |
| 2 | Export error |

---

### 3.7 `edk inspect`

Inspect schema and samples from a dataset.

```bash
edk inspect <dataset> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<dataset>` | Source URI or compiled dataset path |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--split` | `train` | Split to inspect (for sources) |
| `--show-samples` | `3` | Number of sample episodes to show |
| `--format` | `text` | Output format: `text`, `json`, `markdown` |

**Examples:**
```bash
# Inspect compiled dataset
edk inspect ./compiled/ur5

# Inspect source with JSON output
edk inspect oxe://bridge --format json
```

**Output includes:**
- Dataset name and source
- Observation schema (features, dtypes, shapes)
- Action schema and semantics
- Camera list
- Task catalog summary
- Episode count and length statistics
- Sample episode previews

---

### 3.8 `edk eval-rlbench`

Run RLBench-style evaluation.

```bash
edk eval-rlbench <policy> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `<policy>` | Path to policy checkpoint or model |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--protocol` | `default` | Evaluation protocol config name or path |
| `--tasks` | (from protocol) | Comma-separated task names |
| `--episodes` | `25` | Episodes per task |
| `--output`, `-o` | `./eval_results` | Output directory |
| `--save-videos` | `false` | Record evaluation videos |
| `--headless` | `true` | Run in headless mode |
| `--seed` | `0` | Random seed |

**Examples:**
```bash
# Run default evaluation
edk eval-rlbench ./checkpoints/policy.pt -o ./eval/run1

# Run specific tasks with video recording
edk eval-rlbench ./policy.pt --tasks pick_block,place_block --save-videos -o ./eval/run2

# Custom protocol
edk eval-rlbench ./policy.pt --protocol configs/eval_custom.yaml -o ./eval/run3
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Evaluation completed |
| 1 | Policy/protocol not found |
| 2 | RLBench environment error |
| 3 | Evaluation error |

---

## 4. Exit Code Summary

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input not found (dataset, config, etc.) |
| 2 | Processing error (validation, compilation, export) |
| 3 | Partial failure or warning-level issues |
| 4 | Fatal error (unrecoverable) |
| 130 | Interrupted (Ctrl+C) |

---

## 5. Environment Variables

| Variable | Description |
|----------|-------------|
| `EDK_CONFIG` | Default config file path |
| `EDK_CACHE_DIR` | Cache directory for downloads |
| `EDK_LOG_LEVEL` | Default log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `TFDS_DATA_DIR` | TFDS data directory (for OXE datasets) |
| `CUDA_VISIBLE_DEVICES` | GPU selection (for video encoding) |
