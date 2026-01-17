# Validation Policy

This document defines the validation severity model, validation rules, and execution modes for EmbodiedDataKit.

---

## 1. Severity Levels

### 1.1 Definitions

| Level | Action | Description |
|-------|--------|-------------|
| `ERROR` | **Reject Episode** | Fatal violation; episode cannot be compiled. Compilation aborts or quarantines the episode. |
| `WARN` | **Mark Invalid** | Episode compiled but marked with `invalid=True`. Training code should skip by default. |
| `INFO` | **Log Only** | Statistical observation or minor issue. No action taken; recorded in validation report. |

### 1.2 Escalation Policy

- Default severity levels can be escalated via configuration
- `INFO` → `WARN`: Add to warning tracking
- `WARN` → `ERROR`: Use `--strict` mode or `fail_on_warn: true` in config
- Downgrading severity is NOT supported (prevents silent data corruption)

---

## 2. Validation Categories

### 2.1 Structural Validation (ERROR)

These validations enforce fundamental RLDS invariants. Violations indicate corrupt or malformed data.

| Code | Rule | Description |
|------|------|-------------|
| `E001` | `is_first` invariant | `is_first=True` MUST appear exactly once at step 0 |
| `E002` | `is_last` invariant | `is_last=True` MUST appear exactly once at final step |
| `E003` | Empty episode | Episode MUST have at least 1 step |
| `E004` | Schema consistency | All steps MUST have identical field keys |
| `E005` | Dtype consistency | Field dtypes MUST not change within episode |
| `E006` | Shape consistency | Fixed-shape fields MUST maintain shape (variable-length allowed if declared) |

### 2.2 Semantic Validation (WARN)

These validations catch data quality issues that may affect training but don't prevent compilation.

| Code | Rule | Description |
|------|------|-------------|
| `W001` | `is_terminal` missing | `is_terminal` field not present (defaults to `is_last`) |
| `W002` | Action after `is_last` | Non-null action at final step (semantically invalid per RLDS) |
| `W003` | Reward after `is_last` | Non-null reward at final step |
| `W004` | Episode too short | Episode length < configured `min_episode_length` |
| `W005` | Episode too long | Episode length > configured `max_episode_length` |
| `W006` | Truncated episode | `is_last=True` but `is_terminal=False` (episode interrupted) |
| `W007` | Missing task text | Task text empty or not extractable |
| `W008` | Bytes task text | Task text is bytes (should be decoded to string) |

### 2.3 Numeric Validation (WARN)

| Code | Rule | Description |
|------|------|-------------|
| `W101` | NaN in observation | NaN value detected in observation tensor |
| `W102` | Inf in observation | Infinite value detected in observation tensor |
| `W103` | NaN in action | NaN value detected in action vector |
| `W104` | Inf in action | Infinite value detected in action vector |
| `W105` | Action out of bounds | Action value outside configured bounds |
| `W106` | Extreme action | Action value > 5σ from dataset mean |

### 2.4 Temporal Validation (WARN/INFO)

| Code | Rule | Severity | Description |
|------|------|----------|-------------|
| `W201` | Timestamp non-monotonic | WARN | Timestamp decreased between consecutive steps |
| `W202` | Large timestamp gap | WARN | Timestamp gap > 2× expected interval |
| `I201` | Missing timestamps | INFO | Timestamps synthesized from control rate |
| `I202` | Variable control rate | INFO | Control rate varies > 10% within episode |

### 2.5 Image Validation (WARN/INFO)

| Code | Rule | Severity | Description |
|------|------|----------|-------------|
| `W301` | Invalid image dtype | WARN | Image dtype not uint8 or float32 |
| `W302` | Invalid channel count | WARN | Image channels not 1, 3, or 4 |
| `W303` | Corrupted frame | WARN | Image contains NaN/Inf values |
| `I301` | Low entropy image | INFO | Image has unusually low entropy (possible blank) |
| `I302` | Resolution mismatch | INFO | Image resolution differs from declared |

### 2.6 Multi-Camera Validation (WARN/INFO)

| Code | Rule | Severity | Description |
|------|------|----------|-------------|
| `W401` | Frame count mismatch | WARN | Camera frame count ≠ step count |
| `W402` | Missing camera | WARN | Declared camera not present in episode |
| `I401` | Single camera available | INFO | Only one camera in multi-camera dataset |

### 2.7 Metadata Validation (WARN)

| Code | Rule | Description |
|------|------|-------------|
| `W501` | Missing dataset_id | Dataset identifier not present |
| `W502` | Missing robot_id | Robot identifier not present |
| `W503` | Missing action_space_type | Action space type not declared |
| `W504` | Missing control_rate | Control rate not specified |

---

## 3. Validation Report

### 3.1 Report Structure

```json
{
    "summary": {
        "total_episodes": 1000,
        "valid_episodes": 950,
        "invalid_episodes": 45,
        "error_episodes": 5,
        "error_count": 5,
        "warn_count": 120,
        "info_count": 500
    },
    "by_severity": {
        "ERROR": {"E001": 3, "E003": 2},
        "WARN": {"W004": 45, "W101": 30, "W201": 45},
        "INFO": {"I201": 500}
    },
    "findings": [
        {
            "episode_id": "ep_0001",
            "step_index": null,
            "code": "E001",
            "severity": "ERROR",
            "message": "is_first=True at step 5, expected step 0",
            "field": "is_first",
            "value": "True at step 5"
        }
    ]
}
```

### 3.2 Report Outputs

| Format | File | Contents |
|--------|------|----------|
| JSON | `validation_summary.json` | Summary statistics and aggregated counts |
| JSONL | `findings.jsonl` | Per-finding records (streaming-friendly) |
| HTML | `validation_report.html` | Human-readable report with charts |
| CSV | `findings.csv` | Tabular findings for analysis |

---

## 4. Execution Modes

### 4.1 Default Mode

```bash
edk compile <source> -o <output>
```

- Episodes with `ERROR` findings are **rejected** (not written)
- Episodes with `WARN` findings are **marked invalid** (`invalid=True`)
- Compilation continues after errors (non-fatal)
- Report written to `<output>/reports/`

### 4.2 Fail-Fast Mode

```bash
edk compile <source> -o <output> --fail-fast
```

- Compilation **aborts** on first `ERROR` finding
- Use for strict CI/CD pipelines
- Exit code 2 on first error

### 4.3 Quarantine Mode

```bash
edk compile <source> -o <output> --quarantine
```

- Invalid episodes written to separate partition: `<output>/quarantine/`
- Main dataset contains only valid episodes
- Allows post-hoc analysis of problematic episodes

### 4.4 Strict Mode

```bash
edk compile <source> -o <output> --strict
```

- Elevates all `WARN` to `ERROR`
- Enables all optional validations
- Zero-tolerance for data quality issues

### 4.5 Lenient Mode (Not Recommended)

```bash
edk compile <source> -o <output> --skip-validation
```

> [!WARNING]
> Skipping validation may produce corrupted or unusable datasets. Use only for debugging.

- Disables all validation
- Episodes written regardless of issues
- No validation report generated

---

## 5. Configuration

### 5.1 Validation Config Schema

```yaml
validation:
  # Severity overrides
  severity_overrides:
    W004: ERROR  # Escalate short episodes to error
    W201: INFO   # Downgrade timestamp warnings to info

  # Thresholds
  thresholds:
    min_episode_length: 10
    max_episode_length: 10000
    action_bounds: [-1.5, 1.5]
    action_sigma_threshold: 5.0
    timestamp_gap_factor: 2.0

  # Mode flags
  fail_fast: false
  fail_on_warn: false
  quarantine: false
  strict: false

  # Report options
  report:
    formats: [json, html]
    include_findings: true
    max_findings_per_code: 100
```

### 5.2 Per-Dataset Overrides

```yaml
# configs/pipelines/bridge.yaml
validation:
  thresholds:
    min_episode_length: 5  # Bridge has short demos
  severity_overrides:
    W007: INFO  # Some Bridge episodes lack task text
```

---

## 6. Invalid Episode Handling

### 6.1 RLDS `invalid` Flag Semantics

Per RLDS documentation, the `invalid` flag indicates episodes that should be excluded from training:

> Episodes that should be skipped during training (e.g., corrupted, incomplete, or failed demonstrations).

### 6.2 Training Consumption

Training code SHOULD filter invalid episodes by default:

```python
from embodied_datakit.training import LeRobotDataset

# Default: skip invalid episodes
dataset = LeRobotDataset("./compiled", skip_invalid=True)

# Include invalid for analysis
dataset_all = LeRobotDataset("./compiled", skip_invalid=False)
```

### 6.3 Querying Invalid Episodes

```bash
# Count invalid
edk slice ./compiled --query "invalid = true" --mode view -o /dev/null

# Extract invalid for debugging
edk slice ./compiled --query "invalid = true" -o ./invalid_subset
```

---

## 7. Best Practices

1. **Always validate before training**: Run `edk validate` on compiled datasets
2. **Review validation reports**: Investigate patterns in warnings
3. **Use strict mode for production**: Ensure high-quality training data
4. **Quarantine for debugging**: Analyze problematic episodes separately
5. **Track validation metrics**: Monitor data quality over time
