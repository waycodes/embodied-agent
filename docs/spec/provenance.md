# Provenance Policy

This document defines dataset identity, versioning, and provenance tracking.

---

## 1. Dataset Identity

### 1.1 Build ID

Every compiled dataset has a unique `build_id`:

```python
def compute_build_id(source_dataset, source_version, transform_config, edk_version, seed):
    payload = json.dumps({
        "source_dataset": source_dataset,
        "source_version": source_version,
        "transform_config": transform_config,
        "edk_version": edk_version,
        "random_seed": seed,
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
```

### 1.2 Episode ID

Globally unique: `{dataset}-{index:06d}-{build_id[:8]}`

Example: `berkeley_autolab_ur5-000042-a1b2c3d4`

---

## 2. Provenance Fields

### 2.1 Dataset-Level (meta/info.json)

```json
{
    "provenance": {
        "build_id": "a1b2c3d4e5f67890",
        "build_timestamp": "2024-01-15T10:30:00Z",
        "edk_version": "0.1.0",
        "source": {
            "dataset": "berkeley_autolab_ur5",
            "version": "1.0.0",
            "split": "train",
            "slice": "[0:1000]"
        },
        "transform_pipeline": ["extract_task_text", "select_camera", "resize_images"],
        "random_seed": 42
    }
}
```

### 2.2 Episode-Level (Parquet)

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Globally unique ID |
| `dataset_id` | string | Source dataset |
| `source_episode_index` | int | Original index |
| `invalid` | bool | Validation status |

---

## 3. Versioning

Schema version format: `MAJOR.MINOR.PATCH`

| Change | Bump | Compatible |
|--------|------|------------|
| Breaking | MAJOR | No |
| New optional fields | MINOR | Yes |
| Bug fixes | PATCH | Yes |

---

## 4. Reproducibility

- All random operations use seeded RNGs
- Seed recorded in provenance
- Configuration hash stored for verification

---

## 5. Lineage Tracking

Parent datasets and merges are tracked:

```json
{
    "provenance": {
        "parent": {"build_id": "...", "query": "..."},
        "sources": [{"dataset": "...", "weight": 0.5}]
    }
}
```
