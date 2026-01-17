# Storage Targets

Primary and secondary storage format decisions.

---

## 1. Primary: LeRobotDataset v3

**Rationale**: Explicitly designed for Hub-native streaming with relational metadata.

### Components
- `meta/info.json` - Schema, fps, path templates
- `meta/tasks.jsonl` - Task catalog
- `meta/stats.json` - Normalization stats
- `meta/episodes/*.parquet` - Episode metadata with offsets
- `data/*.parquet` - Step-level tabular data
- `videos/{camera}/*.mp4` - Video shards

### Advantages
- Hub streaming without full download
- Relational indexing for slicing
- Parquet for efficient column access
- MP4 for compressed video storage

---

## 2. Secondary: RLDS/TFDS

**Rationale**: Compatibility with Open X-Embodiment ecosystem.

### Components
- TFRecord shards with RLDS episodes
- TFDS metadata for `builder_from_directory`
- RLDS-compliant step structure

### Use Cases
- Integration with existing RLDS pipelines
- OXE-compatible data sharing

---

## 3. Optional Extension: Robo-DM

**Status**: Future consideration

**Description**: EBML container with compression claims.

**When to consider**: High-performance caching requirements.

---

## 4. Dependency Strategy

| Format | Dependencies | Extra |
|--------|--------------|-------|
| LeRobot v3 | pyarrow, ffmpeg | Core |
| RLDS/TFDS | tensorflow-datasets | `[tfds]` |
| Robo-DM | (TBD) | Optional |
