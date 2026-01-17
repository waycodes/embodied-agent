# Performance Targets

Default performance targets and sharding configuration.

---

## 1. Sharding Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodes_per_parquet_shard` | 1000 | Episodes per Parquet file |
| `max_video_frames_per_shard` | 10000 | Max frames per MP4 shard |
| `video_crf` | 23 | H.264 CRF quality (0-51) |
| `video_preset` | medium | ffmpeg encoding preset |

## 2. Memory Targets

| Stage | Budget | Description |
|-------|--------|-------------|
| Episode buffering | 100MB | Max per-episode memory |
| Video encoding | 1GB | Per-camera video buffer |
| Index building | 500MB | Episode index memory |

## 3. Throughput Targets

| Operation | Target | Conditions |
|-----------|--------|------------|
| Episode ingestion | 100 eps/sec | Without video decode |
| Video encoding | 30 fps | Single camera, 256x256 |
| Parquet writing | 10K rows/sec | Tabular data only |

## 4. File Count Guidelines

Following LeRobot v3 "fewer, larger files" principle:
- Target < 100 Parquet shards per dataset
- Target < 50 MP4 shards per camera
- Consolidate small episodes into larger shards
