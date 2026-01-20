# EmbodiedDataKit

A **dataset compiler** that converts heterogeneous robot trajectories (RLDS, LeRobot, Open X-Embodiment) into a **streamable storage format** with **strict validation** and **sliceable metadata**.

## Features

- **Multi-format ingestion**: Load from RLDS/TFDS, LeRobotDataset v3, Open X-Embodiment
- **Canonical representation**: RLDS-aligned Episode/Step structure
- **Strict validation**: Structural, semantic, and statistical checks
- **LeRobotDataset v3 output**: Parquet + MP4 + relational metadata for Hub streaming
- **RLDS/TFDS export**: Secondary format for ecosystem compatibility
- **Sliceable metadata**: Query and filter by task, robot, episode properties
- **Cross-embodiment transforms**: Camera selection, action mapping, normalization

## Installation

```bash
# Core installation
pip install embodied-datakit

# With TensorFlow Datasets support (for OXE)
pip install embodied-datakit[tfds]

# With video encoding
pip install embodied-datakit[video]

# Full installation
pip install embodied-datakit[all]
```

## Quick Start

```bash
# Probe a dataset
edk ingest oxe://berkeley_autolab_ur5

# Compile to LeRobotDataset v3
edk compile oxe://berkeley_autolab_ur5 --split "train[:100]" -o ./compiled/ur5

# Inspect the result
edk inspect ./compiled/ur5

# Export to RLDS format
edk export-rlds ./compiled/ur5 -o ./rlds/ur5
```

## Documentation

See [docs/](docs/) for detailed documentation:

- [Upstream Formats](docs/spec/upstream_formats.md) - RLDS, LeRobot v3, OXE requirements
- [Compiler Contract](docs/spec/compiler_contract.md) - Transformation guarantees
- [CLI Reference](docs/spec/cli.md) - Command-line interface
- [Validation Policy](docs/spec/validation_policy.md) - Validation rules and severity

## License

**Proprietary – Research/Portfolio Use Only**

This software is provided under a restrictive proprietary license:
- ❌ No commercial or professional use
- ❌ No redistribution or derivative works
- ✅ Non-commercial research with required citation
- ✅ Personal evaluation

If you use this software in academic research, you **must** cite it:

```bibtex
@software{bharadwaj2026embodieddatakit,
  author = {Bharadwaj, Varun},
  title = {EmbodiedDataKit},
  year = {2026},
  url = {https://github.com/waycodes/embodied-agent}
}
```

See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff) for full terms.
