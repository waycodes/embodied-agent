# Golden Path Scenario

Minimal end-to-end test scenario for EmbodiedDataKit.

---

## Scenario

**Dataset**: `berkeley_autolab_ur5` (Open X-Embodiment)  
**Episodes**: 1 episode from `train[40:41]`  
**Output**: LeRobot v3 format

---

## Steps

### 1. Probe Dataset

```bash
edk ingest oxe://berkeley_autolab_ur5 --split "train[40:41]" --sample 1
```

Expected: DatasetSpec with observation/action schema.

### 2. Validate Source

```bash
edk validate oxe://berkeley_autolab_ur5 --slice "[40:41]"
```

Expected: No ERRORs, validation report generated.

### 3. Compile to LeRobot v3

```bash
edk compile oxe://berkeley_autolab_ur5 --split "train[40:41]" -o ./test_output
```

Expected: `./test_output/` with meta/, data/, videos/.

### 4. Build Index

```bash
edk index ./test_output
```

Expected: `./test_output/indexes/episodes.parquet`.

### 5. Slice (No-op)

```bash
edk slice ./test_output --query "invalid = false" --mode view -o ./test_view.json
```

Expected: View manifest with 1 episode.

### 6. Inspect Output

```bash
edk inspect ./test_output
```

Expected: Schema, sample, and statistics displayed.

### 7. Smoke Test Training Load

```python
from embodied_datakit.training import LeRobotDataset
ds = LeRobotDataset("./test_output")
sample = ds[0]
assert "observation.images.front" in sample
assert "action" in sample
```

---

## Success Criteria

- [ ] Probe returns valid DatasetSpec
- [ ] Validation passes with 0 ERRORs
- [ ] Compiled dataset has all LeRobot v3 components
- [ ] Index is queryable
- [ ] Training loader returns valid samples
