# Action Space Specification

Canonical action representation for cross-embodiment training.

---

## 1. Canonical 7D End-Effector

Primary representation for cross-embodiment tasks:

```
action[0:3]  = (x, y, z)           # Position
action[3:6]  = (roll, pitch, yaw)  # Orientation (Euler)
action[6]    = gripper             # Gripper command
```

---

## 2. Action Space Types

| Type | Description | Gripper |
|------|-------------|---------|
| `ee_delta_7` | Delta position/orientation | 0=open, 1=close |
| `ee_abs_7` | Absolute pose | Binary |
| `ee_velocity_7` | Velocity commands | Continuous |
| `joint_delta_n` | Delta joint angles | Last element |
| `joint_abs_n` | Absolute joint angles | Last element |
| `custom` | Dataset-specific | Varies |

---

## 3. Coordinate Frame

> [!WARNING]
> Coordinate frames are NOT aligned across datasets.
> Frame convention is recorded in metadata.

Common conventions:
- **Base frame**: Robot base origin
- **World frame**: Fixed world coordinates
- **Camera frame**: Camera-centric

---

## 4. Normalization

Actions may be normalized per-dataset:
- Mean/std normalization
- Min/max scaling to [-1, 1]
- Statistics stored in `meta/stats.json`
