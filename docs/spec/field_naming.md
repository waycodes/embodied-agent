# Observation Field Naming Convention

Standard naming for observation fields in EmbodiedDataKit.

---

## Hierarchical Key Format

All observations use dot-separated keys:

```
observation.{category}.{name}
```

---

## Standard Keys

### Images
```
observation.images.front          # Front workspace camera
observation.images.wrist          # Wrist camera
observation.images.overhead       # Overhead camera
observation.images.left           # Left camera
observation.images.right          # Right camera
```

### Depth
```
observation.depth.front           # Front depth map
observation.depth.wrist           # Wrist depth
```

### State
```
observation.state                 # Proprioceptive state vector
observation.proprio.joints        # Joint positions/velocities
observation.proprio.ee_pose       # End-effector pose
observation.proprio.gripper       # Gripper state
```

### Language
```
observation.language              # Task instruction text
```

### Sensors
```
observation.force_torque          # Force/torque readings
observation.tactile               # Tactile sensor data
```

---

## Action Keys

```
action                            # Action vector (canonical)
action.ee_delta_7                 # 7D EE delta
action.joint_delta                # Joint delta
```
