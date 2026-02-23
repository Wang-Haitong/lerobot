# SO-ARM 101 on TurtleBot4

Teleoperation and data recording for a SO-ARM 101 mounted on a TurtleBot4 mobile base.

## Files

| File | Description |
|------|-------------|
| `teleoperate.py` | Teleoperation entry point (arm-only, arm+base single-machine, or distributed). |
| `leader_teleop.py` | Leader-side node for distributed teleoperation — publishes arm commands over ROS2 (runs on server/laptop). |
| `follower_node.py` | Follower-side node for distributed teleoperation without recording (runs on Pi). |
| `record.py` | Data recording for distributed teleoperation (runs on Pi, saves to LeRobotDataset). |
| `ros2_teleop.py` | ROS2-based teleoperator — receives arm commands from leader and base commands from `/cmd_vel`. |
| `so101_keyboard_teleop.py` | Composite teleoperator combining SO-101 leader arm + keyboard (used by single-machine mode). |
| `so101_turtlebot4_robot.py` | Composite robot combining SO-101 follower arm + TurtleBot4 base. |
| `turtlebot4_robot.py` | TurtleBot4 base wrapper (publishes `/cmd_vel`, subscribes `/odom`). |

## Prerequisites

- [lerobot](https://github.com/huggingface/lerobot) installed
- ROS2 Jazzy sourced (required for base control, not needed for arm-only mode)
- For distributed mode: both machines on the same network with the same `ROS_DOMAIN_ID`

## Usage

### Arm-only (single machine, no ROS2)

Both leader and follower arms connected to the same machine via USB.

```bash
python teleoperate.py --mode arm_only \
    --follower_port /dev/ttyACM0 \
    --leader_port /dev/ttyACM1
```

### Distributed arm + base teleoperation (two machines)

Leader arm on the operator's laptop, follower arm + TurtleBot4 on the Pi.
Base is controlled via the standard `teleop_twist_keyboard` node.

**Option A — via `teleoperate.py` (recommended):**

```bash
# On the Pi (only follower_port needed — no leader arm here):
python teleoperate.py --mode distributed \
    --follower_port /dev/ttyACM0

# On the operator's laptop (two terminals):
# Terminal 1 — arm teleoperation:
python leader_teleop.py --leader_port /dev/ttyACM0

# Terminal 2 — base teleoperation:
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

**Option B — via `follower_node.py` directly:**

```bash
# On the Pi:
python follower_node.py --follower_port /dev/ttyACM0

# On the operator's laptop (two terminals):
python leader_teleop.py --leader_port /dev/ttyACM0
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

### Distributed arm + base recording (two machines)

Same setup as above, but saves data to a LeRobotDataset on the Pi.

```bash
# On the Pi:
python record.py \
    --repo_id user/so101_turtlebot4_pick_cube \
    --single_task "Pick the cube and place it in the bin" \
    --follower_port /dev/ttyACM0 \
    --num_episodes 10

# On the operator's machine (two terminals, same as teleoperation):
python leader_teleop.py --leader_port /dev/ttyACM0
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

To add cameras (e.g. a wrist camera on the arm), pass them to `record.py` or
`follower_node.py` via `SO101TurtleBot4Config.cameras` in code, or extend the
argparse to accept camera configs.

Episode controls (keyboard on Pi, requires display):

| Key | Action |
|-----|--------|
| Right arrow | End current episode early |
| Left arrow | Re-record current episode |
| ESC | Stop recording |

### Single-machine arm + base (testing)

Both arms on the same machine. Useful for bench testing before deploying to two machines.

```bash
python teleoperate.py --mode arm_base \
    --follower_port /dev/ttyACM0 \
    --leader_port /dev/ttyACM1
```

## What Gets Recorded

Each frame saved to the dataset contains:

| Dataset key | Content | Source |
|-------------|---------|--------|
| `observation.state` | 8 floats: 6 arm joint positions (degrees) + 2 base velocities (m/s, rad/s) | Follower arm motors + `/odom` |
| `observation.images.<name>` | Camera frames (if cameras configured) | USB cameras on Pi |
| `action` | 8 floats: 6 arm joint targets + 2 base velocity commands | `/lerobot/arm_commands` + `/cmd_vel` |
| `task` | Task description string | User-provided |
