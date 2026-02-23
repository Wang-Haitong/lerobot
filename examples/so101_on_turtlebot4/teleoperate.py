#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperation script for SO-ARM 101 mounted on a TurtleBot4.

Three modes:

  arm_only     — SO-ARM 101 leader → SO-ARM 101 follower only.
                 No ROS2 required. Useful for arm-only data collection on any machine.
                 Equivalent to the built-in lerobot-teleoperate CLI.

  arm_base     — SO-ARM 101 leader  → SO-ARM 101 follower arm (joint positions)
                 Keyboard (WASD)    → TurtleBot4 base (velocity, via ROS2 Jazzy)
                 Requires ROS2 Jazzy to be sourced. Both leader and follower on same machine.

  distributed  — Follower arm + TurtleBot4 base on the robot (Pi).
                 Leader arm on the laptop (runs leader_teleop.py separately).
                 Arm commands arrive via ROS2 JointState; base via /cmd_vel.
                 Only the follower_port is needed on the robot side.

Usage::

    # Arm only (works on any machine with both arm USB ports):
    python teleoperate.py --mode arm_only \\
        --follower_port /dev/ttyACM0 \\
        --leader_port   /dev/ttyACM1

    # Arm + base, same machine (run on TurtleBot4 or any ROS2-networked machine):
    python teleoperate.py --mode arm_base \\
        --follower_port /dev/ttyACM0 \\
        --leader_port   /dev/ttyACM1

    # Distributed: robot side (run on TurtleBot4 / Pi):
    python teleoperate.py --mode distributed \\
        --follower_port /dev/ttyACM0

    # Distributed: laptop side (run on operator laptop):
    python leader_teleop.py --leader_port /dev/ttyACM0

Keyboard controls (arm_base mode — base only):
    W / S     Forward / backward
    A / D     Turn left / right (with slight forward assist)
    Q / E     Rotate in place
    X         Emergency stop (base)
    + / -     Increase / decrease speed
    ESC       Quit
"""

import argparse
import logging
import sys
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

logger = logging.getLogger(__name__)


def teleop_loop(teleop, robot, fps: int, display_data: bool) -> None:
    """Run the teleoperation control loop at a fixed frequency.

    Args:
        teleop: Connected teleoperator (arm leader, or arm leader + keyboard).
        robot:  Connected robot (arm follower, or arm follower + TurtleBot4 base).
        fps:    Target control loop frequency in Hz.
        display_data: If True, print live action values to stdout.
    """
    display_len = max(len(k) for k in robot.action_features)

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()  # noqa: F841 (used implicitly; kept for future processing)
        action = teleop.get_action()
        robot.send_action(action)

        if display_data:
            print("\n" + "-" * (display_len + 12))
            print(f"{'KEY':<{display_len}} | {'VALUE':>8}")
            for k, v in sorted(action.items()):
                print(f"{k:<{display_len}} | {float(v):>8.3f}")
            move_cursor_up(len(action) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))


def build_arm_only(args):
    """Instantiate arm-only robot and teleoperator (no ROS2 needed)."""
    robot = SOFollower(
        SOFollowerRobotConfig(
            id=args.follower_id,
            port=args.follower_port,
        )
    )
    teleop = SOLeader(
        SOLeaderTeleopConfig(
            id=args.leader_id,
            port=args.leader_port,
        )
    )
    return robot, teleop


def build_arm_base(args):
    """Instantiate composite arm+base robot and teleoperator (requires ROS2 Jazzy)."""
    # Import here so arm_only mode works without ROS2 installed.
    from so101_keyboard_teleop import SO101KeyboardTeleop, SO101KeyboardTeleopConfig
    from so101_turtlebot4_robot import SO101TurtleBot4Config, SO101TurtleBot4Robot

    robot = SO101TurtleBot4Robot(
        SO101TurtleBot4Config(
            id=args.follower_id,
            follower_port=args.follower_port,
            arm_id=args.follower_id,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            max_linear_vel=args.max_linear_vel,
            max_angular_vel=args.max_angular_vel,
        )
    )
    teleop = SO101KeyboardTeleop(
        SO101KeyboardTeleopConfig(
            id=args.leader_id,
            leader_port=args.leader_port,
            leader_id=args.leader_id,
            keyboard_linear_speed=args.keyboard_linear_speed,
            keyboard_angular_speed=args.keyboard_angular_speed,
        )
    )
    return robot, teleop


def build_distributed(args):
    """Instantiate robot-side components for distributed teleoperation.

    The follower arm + TurtleBot4 base run here (on the robot / Pi).
    The leader arm runs on the laptop via leader_teleop.py, sending arm
    commands over ROS2. Base commands come from teleop_twist_keyboard on /cmd_vel.
    No leader_port is needed on this side.
    """
    from so101_turtlebot4_robot import SO101TurtleBot4Config, SO101TurtleBot4Robot

    from ros2_teleop import ROS2Teleoperator, ROS2TeleopConfig

    robot = SO101TurtleBot4Robot(
        SO101TurtleBot4Config(
            id=args.follower_id,
            follower_port=args.follower_port,
            arm_id=args.follower_id,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            max_linear_vel=args.max_linear_vel,
            max_angular_vel=args.max_angular_vel,
        )
    )
    teleop = ROS2Teleoperator(
        ROS2TeleopConfig(
            arm_topic=args.arm_topic,
        )
    )
    return robot, teleop


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM 101 + TurtleBot4 teleoperation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        default="arm_base",
        choices=["arm_only", "arm_base", "distributed"],
        help=(
            "arm_only: arm leader→follower only (no ROS2). "
            "arm_base: arm+keyboard→arm+TurtleBot4, same machine (ROS2 required). "
            "distributed: follower+base on robot, leader on laptop via ROS2."
        ),
    )
    # Hardware ports
    parser.add_argument("--follower_port", default="/dev/ttyACM0", help="Serial port for SO-ARM 101 follower.")
    parser.add_argument("--leader_port", default="/dev/ttyACM1", help="Serial port for SO-ARM 101 leader.")
    parser.add_argument("--follower_id", default="follower", help="ID used to look up follower arm calibration.")
    parser.add_argument("--leader_id", default="leader", help="ID used to look up leader arm calibration.")
    # Control loop
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency in Hz.")
    parser.add_argument("--display_data", action="store_true", help="Print live action values to stdout.")
    # TurtleBot4 settings (arm_base mode only)
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel", help="ROS2 topic for base velocity commands.")
    parser.add_argument("--odom_topic", default="/odom", help="ROS2 topic for odometry feedback.")
    parser.add_argument("--max_linear_vel", type=float, default=0.3, help="Safety cap on linear velocity (m/s).")
    parser.add_argument("--max_angular_vel", type=float, default=1.0, help="Safety cap on angular velocity (rad/s).")
    # Keyboard speed settings
    parser.add_argument("--keyboard_linear_speed", type=float, default=0.2, help="Initial keyboard linear speed (m/s).")
    parser.add_argument("--keyboard_angular_speed", type=float, default=0.5, help="Initial keyboard angular speed (rad/s).")
    # Distributed mode settings
    parser.add_argument("--arm_topic", default="/lerobot/arm_commands", help="ROS2 topic for arm JointState commands (distributed mode).")

    args = parser.parse_args()
    init_logging()

    if args.mode == "arm_only":
        logger.info("Mode: arm_only — no ROS2 required.")
        robot, teleop = build_arm_only(args)
    elif args.mode == "arm_base":
        logger.info("Mode: arm_base — requires ROS2 Jazzy. Use WASD to drive, leader arm to control the arm.")
        robot, teleop = build_arm_base(args)
    elif args.mode == "distributed":
        logger.info(
            "Mode: distributed — follower + base on this machine. "
            "Run leader_teleop.py on the laptop for arm control, "
            "teleop_twist_keyboard for base control."
        )
        robot, teleop = build_distributed(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Distributed mode needs rclpy initialized before connecting.
    if args.mode == "distributed":
        import rclpy
        rclpy.init()

    try:
        teleop.connect()
        robot.connect()
        logger.info(f"Teleop loop running at {args.fps} Hz. Press Ctrl+C to stop.")
        teleop_loop(teleop, robot, fps=args.fps, display_data=args.display_data)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        robot.disconnect()
        teleop.disconnect()
        if args.mode == "distributed":
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
