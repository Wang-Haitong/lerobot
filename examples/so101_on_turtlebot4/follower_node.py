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

"""Follower-side node for distributed SO-101 + TurtleBot4 teleoperation.

Runs on the **Raspberry Pi** aboard the TurtleBot4.  Uses
:class:`ROS2Teleoperator` to receive arm commands from the remote leader
(``leader_teleop.py``) and :class:`SOFollower` to execute them.

Base control is handled separately by ``teleop_twist_keyboard`` publishing
directly to ``/cmd_vel`` — the TurtleBot4 acts on it natively.

For **data recording**, use ``record.py`` instead — it follows the same
pattern but also saves observations and actions to a ``LeRobotDataset``.

Safety features:
    - 500 ms watchdog (in ROS2Teleoperator): holds arm if no commands received.
    - Zero-velocity stop on disconnect.

Requires:
    - ROS2 Jazzy sourced
    - Same ``ROS_DOMAIN_ID`` as the operator machine
    - Follower arm connected via USB

Usage::

    python follower_node.py --follower_port /dev/ttyACM0
"""

import argparse
import logging
import time

import rclpy

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

from ros2_teleop import ROS2Teleoperator, ROS2TeleopConfig

logger = logging.getLogger(__name__)

_ARM = "arm_"


def follower_loop(
    teleop: ROS2Teleoperator,
    robot: SOFollower,
    fps: int,
    display_data: bool,
) -> None:
    """Teleoperation loop: get arm action from ROS2, send to follower arm."""
    display_len: int | None = None

    while rclpy.ok():
        loop_start = time.perf_counter()

        action = teleop.get_action()

        # Extract only arm keys (strip "arm_" prefix) for the follower
        arm_action = {k.removeprefix(_ARM): v for k, v in action.items() if k.startswith(_ARM)}
        if arm_action:
            robot.send_action(arm_action)

        if display_data and arm_action:
            if display_len is None:
                display_len = max(len(k) for k in arm_action)
            print("\n" + "-" * (display_len + 12))
            print(f"{'KEY':<{display_len}} | {'VALUE':>8}")
            for k, v in sorted(arm_action.items()):
                print(f"{k:<{display_len}} | {float(v):>8.3f}")
            move_cursor_up(len(arm_action) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1.0 / fps - dt_s, 0.0))


def main():
    parser = argparse.ArgumentParser(
        description="Follower-side node: receives arm commands from ROS2, controls SO-101 follower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Follower arm settings
    parser.add_argument("--follower_port", default="/dev/ttyACM0", help="Serial port for SO-ARM 101 follower.")
    parser.add_argument("--follower_id", default="follower", help="ID used to look up follower arm calibration.")
    # Control loop
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency in Hz.")
    parser.add_argument("--display_data", action="store_true", help="Print live action values to stdout.")
    # ROS2 topics
    parser.add_argument("--arm_topic", default="/lerobot/arm_commands", help="ROS2 topic for arm JointState commands.")
    # Watchdog
    parser.add_argument("--watchdog_timeout", type=float, default=0.5, help="Seconds before watchdog holds arm.")

    args = parser.parse_args()
    init_logging()

    # Initialize rclpy before creating components.
    rclpy.init()

    robot = SOFollower(
        SOFollowerRobotConfig(
            id=args.follower_id,
            port=args.follower_port,
        )
    )

    teleop = ROS2Teleoperator(
        ROS2TeleopConfig(
            arm_topic=args.arm_topic,
            watchdog_timeout_s=args.watchdog_timeout,
        )
    )

    try:
        robot.connect()
        teleop.connect()
        logger.info(
            f"Follower node running at {args.fps} Hz. "
            f"Listening for arm commands on '{args.arm_topic}'. "
            f"Base control via teleop_twist_keyboard on /cmd_vel. "
            f"Watchdog timeout: {args.watchdog_timeout * 1000:.0f} ms. "
            f"Press Ctrl+C to stop."
        )
        follower_loop(teleop, robot, fps=args.fps, display_data=args.display_data)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
