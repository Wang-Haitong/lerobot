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

"""Leader-side teleoperation node for distributed SO-101 + TurtleBot4.

Runs on the **operator's machine** (server or laptop) where the SO-ARM 101
leader arm is physically connected via USB.  Reads joint positions from the
leader arm and publishes them as ROS2 ``JointState`` messages for the remote
follower node on the Pi.

Base control is handled separately by the standard ``teleop_twist_keyboard``
node — run it on the same machine::

    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true

Requires:
    - ROS2 Jazzy sourced
    - Same ``ROS_DOMAIN_ID`` as the Pi (e.g. ``export ROS_DOMAIN_ID=42``)
    - Leader arm connected via USB

Usage::

    python leader_teleop.py --leader_port /dev/ttyACM0
"""

import argparse
import logging
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

logger = logging.getLogger(__name__)


def _make_teleop_qos() -> QoSProfile:
    """QoS profile tuned for low-latency teleoperation.

    - BEST_EFFORT: prefer low latency over guaranteed delivery (ok to drop)
    - VOLATILE: late-joining subscribers should not replay old commands
    - KEEP_LAST(1): only the newest command matters
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class LeaderTeleopNode(Node):
    """ROS2 node that publishes leader arm joint commands."""

    def __init__(self, teleop: SOLeader, arm_topic: str):
        super().__init__("lerobot_leader_teleop")
        qos = _make_teleop_qos()
        self._arm_pub = self.create_publisher(JointState, arm_topic, qos)
        self._teleop = teleop

    def publish_once(self) -> dict[str, float]:
        """Read leader arm, publish JointState. Returns raw action dict."""
        action = self._teleop.get_action()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(action.keys())
        msg.position = [float(v) for v in action.values()]
        self._arm_pub.publish(msg)

        return action


def leader_loop(node: LeaderTeleopNode, fps: int, display_data: bool) -> None:
    """Run the leader publish loop at a fixed frequency."""
    display_len: int | None = None

    while rclpy.ok():
        loop_start = time.perf_counter()
        action = node.publish_once()

        if display_data:
            if display_len is None:
                display_len = max(len(k) for k in action)
            print("\n" + "-" * (display_len + 12))
            print(f"{'KEY':<{display_len}} | {'VALUE':>8}")
            for k, v in sorted(action.items()):
                print(f"{k:<{display_len}} | {float(v):>8.3f}")
            move_cursor_up(len(action) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1.0 / fps - dt_s, 0.0))


def main():
    parser = argparse.ArgumentParser(
        description="Leader-side teleoperation: reads SO-101 leader arm, publishes to ROS2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--leader_port", default="/dev/ttyACM0", help="Serial port for SO-ARM 101 leader.")
    parser.add_argument("--leader_id", default="leader", help="ID used to look up leader arm calibration.")
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency in Hz.")
    parser.add_argument("--display_data", action="store_true", help="Print live action values to stdout.")
    parser.add_argument("--arm_topic", default="/lerobot/arm_commands", help="ROS2 topic for arm JointState commands.")

    args = parser.parse_args()
    init_logging()

    teleop = SOLeader(
        SOLeaderTeleopConfig(
            id=args.leader_id,
            port=args.leader_port,
        )
    )

    rclpy.init()
    node = LeaderTeleopNode(teleop=teleop, arm_topic=args.arm_topic)

    try:
        teleop.connect()
        logger.info(
            f"Leader teleop running at {args.fps} Hz. "
            f"Publishing arm to '{args.arm_topic}'. "
            f"Use teleop_twist_keyboard for base control. "
            f"Press Ctrl+C to stop."
        )
        leader_loop(node, fps=args.fps, display_data=args.display_data)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        teleop.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
