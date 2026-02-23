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

"""ROS2-based teleoperator that receives actions from remote sources.

Implements the LeRobot :class:`Teleoperator` interface.  Subscribes to:

- ``/lerobot/arm_commands`` (``JointState``) — arm joint targets from
  ``leader_teleop.py`` on the operator machine.
- ``/cmd_vel`` (``TwistStamped``) — base velocity commands from
  ``teleop_twist_keyboard`` (or any other source publishing to ``/cmd_vel``).

Action keys::

    arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos,
    arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos,
    base_linear.vel, base_angular.vel
"""

import logging
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)

_ARM = "arm_"
_BASE = "base_"


def _make_teleop_qos() -> QoSProfile:
    """QoS profile for low-latency teleoperation (must match the leader)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


@dataclass
class ROS2TeleopConfig:
    """Configuration for the ROS2 remote teleoperator."""

    id: str | None = "ros2_teleop"
    calibration_dir: Path | None = None

    # ROS2 topics
    arm_topic: str = "/lerobot/arm_commands"  # JointState from leader_teleop.py
    base_topic: str = "/cmd_vel"  # TwistStamped from teleop_twist_keyboard


class ROS2Teleoperator(Teleoperator):
    """Receives teleoperator actions from ROS2 topics published by the leader.

    This class bridges the network gap in the distributed teleoperation setup:
    the leader arm and keyboard are on the operator machine, and this teleoperator
    runs on the Pi, receiving their commands over ROS2.

    Usage::

        teleop = ROS2Teleoperator(ROS2TeleopConfig())
        teleop.connect()        # creates ROS2 node, starts spin thread
        action = teleop.get_action()  # returns latest commands from leader
        teleop.disconnect()
    """

    config_class = ROS2TeleopConfig
    name = "ros2_teleop"

    def __init__(self, config: ROS2TeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected: bool = False
        self._node: Node | None = None
        self._spin_thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._arm_cmd: dict[str, float] = {}
        self._base_cmd: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Teleoperator interface: features
    # ------------------------------------------------------------------

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            f"{_ARM}shoulder_pan.pos": float,
            f"{_ARM}shoulder_lift.pos": float,
            f"{_ARM}elbow_flex.pos": float,
            f"{_ARM}wrist_flex.pos": float,
            f"{_ARM}wrist_roll.pos": float,
            f"{_ARM}gripper.pos": float,
            f"{_BASE}linear.vel": float,
            f"{_BASE}angular.vel": float,
        }

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Teleoperator interface: connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        """Create a ROS2 subscriber node and start a background spin thread."""
        if not rclpy.ok():
            rclpy.init()

        qos = _make_teleop_qos()
        self._node = rclpy.create_node("lerobot_ros2_teleop")
        self._node.create_subscription(JointState, self.config.arm_topic, self._arm_cb, qos)
        self._node.create_subscription(TwistStamped, self.config.base_topic, self._base_cb, qos)

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True, name="ros2_teleop_spin"
        )
        self._spin_thread.start()
        self._connected = True
        logger.info(
            f"ROS2Teleoperator connected. Listening on '{self.config.arm_topic}' "
            f"and '{self.config.base_topic}'."
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        self._connected = False
        if self._node is not None:
            self._node.destroy_node()
        logger.info("ROS2Teleoperator disconnected.")

    # ------------------------------------------------------------------
    # Teleoperator interface: calibration (no-op)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # ROS2 callbacks (called from the spin thread)
    # ------------------------------------------------------------------

    def _arm_cb(self, msg: JointState) -> None:
        with self._lock:
            self._arm_cmd = dict(zip(msg.name, msg.position))

    def _base_cb(self, msg: TwistStamped) -> None:
        with self._lock:
            self._base_cmd = {
                "linear.vel": msg.twist.linear.x,
                "angular.vel": msg.twist.angular.z,
            }

    # ------------------------------------------------------------------
    # Teleoperator interface: action / feedback
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Return the latest commands from the remote leader."""
        with self._lock:
            arm_cmd = dict(self._arm_cmd)
            base_cmd = dict(self._base_cmd)

        action: dict[str, float] = {}
        for k, v in arm_cmd.items():
            action[f"{_ARM}{k}"] = float(v)
        action[f"{_BASE}linear.vel"] = float(base_cmd.get("linear.vel", 0.0))
        action[f"{_BASE}angular.vel"] = float(base_cmd.get("angular.vel", 0.0))
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No haptic feedback
