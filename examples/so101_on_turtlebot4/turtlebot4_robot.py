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

"""TurtleBot4 mobile base integration for LeRobot via ROS2 Jazzy.

Wraps the TurtleBot4 Create3 base as a LeRobot Robot:
  - Publishes velocity commands to /cmd_vel (geometry_msgs/TwistStamped)
  - Subscribes to /odom (nav_msgs/Odometry) for feedback
  - Spins rclpy in a background daemon thread so the control loop stays unblocked

Action / observation keys:
  linear.vel  (float, m/s)
  angular.vel (float, rad/s)
"""

import logging
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

# ROS2 Jazzy imports — only available on the TurtleBot4 or a machine with ROS2 sourced.
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)


@dataclass
class TurtleBot4Config:
    """Configuration for TurtleBot4 base control via ROS2 Jazzy."""

    # Robot identity (used by Robot base class for calibration directory naming)
    id: str | None = "turtlebot4"
    calibration_dir: Path | None = None

    # ROS2 topic names
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"

    # Safety velocity caps applied before publishing
    max_linear_vel: float = 0.3   # m/s
    max_angular_vel: float = 1.0  # rad/s


class TurtleBot4Robot(Robot):
    """LeRobot-compatible wrapper for the TurtleBot4 mobile base.

    Communicates with the Create3 base over ROS2 Jazzy. A daemon background thread
    runs the rclpy spin loop so odometry callbacks are processed without blocking
    the main teleoperation control loop.

    Usage::

        config = TurtleBot4Config(cmd_vel_topic="/cmd_vel")
        with TurtleBot4Robot(config) as robot:
            robot.connect()
            obs = robot.get_observation()
            robot.send_action({"linear.vel": 0.1, "angular.vel": 0.0})
    """

    config_class = TurtleBot4Config
    name = "turtlebot4"

    def __init__(self, config: TurtleBot4Config):
        super().__init__(config)
        self.config = config
        self._connected: bool = False
        self._owns_rclpy: bool = False
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._last_linear_vel: float = 0.0
        self._last_angular_vel: float = 0.0

    # ------------------------------------------------------------------
    # Robot interface: features
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    # ------------------------------------------------------------------
    # Robot interface: connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        """Initialize rclpy, create the ROS2 node, and start background spin."""
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self._node = rclpy.create_node("lerobot_turtlebot4")
        self._cmd_vel_pub = self._node.create_publisher(TwistStamped, self.config.cmd_vel_topic, 10)
        # Create3 publishes /odom with SensorDataQoS (BEST_EFFORT).
        # Using the matching profile avoids a silent QoS mismatch.
        self._odom_sub = self._node.create_subscription(
            Odometry,
            self.config.odom_topic,
            self._odom_callback,
            qos_profile_sensor_data,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name="turtlebot4_spin",
        )
        self._spin_thread.start()
        self._connected = True
        logger.info(
            f"TurtleBot4 connected. Publishing to '{self.config.cmd_vel_topic}', "
            f"listening on '{self.config.odom_topic}'."
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        """Publish a zero-velocity command (safety stop), then shut down the node."""
        self._cmd_vel_pub.publish(TwistStamped())  # stop the base
        self._connected = False
        if self._executor is not None:
            self._executor.shutdown()
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        logger.info("TurtleBot4 disconnected.")

    # ------------------------------------------------------------------
    # Robot interface: calibration (no-op for TurtleBot4)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True  # TurtleBot4 base requires no LeRobot calibration

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # ROS2 callback
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry) -> None:
        self._last_linear_vel = msg.twist.twist.linear.x
        self._last_angular_vel = msg.twist.twist.angular.z

    # ------------------------------------------------------------------
    # Robot interface: observation / action
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        return {
            "linear.vel": self._last_linear_vel,
            "angular.vel": self._last_angular_vel,
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        linear_vel = float(action.get("linear.vel", 0.0))
        angular_vel = float(action.get("angular.vel", 0.0))

        # Apply safety caps
        linear_vel = max(-self.config.max_linear_vel, min(self.config.max_linear_vel, linear_vel))
        angular_vel = max(-self.config.max_angular_vel, min(self.config.max_angular_vel, angular_vel))

        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.twist.linear.x = linear_vel
        msg.twist.angular.z = angular_vel
        self._cmd_vel_pub.publish(msg)

        return {"linear.vel": linear_vel, "angular.vel": angular_vel}
