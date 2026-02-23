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

"""Composite robot: SO-ARM 101 follower arm + TurtleBot4 mobile base.

Follows the BiSOFollower composition pattern. Action and observation keys are
namespaced with "arm_" (6-DOF arm) and "base_" (mobile base velocities).

Action features:
    arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos,
    arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos,
    base_linear.vel, base_angular.vel

Observation features: same as action features (plus any arm cameras if configured).
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig, SOFollowerRobotConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from turtlebot4_robot import TurtleBot4Config, TurtleBot4Robot

logger = logging.getLogger(__name__)

_ARM = "arm_"
_BASE = "base_"


@dataclass
class SO101TurtleBot4Config:
    """Configuration for the composite SO-ARM 101 follower + TurtleBot4 robot.

    Arm calibration is looked up from the LeRobot default calibration directory
    using ``arm_id``. Run ``lerobot-teleoperate`` with ``--robot.id=<arm_id>``
    first to calibrate the arm if you haven't already.
    """

    # Identity used by Robot base class
    id: str | None = "so101_turtlebot4"
    calibration_dir: Path | None = None

    # Arm settings
    follower_port: str = "/dev/ttyACM0"
    arm_id: str = "follower"
    arm_max_relative_target: float | dict | None = None
    arm_use_degrees: bool = True

    # Cameras (passed through to SOFollower)
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
        "front": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
    })

    # TurtleBot4 base settings
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    max_linear_vel: float = 0.3   # m/s safety cap
    max_angular_vel: float = 1.0  # rad/s safety cap


class SO101TurtleBot4Robot(Robot):
    """Composite robot: SO-ARM 101 follower arm mounted on a TurtleBot4 base.

    Delegates arm control to :class:`lerobot.robots.so_follower.SOFollower`
    and base control to :class:`TurtleBot4Robot`. Action / observation keys
    are prefixed with ``"arm_"`` or ``"base_"`` following the BiSOFollower
    composition pattern.
    """

    config_class = SO101TurtleBot4Config
    name = "so101_turtlebot4"

    def __init__(self, config: SO101TurtleBot4Config):
        super().__init__(config)
        self.config = config

        arm_config = SOFollowerRobotConfig(
            id=config.arm_id,
            calibration_dir=config.calibration_dir,
            port=config.follower_port,
            max_relative_target=config.arm_max_relative_target,
            use_degrees=config.arm_use_degrees,
            cameras=config.cameras,
        )
        base_config = TurtleBot4Config(
            id=f"{config.id}_base" if config.id else "turtlebot4_base",
            calibration_dir=config.calibration_dir,
            cmd_vel_topic=config.cmd_vel_topic,
            odom_topic=config.odom_topic,
            max_linear_vel=config.max_linear_vel,
            max_angular_vel=config.max_angular_vel,
        )

        self.arm = SOFollower(arm_config)
        self.base = TurtleBot4Robot(base_config)

        # Expose arm cameras so other parts of the codebase can access them.
        self.cameras = {**self.arm.cameras}

    # ------------------------------------------------------------------
    # Robot interface: features
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        arm_ft = {f"{_ARM}{k}": v for k, v in self.arm.observation_features.items()}
        base_ft = {f"{_BASE}{k}": v for k, v in self.base.observation_features.items()}
        return {**arm_ft, **base_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        arm_ft = {f"{_ARM}{k}": v for k, v in self.arm.action_features.items()}
        base_ft = {f"{_BASE}{k}": v for k, v in self.base.action_features.items()}
        return {**arm_ft, **base_ft}

    # ------------------------------------------------------------------
    # Robot interface: connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self.base.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.arm.connect(calibrate)
        self.base.connect(calibrate=False)  # TurtleBot4 needs no calibration

    @check_if_not_connected
    def disconnect(self) -> None:
        self.arm.disconnect()
        self.base.disconnect()

    # ------------------------------------------------------------------
    # Robot interface: calibration
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated and self.base.is_calibrated

    def calibrate(self) -> None:
        self.arm.calibrate()

    def configure(self) -> None:
        self.arm.configure()
        self.base.configure()

    # ------------------------------------------------------------------
    # Robot interface: observation / action
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        arm_obs = self.arm.get_observation()
        base_obs = self.base.get_observation()
        return {
            **{f"{_ARM}{k}": v for k, v in arm_obs.items()},
            **{f"{_BASE}{k}": v for k, v in base_obs.items()},
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        arm_action = {k.removeprefix(_ARM): v for k, v in action.items() if k.startswith(_ARM)}
        base_action = {k.removeprefix(_BASE): v for k, v in action.items() if k.startswith(_BASE)}

        sent: dict[str, float] = {}
        if arm_action:
            sent_arm = self.arm.send_action(arm_action)
            sent.update({f"{_ARM}{k}": v for k, v in sent_arm.items()})
        if base_action:
            sent_base = self.base.send_action(base_action)
            sent.update({f"{_BASE}{k}": v for k, v in sent_base.items()})
        return sent
