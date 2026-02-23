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

"""Composite teleoperator: SO-ARM 101 leader arm + keyboard for TurtleBot4 base.

Reads from two input devices simultaneously and merges their actions under
namespaced keys following the BiSOLeader composition pattern:

  - ``arm_*``  — position targets from the SO-ARM 101 leader arm (6 DOF)
  - ``base_*`` — velocity commands from the keyboard (WASD-style)

Keyboard controls (base):
    W / S   Forward / backward
    A / D   Turn left / right (with slight forward assist)
    Q / E   Rotate in place
    X       Emergency stop
    + / -   Increase / decrease speed
    ESC     Disconnect
"""

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.keyboard import KeyboardRoverTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardRoverTeleopConfig
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)

_ARM = "arm_"
_BASE = "base_"


@dataclass
class SO101KeyboardTeleopConfig:
    """Configuration for the composite SO-ARM 101 leader + keyboard teleoperator."""

    # Identity used by Teleoperator base class
    id: str | None = "so101_keyboard"
    calibration_dir: Path | None = None

    # Leader arm settings
    leader_port: str = "/dev/ttyACM1"
    leader_id: str = "leader"
    leader_use_degrees: bool = True

    # Keyboard rover speed settings
    keyboard_linear_speed: float = 0.2   # m/s initial speed
    keyboard_angular_speed: float = 0.5  # rad/s initial speed
    keyboard_speed_increment: float = 0.05


class SO101KeyboardTeleop(Teleoperator):
    """Composite teleoperator: SO-ARM 101 leader (arm) + keyboard (base).

    Combines :class:`lerobot.teleoperators.so_leader.SOLeader` and
    :class:`lerobot.teleoperators.keyboard.KeyboardRoverTeleop` into a single
    teleoperator whose :meth:`get_action` returns a merged dict with ``arm_``
    and ``base_`` prefixed keys.

    This is designed to be paired with :class:`SO101TurtleBot4Robot`.
    """

    config_class = SO101KeyboardTeleopConfig
    name = "so101_keyboard"

    def __init__(self, config: SO101KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config

        leader_config = SOLeaderTeleopConfig(
            id=config.leader_id,
            calibration_dir=config.calibration_dir,
            port=config.leader_port,
            use_degrees=config.leader_use_degrees,
        )
        keyboard_config = KeyboardRoverTeleopConfig(
            id=f"{config.id}_keyboard" if config.id else None,
            linear_speed=config.keyboard_linear_speed,
            angular_speed=config.keyboard_angular_speed,
            speed_increment=config.keyboard_speed_increment,
        )

        self.leader = SOLeader(leader_config)
        self.keyboard = KeyboardRoverTeleop(keyboard_config)

    # ------------------------------------------------------------------
    # Teleoperator interface: features
    # ------------------------------------------------------------------

    @cached_property
    def action_features(self) -> dict[str, type]:
        arm_ft = {f"{_ARM}{k}": v for k, v in self.leader.action_features.items()}
        base_ft = {f"{_BASE}{k}": v for k, v in self.keyboard.action_features.items()}
        return {**arm_ft, **base_ft}

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Teleoperator interface: connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected and self.keyboard.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.leader.connect(calibrate)
        self.keyboard.connect()

    @check_if_not_connected
    def disconnect(self) -> None:
        self.leader.disconnect()
        self.keyboard.disconnect()

    # ------------------------------------------------------------------
    # Teleoperator interface: calibration
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated and self.keyboard.is_calibrated

    def calibrate(self) -> None:
        self.leader.calibrate()

    def configure(self) -> None:
        self.leader.configure()
        self.keyboard.configure()

    # ------------------------------------------------------------------
    # Teleoperator interface: action / feedback
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        arm_action = self.leader.get_action()
        base_action = self.keyboard.get_action()
        return {
            **{f"{_ARM}{k}": v for k, v in arm_action.items()},
            **{f"{_BASE}{k}": v for k, v in base_action.items()},
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No haptic feedback implemented
