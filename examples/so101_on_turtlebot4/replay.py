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

"""Replay recorded trajectories for SO-101 + TurtleBot4 on the robot side.

This script follows the official LeRobot replay flow:

    dataset action -> robot action processor -> robot.send_action()

Expected deployment is the same as distributed recording:
    - Run this script on the TurtleBot4 / Pi
    - Follower arm is connected to the Pi over USB
    - Base commands are sent directly to /cmd_vel through TurtleBot4Robot
"""

import argparse
import logging
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from so101_turtlebot4_robot import SO101TurtleBot4Config, SO101TurtleBot4Robot

logger = logging.getLogger(__name__)
_ARM = "arm_"
_BASE = "base_"


def _build_replay_action(action, action_names: list[str]) -> dict[str, float]:
    replay_action: dict[str, float] = {}
    for idx, name in enumerate(action_names):
        replay_action[name] = float(action[idx])
    return replay_action


def _normalize_action_keys(
    raw_action: dict[str, float],
    expected_keys: set[str],
    arm_unprefixed_keys: set[str],
    base_unprefixed_keys: set[str],
) -> dict[str, float]:
    normalized: dict[str, float] = {}

    for key, value in raw_action.items():
        candidates = [key]

        key_no_main = key.removeprefix("main_")
        if key_no_main != key:
            candidates.append(key_no_main)

        for k in (key, key_no_main):
            if not k.startswith((_ARM, _BASE)):
                if k in arm_unprefixed_keys:
                    candidates.append(f"{_ARM}{k}")
                if k in base_unprefixed_keys:
                    candidates.append(f"{_BASE}{k}")

        for candidate in candidates:
            if candidate in expected_keys:
                normalized[candidate] = float(value)
                break

    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded SO-101 + TurtleBot4 trajectory (robot-side).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", required=True, help="Dataset repo id, e.g. 'user/so101_turtlebot4_pick_cube'.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to replay.")
    parser.add_argument("--root", default=None, help="Optional local dataset root.")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Replay FPS cap. Defaults to dataset FPS. Values above dataset FPS do not speed up playback.",
    )

    parser.add_argument("--follower_port", default="/dev/ttyACM0", help="Serial port for SO-ARM 101 follower.")
    parser.add_argument("--follower_id", default="follower", help="Calibration ID for the follower arm.")
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel", help="ROS2 topic for base velocity commands.")
    parser.add_argument("--odom_topic", default="/odom", help="ROS2 topic for odometry feedback.")
    parser.add_argument("--max_linear_vel", type=float, default=0.5, help="Safety cap on linear velocity (m/s).")
    parser.add_argument("--max_angular_vel", type=float, default=1.0, help="Safety cap on angular velocity (rad/s).")

    parser.add_argument(
        "--enable_base",
        action="store_true",
        help="Replay base velocity actions in addition to arm actions.",
    )
    parser.add_argument("--display_data", action="store_true", help="Print live outgoing action values.")
    parser.add_argument("--play_sounds", action="store_true", default=True, help="Enable event audio.")
    parser.add_argument("--no_play_sounds", action="store_true", help="Disable event audio.")

    args = parser.parse_args()
    init_logging()

    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0 when provided.")

    play_sounds = args.play_sounds and not args.no_play_sounds

    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python packages are unavailable. Source ROS2 Jazzy before running replay.py."
        ) from exc

    rclpy.init()

    robot = SO101TurtleBot4Robot(
        SO101TurtleBot4Config(
            follower_port=args.follower_port,
            arm_id=args.follower_id,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            max_linear_vel=args.max_linear_vel,
            max_angular_vel=args.max_angular_vel,
            cameras={},  # No cameras needed for replay
        )
    )

    robot_action_processor = make_default_robot_action_processor()
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])

    # Episode-aware filtering is required for v3 datasets where frames are chunked.
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == args.episode)
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {args.episode} not found in dataset '{args.repo_id}'.")

    actions = episode_frames.select_columns(ACTION)
    action_names = list(dataset.features[ACTION]["names"])
    expected_keys = set(robot.action_features)
    arm_unprefixed_keys = {k.removeprefix(_ARM) for k in expected_keys if k.startswith(_ARM)}
    base_unprefixed_keys = {k.removeprefix(_BASE) for k in expected_keys if k.startswith(_BASE)}

    replay_fps = dataset.fps
    if args.fps is not None:
        replay_fps = min(args.fps, dataset.fps)
        if args.fps > dataset.fps:
            logger.info(
                "Requested fps (%s) is above dataset fps (%s). Using dataset fps.",
                args.fps,
                dataset.fps,
            )

    logger.info(
        "Loaded episode %s with %s frames. Replaying at %.2f Hz (%s).",
        args.episode,
        len(episode_frames),
        replay_fps,
        "arm + base" if args.enable_base else "arm only",
    )
    logger.info("Dataset action keys: %s", ", ".join(action_names))

    robot.connect()

    try:
        log_say(f"Replaying episode {args.episode}", play_sounds, blocking=True)
        arm_key_found = False
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            raw_action = _build_replay_action(actions[idx][ACTION], action_names)
            normalized_raw_action = _normalize_action_keys(
                raw_action,
                expected_keys=expected_keys,
                arm_unprefixed_keys=arm_unprefixed_keys,
                base_unprefixed_keys=base_unprefixed_keys,
            )

            if not normalized_raw_action and idx == 0:
                raise RuntimeError(
                    "No replay action keys matched robot action features. "
                    f"Dataset keys={sorted(raw_action.keys())}, expected keys={sorted(expected_keys)}."
                )

            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((normalized_raw_action, robot_obs))

            if args.enable_base:
                action_to_send = processed_action
            else:
                action_to_send = {
                    key: value for key, value in processed_action.items() if not key.startswith(_BASE)
                }

            if any(key.startswith(_ARM) for key in action_to_send):
                arm_key_found = True

            if args.display_data and idx % 5 == 0:
                logger.info("Frame %s action: %s", idx, action_to_send)

            _ = robot.send_action(action_to_send)

            precise_sleep(max(1.0 / replay_fps - (time.perf_counter() - t0), 0.0))

        if not arm_key_found:
            raise RuntimeError(
                "Replay finished without any arm commands being sent. "
                "Check dataset action schema and replay logs."
            )
    finally:
        if robot.is_connected:
            # Always issue an explicit base stop for safety at replay end/interruption.
            try:
                _ = robot.send_action({"base_linear.vel": 0.0, "base_angular.vel": 0.0})
            except Exception:
                logger.exception("Failed to send base stop command during replay shutdown.")
            robot.disconnect()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
