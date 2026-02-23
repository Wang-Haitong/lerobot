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

"""Data recording for distributed SO-101 + TurtleBot4 teleoperation.

Runs on the **Raspberry Pi** alongside the follower arm and TurtleBot4 base.
Follows the same data flow as ``lerobot_record.py``::

    robot.get_observation() -> obs        (arm motors + /odom)
    teleop.get_action()     -> action     (arm from leader + base from /cmd_vel)
    robot.arm.send_action(arm_action)     (only arm — base is driven by teleop_twist_keyboard)
    dataset.add_frame({obs, action, task})
    dataset.save_episode()

On the operator machine, run:
  - ``leader_teleop.py`` — publishes arm commands to ``/lerobot/arm_commands``
  - ``teleop_twist_keyboard`` — publishes base commands to ``/cmd_vel``

Requires:
    - ROS2 Jazzy sourced
    - Same ``ROS_DOMAIN_ID`` as the operator machine
    - Follower arm connected via USB
    - TurtleBot4 base operational (``/cmd_vel`` and ``/odom`` topics active)

Usage::

    # On the Pi (start this first, then start leader + teleop_twist_keyboard on operator):
    python record.py \\
        --repo_id user/so101_turtlebot4_pick_cube \\
        --single_task "Pick the cube and place it in the bin" \\
        --follower_port /dev/ttyACM0 \\
        --num_episodes 10

Episode controls (keyboard on Pi, requires display):
    Right arrow    End current episode early
    Left arrow     Re-record current episode
    ESC            Stop recording entirely
"""

import argparse
import logging
import time

import rclpy

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import RobotAction, RobotObservation
from lerobot.processor.factory import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say, move_cursor_up

from ros2_teleop import ROS2Teleoperator, ROS2TeleopConfig
from so101_turtlebot4_robot import SO101TurtleBot4Config, SO101TurtleBot4Robot

logger = logging.getLogger(__name__)

_ARM = "arm_"


def record_loop(
    robot: SO101TurtleBot4Robot,
    teleop: ROS2Teleoperator,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    dataset: LeRobotDataset | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
) -> None:
    """Run one episode (or reset phase) following the lerobot_record.py data flow."""
    if display_data:
        display_len = max(len(k) for k in robot.action_features)
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 1. Get robot observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # 2. Get action from remote leader via ROS2
        act = teleop.get_action()
        act_processed = teleop_action_processor((act, obs))

        # 3. Apply robot action processor and send arm only
        # Base is driven by teleop_twist_keyboard directly — we only relay arm commands.
        robot_action_to_send = robot_action_processor((act_processed, obs))
        arm_action = {k.removeprefix(_ARM): v for k, v in robot_action_to_send.items() if k.startswith(_ARM)}
        if arm_action:
            robot.arm.send_action(arm_action)

        # 4. Save to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            print("\n" + "-" * (display_len + 12))
            print(f"{'KEY':<{display_len}} | {'VALUE':>8}")
            for k, v in sorted(act_processed.items()):
                print(f"{k:<{display_len}} | {float(v):>8.3f}")
            move_cursor_up(len(act_processed) + 3)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = 1.0 / fps - dt_s
        if sleep_time_s < 0:
            logger.warning(
                f"Record loop running at {1 / dt_s:.1f} Hz (target {fps} Hz). "
                "Frames might be dropped."
            )
        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


def main():
    parser = argparse.ArgumentParser(
        description="Record teleoperation data for SO-101 + TurtleBot4 (runs on Pi).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset settings (matching lerobot DatasetRecordConfig defaults)
    parser.add_argument("--repo_id", required=True, help="Dataset identifier, e.g. 'user/dataset_name'.")
    parser.add_argument("--single_task", required=True, help="Short description of the task being recorded.")
    parser.add_argument("--root", default=None, help="Local root directory for dataset storage.")
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate.")
    parser.add_argument("--episode_time_s", type=float, default=60, help="Seconds per episode.")
    parser.add_argument("--reset_time_s", type=float, default=60, help="Seconds for environment reset between episodes.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to record.")
    parser.add_argument("--video", action="store_true", default=True, help="Encode frames as video.")
    parser.add_argument("--no_video", action="store_true", help="Store frames as images instead of video.")
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec (h264, hevc, libsvtav1).")
    parser.add_argument("--num_image_writer_processes", type=int, default=0, help="Subprocesses for image writing.")
    parser.add_argument("--num_image_writer_threads_per_camera", type=int, default=4, help="Threads per camera for image writing.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload dataset to HuggingFace Hub after recording.")
    parser.add_argument("--resume", action="store_true", help="Resume recording on an existing dataset.")
    parser.add_argument("--display_data", action="store_true", help="Print live action values to terminal.")
    parser.add_argument("--play_sounds", action="store_true", default=True, help="Audio feedback for episode events.")
    parser.add_argument("--no_play_sounds", action="store_true", help="Disable audio feedback.")
    # Follower arm settings
    parser.add_argument("--follower_port", default="/dev/ttyACM0", help="Serial port for SO-ARM 101 follower.")
    parser.add_argument("--follower_id", default="follower", help="Calibration ID for follower arm.")
    # TurtleBot4 base settings
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel", help="ROS2 topic for base velocity.")
    parser.add_argument("--odom_topic", default="/odom", help="ROS2 topic for odometry.")
    parser.add_argument("--max_linear_vel", type=float, default=0.5, help="Safety cap on linear velocity (m/s).")
    parser.add_argument("--max_angular_vel", type=float, default=1.0, help="Safety cap on angular velocity (rad/s).")
    # ROS2 teleop topics
    parser.add_argument("--arm_topic", default="/lerobot/arm_commands", help="ROS2 topic for arm commands from leader.")
    parser.add_argument("--cmd_vel_record_topic", default="/cmd_vel", help="ROS2 topic to capture base velocity commands for recording.")

    args = parser.parse_args()
    init_logging()

    use_video = args.video and not args.no_video
    play_sounds = args.play_sounds and not args.no_play_sounds

    # Initialize rclpy before creating robot (so TurtleBot4Robot doesn't own rclpy lifecycle).
    rclpy.init()

    robot = SO101TurtleBot4Robot(
        SO101TurtleBot4Config(
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
            base_topic=args.cmd_vel_record_topic,
        )
    )

    # Build dataset features following the standard lerobot pipeline
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_video,
        ),
    )

    dataset = None
    listener = None

    try:
        if args.resume:
            dataset = LeRobotDataset(
                args.repo_id,
                root=args.root,
                batch_encoding_size=1,
                vcodec=args.vcodec,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=args.num_image_writer_processes,
                    num_threads=args.num_image_writer_threads_per_camera * len(robot.cameras),
                )
        else:
            dataset = LeRobotDataset.create(
                args.repo_id,
                args.fps,
                root=args.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=use_video,
                image_writer_processes=args.num_image_writer_processes,
                image_writer_threads=args.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=1,
                vcodec=args.vcodec,
            )

        robot.connect()
        teleop.connect()

        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < args.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                record_loop(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    control_time_s=args.episode_time_s,
                    single_task=args.single_task,
                    display_data=args.display_data,
                )

                # Reset phase between episodes (no recording)
                if not events["stop_recording"] and (
                    (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", play_sounds)
                    record_loop(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=args.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        control_time_s=args.reset_time_s,
                        single_task=args.single_task,
                        display_data=args.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if args.push_to_hub and dataset:
            dataset.push_to_hub()

        if rclpy.ok():
            rclpy.shutdown()

        log_say("Exiting", play_sounds)


if __name__ == "__main__":
    main()
