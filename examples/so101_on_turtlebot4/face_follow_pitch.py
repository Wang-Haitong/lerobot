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
# limitations in the License.

"""Face-follow pitch: control SO-101 wrist_flex from a camera using OpenCV face detection.

Standalone usage (run on the machine with the arm and camera, e.g. Pi):
    python face_follow_pitch.py [--port /dev/ttyACM0] [--cam-index 0] [--headless]

To expose this as a primitive from the roman actions_ws executor, run the
arm_face_follow node on the Pi (see actions_ws docs).
"""

import argparse
import time

import cv2

from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.robots.utils import make_robot_from_config

CAM_INDEX_DEFAULT = 0
PITCH_KEY = "wrist_flex.pos"

Kp = 0.6
DEADBAND = 0.04
MAX_STEP = 0.015

PITCH_MIN = -0.6
PITCH_MAX = 0.6


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def run_face_follow_loop(robot, get_frame_fn, should_stop_fn, draw_fn=None):
    """Run the face-follow pitch control loop.

    get_frame_fn() -> (frame or None, timestamp)
    should_stop_fn() -> bool
    draw_fn(frame, faces, cx, cy, pitch_cmd, extra) -> None  (optional, for visualization)
    """
    obs = robot.get_observation()
    pitch_cmd = float(obs[PITCH_KEY])
    last_t = time.time()

    while not should_stop_fn():
        frame, _ = get_frame_fn()
        if frame is None:
            time.sleep(0.02)
            continue

        H, W = frame.shape[:2]
        y0 = H / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cx = x + w / 2.0
            cy = y + h / 2.0

            e = (cy - y0) / y0
            if abs(e) < DEADBAND:
                e = 0.0

            now = time.time()
            dt = max(1e-3, min(now - last_t, 0.1))
            last_t = now

            step = clamp(-Kp * e, -1.0, 1.0) * MAX_STEP
            pitch_cmd = clamp(pitch_cmd + step, PITCH_MIN, PITCH_MAX)

            robot.send_action({PITCH_KEY: float(pitch_cmd)})

            if draw_fn is not None:
                draw_fn(frame, faces, cx, cy, pitch_cmd, e)
        else:
            if draw_fn is not None:
                draw_fn(frame, [], None, None, pitch_cmd, None)

        if draw_fn is not None:
            cv2.line(frame, (0, int(y0)), (W, int(y0)), (255, 255, 0), 1)
        yield frame


def main_standalone(port: str, cam_index: int):
    config = SOFollowerRobotConfig(
        port=port,
        id="face_follow",
        cameras={},
    )
    robot = make_robot_from_config(config)
    robot.connect(calibrate=True)

    cap = cv2.VideoCapture(cam_index)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Could not load cascade at: {cascade_path}")

    def get_frame():
        ok, frame = cap.read()
        return (frame, time.time()) if ok else (None, time.time())

    stop = False

    def should_stop():
        return stop

    def draw(frame, faces, cx, cy, pitch_cmd, e):
        if faces is not None and len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if cx is not None and cy is not None:
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        err_str = f"e_y={e:+.3f}" if e is not None else "e_y=---"
        cv2.putText(
            frame,
            f"faces={len(faces)}  {err_str}  pitch={pitch_cmd:+.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    try:
        for frame in run_face_follow_loop(robot, get_frame, should_stop, draw_fn=draw):
            cv2.imshow("OpenCV Face + LeRobot Pitch", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                stop = True
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main_standalone_headless(port: str, cam_index: int):
    """Run face-follow without any GUI (for Pi / headless)."""
    config = SOFollowerRobotConfig(
        port=port,
        id="face_follow",
        cameras={},
    )
    robot = make_robot_from_config(config)
    robot.connect(calibrate=True)

    cap = cv2.VideoCapture(cam_index)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV face cascade")

    def get_frame():
        ok, frame = cap.read()
        return (frame, time.time()) if ok else (None, time.time())

    stop = False

    def should_stop():
        return stop

    try:
        for _ in run_face_follow_loop(robot, get_frame, should_stop, draw_fn=None):
            # No imshow; run until external interrupt (e.g. Ctrl+C)
            pass
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Face-follow pitch control for SO-101 wrist using OpenCV face detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port for SO-101 follower.",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=CAM_INDEX_DEFAULT,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No GUI window (for Pi / TurtleBot4 headless).",
    )
    args = parser.parse_args()
    if args.headless:
        main_standalone_headless(args.port, args.cam_index)
    else:
        main_standalone(args.port, args.cam_index)


if __name__ == "__main__":
    main()
