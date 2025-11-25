from __future__ import annotations

import cv2
from typing import List


def read_video_frames(path: str) -> List:
    """
    Read all frames from a video file into memory.
    """
    cap = cv2.VideoCapture(path)
    frames = []

    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames


def write_video_frames(frames, output_path: str, fps: float = 24.0) -> None:
    """
    Write a list of frames to a video file using MJPG encoding.
    """
    if not frames:
        raise ValueError("No frames provided to write_video_frames.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()