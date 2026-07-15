#!/usr/bin/env python
"""Extract timestamped frames from the configured input video."""

from __future__ import annotations

import time

import cv2
from tqdm import tqdm

from config import CONFIG


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH-MM-SS for existing frame naming."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    return f"{hours:02d}-{minutes:02d}-{whole_seconds:02d}"


def main() -> None:
    """Sample the configured video and write frames to saved_frames/."""
    if not CONFIG.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {CONFIG.video_path}")

    CONFIG.saved_frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(CONFIG.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {CONFIG.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        raise RuntimeError("Video FPS could not be read.")

    frame_interval = max(1, int(round(fps * CONFIG.frame_sample_seconds)))
    frame_count = 0
    saved_count = 0
    start = time.perf_counter()

    print(f"Video          : {CONFIG.video_path}")
    print(f"FPS            : {fps:.2f}")
    print(f"Frame interval : {frame_interval}")
    print(f"Output         : {CONFIG.saved_frames_dir}\n")

    with tqdm(total=total_frames or None, desc="Extracting frames") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            progress.update(1)

            if frame_count % frame_interval != 0:
                continue

            timestamp_seconds = frame_count / fps
            timestamp = format_timestamp(timestamp_seconds)
            saved_count += 1
            frame_name = CONFIG.saved_frames_dir / f"frame_{saved_count:03d}_{timestamp}.jpg"
            cv2.imwrite(str(frame_name), frame)

    cap.release()
    elapsed = max(time.perf_counter() - start, 1e-9)
    print("\n========== EXTRACTION COMPLETED ==========")
    print(f"Total frames read : {frame_count}")
    print(f"Frames saved      : {saved_count}")
    print(f"Processing speed  : {frame_count / elapsed:.2f} frames/sec")


if __name__ == "__main__":
    main()
