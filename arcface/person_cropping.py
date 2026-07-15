#!/usr/bin/env python
"""Crop persons from saved frames using YOLO11."""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from config import CONFIG

PERSON_CLASS = 0


def setup() -> list[Path]:
    """Validate inputs and discover frames."""
    if not CONFIG.saved_frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {CONFIG.saved_frames_dir}")
    if not CONFIG.yolo_person_model.exists():
        raise FileNotFoundError(f"Model not found: {CONFIG.yolo_person_model}")

    CONFIG.cropped_persons_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in CONFIG.saved_frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CONFIG.image_extensions
    )
    if not images:
        raise ValueError(f"No images found in: {CONFIG.saved_frames_dir}")

    print(f"Frames : {CONFIG.saved_frames_dir} ({len(images)} files)")
    print(f"Output : {CONFIG.cropped_persons_dir}")
    print(f"Model  : {CONFIG.yolo_person_model}\n")
    return images


def load_model() -> YOLO:
    """Load YOLO11 person detector."""
    print("Loading YOLO11 person model...")
    model = YOLO(str(CONFIG.yolo_person_model))
    print("Model loaded\n")
    return model


def extract_frame_number(filename: str) -> int | None:
    """Extract a numeric frame id from current or legacy frame names."""
    for pattern in (r"frame_(\d+)", r"(\d+)"):
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return None


def get_box_data(boxes: object) -> np.ndarray:
    """Return detections as rows of x1, y1, x2, y2, score, class_id."""
    try:
        return boxes.data.cpu().numpy()
    except Exception:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        return np.column_stack([xyxy, confs, cls_ids])


def crop_persons(img_bgr: np.ndarray, box_data: np.ndarray, frame_id: int | str) -> int:
    """Save all valid person detections for one frame."""
    h, w = img_bgr.shape[:2]
    saved = 0

    for row in box_data:
        x1, y1, x2, y2, score, cls_id = row[:6]
        if int(cls_id) != PERSON_CLASS or float(score) < CONFIG.person_confidence:
            continue

        x1i = max(0, int(round(x1)))
        y1i = max(0, int(round(y1)))
        x2i = min(w, int(round(x2)))
        y2i = min(h, int(round(y2)))
        if (x2i - x1i) <= CONFIG.min_crop_size or (y2i - y1i) <= CONFIG.min_crop_size:
            continue

        crop = img_bgr[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            continue

        saved += 1
        out_path = CONFIG.cropped_persons_dir / f"frame_{frame_id}_person_{saved}.jpg"
        cv2.imwrite(str(out_path), crop)
    return saved


def process_frames(images: list[Path], model: YOLO) -> int:
    """Detect and save person crops for all frames."""
    total_saved = 0
    start = time.perf_counter()

    for img_path in tqdm(images, desc="Cropping persons"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: Could not read {img_path.name}")
            continue

        results = model(img_bgr, conf=CONFIG.person_confidence, imgsz=640, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        frame_id = extract_frame_number(img_path.name) or img_path.stem
        saved = crop_persons(img_bgr, get_box_data(results[0].boxes), frame_id)
        total_saved += saved

    elapsed = max(time.perf_counter() - start, 1e-9)
    print(f"Processing speed: {len(images) / elapsed:.2f} frames/sec")
    return total_saved


def print_summary(total_frames: int, total_saved: int) -> None:
    """Print final crop statistics."""
    sep = "=" * 50
    print(f"\n{sep}")
    print("PERSON CROPPING COMPLETE")
    print(f"  Frames processed : {total_frames}")
    print(f"  Persons cropped  : {total_saved}")
    print(f"  Output folder    : {CONFIG.cropped_persons_dir}")
    print(sep)


def main() -> None:
    """Run person cropping."""
    images = setup()
    model = load_model()
    total_saved = process_frames(images, model)
    print_summary(len(images), total_saved)


if __name__ == "__main__":
    main()
