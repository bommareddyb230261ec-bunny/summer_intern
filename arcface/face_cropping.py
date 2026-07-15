#!/usr/bin/env python
"""Crop faces from person crops using YOLOv8-face with RetinaFace fallback."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from config import CONFIG
from face_recognition_utils import (
    configure_logging,
    crop_with_padding,
    detection_to_dict,
    detect_faces_with_fallback,
    load_retinaface,
    save_json,
    select_device,
)


def setup() -> list[Path]:
    """Validate inputs and return person crop images."""
    if not CONFIG.yolo_face_model.exists():
        raise FileNotFoundError(f"Model not found: {CONFIG.yolo_face_model}")
    if not CONFIG.cropped_persons_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {CONFIG.cropped_persons_dir}")

    CONFIG.cropped_faces_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in CONFIG.cropped_persons_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CONFIG.image_extensions
    )
    if not images:
        raise ValueError(f"No images found in: {CONFIG.cropped_persons_dir}")

    print(f"Input : {CONFIG.cropped_persons_dir} ({len(images)} images)")
    print(f"Output: {CONFIG.cropped_faces_dir}")
    return images


def load_models() -> tuple[Any, Any]:
    """Load primary YOLO detector and RetinaFace fallback."""
    device = select_device()
    print("Loading YOLOv8-face model...")
    yolo = YOLO(str(CONFIG.yolo_face_model))
    print("Loading RetinaFace fallback/landmark model...")
    retinaface = load_retinaface(device)
    print("Models loaded\n")
    return yolo, retinaface


def crop_faces_from_image(img_path: Path, yolo: Any, retinaface: Any) -> tuple[int, list[dict[str, Any]]]:
    """Detect and save all faces from one person crop."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  WARNING: Could not read {img_path.name}")
        return 0, []

    detections = detect_faces_with_fallback(image, yolo, retinaface, CONFIG)
    saved = 0
    metadata: list[dict[str, Any]] = []

    for detection in detections:
        crop, padded_bbox = crop_with_padding(image, detection.bbox, CONFIG.face_padding_ratio)
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        if w <= CONFIG.min_crop_size or h <= CONFIG.min_crop_size:
            continue

        saved += 1
        out_path = CONFIG.cropped_faces_dir / f"{img_path.stem}_face_{saved}.jpg"
        cv2.imwrite(str(out_path), crop)
        metadata.append(
            {
                "face_image": out_path.name,
                "face_image_path": str(out_path.resolve()),
                "person_crop": img_path.name,
                "person_crop_path": str(img_path.resolve()),
                "bbox_in_person_crop": list(padded_bbox),
                "raw_detection": detection_to_dict(detection),
                "detector": detection.detector,
                "confidence": detection.confidence,
            }
        )

    if saved == 0:
        print(f"  No face detected: {img_path.name}")
    return saved, metadata


def detect_and_crop(images: list[Path], yolo: Any, retinaface: Any) -> list[dict[str, Any]]:
    """Process all person crops and persist detection metadata."""
    all_metadata: list[dict[str, Any]] = []
    total_saved = 0
    start = time.perf_counter()

    for img_path in tqdm(images, desc="Cropping faces"):
        saved, metadata = crop_faces_from_image(img_path, yolo, retinaface)
        total_saved += saved
        all_metadata.extend(metadata)

    elapsed = max(time.perf_counter() - start, 1e-9)
    save_json(CONFIG.face_detection_metadata_file, all_metadata)
    print(f"\nFace detections saved: {total_saved}")
    print(f"Detection metadata : {CONFIG.face_detection_metadata_file}")
    print(f"Processing speed   : {len(images) / elapsed:.2f} person crops/sec")
    return all_metadata


def print_summary(total_input: int, total_saved: int) -> None:
    """Print a concise pipeline summary."""
    sep = "=" * 50
    print(f"\n{sep}")
    print("FACE CROPPING COMPLETE")
    print(f"  Input images  : {total_input}")
    print(f"  Faces cropped : {total_saved}")
    print(f"  Output folder : {CONFIG.cropped_faces_dir}")
    print(sep)


def main() -> None:
    """Run face cropping."""
    configure_logging()
    images = setup()
    yolo, retinaface = load_models()
    metadata = detect_and_crop(images, yolo, retinaface)
    print_summary(len(images), len(metadata))


if __name__ == "__main__":
    main()
