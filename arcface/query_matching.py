#!/usr/bin/env python
"""Query the FAISS face database with aligned ArcFace embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import CONFIG, activate_video_cache
from face_recognition_utils import (
    align_face,
    confidence_from_similarity,
    configure_logging,
    crop_with_padding,
    detect_faces_with_fallback,
    embed_aligned_faces,
    image_to_bgr,
    load_arcface,
    load_retinaface,
    load_threshold,
    select_device,
    squared_l2_to_cosine,
)


def setup() -> None:
    """Validate required query-time files."""
    activate_video_cache(CONFIG.video_path)
    for path in (CONFIG.query_image_path, CONFIG.faiss_index_file, CONFIG.metadata_file, CONFIG.yolo_face_model):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Query image   : {CONFIG.query_image_path}")
    print(f"FAISS index   : {CONFIG.faiss_index_file}")
    print(f"Metadata file : {CONFIG.metadata_file}\n")


def load_models() -> tuple[Any, Any, Any]:
    """Load ArcFace, YOLOv8-face, and RetinaFace."""
    device = select_device()
    arcface = load_arcface(device)
    yolo = YOLO(str(CONFIG.yolo_face_model))
    retinaface = load_retinaface(device)
    return arcface, yolo, retinaface


def load_search_database() -> tuple[Any, list[dict[str, Any]]]:
    """Load persistent FAISS index and synchronized metadata."""
    activate_video_cache(CONFIG.video_path)
    index = faiss.read_index(str(CONFIG.faiss_index_file))
    with CONFIG.metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Metadata database is empty or invalid: {CONFIG.metadata_file}")
    if index.d != CONFIG.embedding_dim:
        raise ValueError(f"Expected {CONFIG.embedding_dim}-D FAISS index, got {index.d}")
    if index.ntotal != len(metadata):
        raise ValueError("FAISS vector count does not match metadata count.")

    print(f"Loaded {index.ntotal} vectors from {type(index).__name__}")
    return index, metadata


def save_query_face_crop(crop: np.ndarray, output_path: Path | None = None) -> Path:
    """Persist the detected query-face crop to disk."""
    target = output_path or CONFIG.query_face_crop_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), crop):
        raise RuntimeError(f"Failed to write query face crop to {target}")
    return target


def detect_query_face(query_bgr: np.ndarray, yolo: Any, retinaface: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect the best query face and return a crop plus detection details."""
    detections = detect_faces_with_fallback(query_bgr, yolo, retinaface, CONFIG)
    if not detections:
        raise ValueError("No face detected in query image.")

    best = max(detections, key=lambda detection: (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]))
    crop, padded_bbox = crop_with_padding(query_bgr, best.bbox, CONFIG.face_padding_ratio)
    if crop is None or crop.size == 0:
        raise ValueError("Query face crop is empty.")

    save_query_face_crop(crop, CONFIG.query_face_crop_path)
    cv2.imwrite(str(CONFIG.query_preview_path), crop)
    details = {
        "detector": best.detector,
        "confidence": best.confidence,
        "bbox": list(padded_bbox),
        "crop_path": str(CONFIG.query_face_crop_path.resolve()),
        "preview_path": str(CONFIG.query_preview_path.resolve()),
    }
    return crop, details


def embed_query_face(query_face_bgr: np.ndarray, arcface: Any, retinaface: Any) -> np.ndarray:
    """Align and embed one query face."""
    aligned, _ = align_face(query_face_bgr, retinaface, CONFIG)
    embeddings = embed_aligned_faces([aligned], arcface, CONFIG)
    return embeddings[0]


def search(query_embedding: np.ndarray, index: Any, metadata: list[dict[str, Any]], top_k: int = CONFIG.top_k) -> list[dict[str, Any]]:
    """Return ranked FAISS Top-K results with cosine similarity."""
    if index.ntotal == 0:
        return []
    query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(query)
    distances, positions = index.search(query, min(top_k, index.ntotal))

    results: list[dict[str, Any]] = []
    for rank, (squared_l2, position) in enumerate(zip(distances[0], positions[0]), start=1):
        if position < 0:
            continue
        similarity = squared_l2_to_cosine(float(squared_l2))
        results.append(
            {
                **metadata[int(position)],
                "rank": rank,
                "faiss_index": int(position),
                "squared_l2_distance": float(squared_l2),
                "euclidean_distance": float(np.sqrt(max(float(squared_l2), 0.0))),
                "cosine_similarity": similarity,
            }
        )
    return results


def apply_adaptive_threshold(results: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Filter ranked results and add confidence scores."""
    matched: list[dict[str, Any]] = []
    for result in results:
        confidence = confidence_from_similarity(result["cosine_similarity"], threshold)
        enriched = {**result, "match_confidence": confidence, "threshold": threshold}
        if result["cosine_similarity"] >= threshold:
            matched.append(enriched)
    return matched


def show_match(query_face_bgr: np.ndarray, match: dict[str, Any]) -> None:
    """Display query and a matched face side by side."""
    matched_path = Path(match["face_image_path"])
    with Image.open(matched_path) as image:
        matched_image = image.convert("RGB")
    query_rgb = cv2.cvtColor(query_face_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(query_rgb)
    plt.title("Query Face")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(matched_image)
    plt.title(f"Rank {match['rank']} | sim {match['cosine_similarity']:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_ranked_results(results: list[dict[str, Any]], threshold: float) -> None:
    """Print Top-K retrieval table."""
    print(f"\nTop-{CONFIG.top_k} ranked retrieval (threshold={threshold:.4f})")
    for result in results:
        status = "MATCH" if result["cosine_similarity"] >= threshold else "UNKNOWN-CANDIDATE"
        print(
            f"#{result['rank']:02d} {status:<17} "
            f"sim={result['cosine_similarity']:.4f} "
            f"dist={result['euclidean_distance']:.4f} "
            f"time={float(result['timestamp']):.2f}s "
            f"frame={result['frame_name']} face={result['face_image']}"
        )


def print_match_results(results: list[dict[str, Any]], matched: list[dict[str, Any]], threshold: float, query_time: float) -> None:
    """Print final retrieval decision."""
    sep = "=" * 60
    print(f"\n{sep}\nMATCH RESULTS\n{sep}")
    if not matched:
        print("UNKNOWN PERSON")
        if results:
            print(f"Best similarity : {results[0]['cosine_similarity']:.4f}")
            print(f"Threshold       : {threshold:.4f}")
            print(f"Best candidate  : {results[0]['face_image']}")
    else:
        print(f"QUERY FOUND - {len(matched)} ranked match(es) above threshold\n")
        for match in matched:
            print(f"Rank       : {match['rank']}")
            print(f"Confidence : {match['match_confidence']:.4f}")
            print(f"Similarity : {match['cosine_similarity']:.4f}")
            print(f"Video      : {match['video_name']}")
            print(f"Frame      : {match['frame_name']}")
            print(f"Timestamp  : {float(match['timestamp']):.2f} seconds")
            print(f"Face image : {match['face_image']}")
            print(f"Person crop: {match['person_crop']}\n")
    print(f"Query time : {query_time:.4f}s")
    print(sep)


def main() -> None:
    """Run query matching."""
    configure_logging()
    setup()
    arcface, yolo, retinaface = load_models()
    index, metadata = load_search_database()
    threshold = load_threshold(CONFIG)

    query_start = time.perf_counter()
    query_bgr = image_to_bgr(CONFIG.query_image_path)
    query_face, query_detection = detect_query_face(query_bgr, yolo, retinaface)
    query_embedding = embed_query_face(query_face, arcface, retinaface)
    results = search(query_embedding, index, metadata, CONFIG.top_k)
    matched = apply_adaptive_threshold(results, threshold)
    query_time = time.perf_counter() - query_start

    print(f"Query detector: {query_detection['detector']} conf={query_detection['confidence']:.4f}")
    print_ranked_results(results, threshold)
    print_match_results(results, matched, threshold, query_time)

    for match in matched[:5]:
        show_match(query_face, match)


if __name__ == "__main__":
    main()
