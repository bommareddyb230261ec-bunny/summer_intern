"""Build a metadata-rich FAISS database of aligned ArcFace embeddings."""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
from tqdm import tqdm

from config import CONFIG, activate_video_cache, video_cache_paths
from face_recognition_utils import (
    align_face,
    compute_threshold_stats,
    configure_logging,
    detection_to_dict,
    embed_aligned_faces,
    image_to_bgr,
    load_arcface,
    load_retinaface,
    save_json,
    select_device,
)

FACE_NAME_PATTERNS = (
    re.compile(r"^face_frame(?P<frame>\d+)_person(?P<person>\d+)_face(?P<face>\d+)$", re.IGNORECASE),
    re.compile(r"^frame_(?P<frame>\d+)_person_(?P<person>\d+)_face_(?P<face>\d+)$", re.IGNORECASE),
)

CACHE_LOCK_NAME = ".cache.lock"


def prepare_directories() -> None:
    """Create output folders and verify required input folders."""
    CONFIG.video_cache_dir.mkdir(parents=True, exist_ok=True)
    if not CONFIG.cropped_faces_dir.is_dir():
        raise FileNotFoundError(f"Face crop folder does not exist: {CONFIG.cropped_faces_dir}")
    if not CONFIG.saved_frames_dir.is_dir():
        raise FileNotFoundError(f"Saved frame folder does not exist: {CONFIG.saved_frames_dir}")


def discover_images(folder: Path) -> list[Path]:
    """Return supported images in stable order."""
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in CONFIG.image_extensions),
        key=lambda path: path.name.lower(),
    )


def parse_face_filename(face_path: Path) -> dict[str, int]:
    """Extract frame, person, and face indexes from a crop filename."""
    for pattern in FACE_NAME_PATTERNS:
        match = pattern.fullmatch(face_path.stem)
        if match:
            return {key: int(value) for key, value in match.groupdict().items()}
    raise ValueError(f"Unsupported face filename: {face_path.name}")


def extract_frame_number(frame_path: Path) -> int | None:
    """Extract frame number from saved frame name."""
    match = re.match(r"^frame_(\d+)(?:_|$)", frame_path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def find_matching_frame(frame_number: int, frame_paths: list[Path]) -> Path:
    """Find saved frame matching the parsed face crop frame number."""
    matches = [path for path in frame_paths if extract_frame_number(path) == frame_number]
    if not matches:
        raise FileNotFoundError(f"No saved frame found for frame number {frame_number}")
    if len(matches) > 1:
        raise ValueError(f"Multiple saved frames match frame {frame_number}: {[path.name for path in matches]}")
    return matches[0]


def timestamp_from_frame(frame_path: Path) -> float:
    """Extract timestamp seconds from supported frame names."""
    seconds_match = re.fullmatch(r"frame_\d+_time_(?P<seconds>\d+(?:\.\d+)?)", frame_path.stem, re.IGNORECASE)
    if seconds_match:
        return float(seconds_match.group("seconds"))

    clock_match = re.fullmatch(
        r"frame_\d+_(?P<hours>\d{2})-(?P<minutes>\d{2})-(?P<seconds>\d{2}(?:\.\d+)?)",
        frame_path.stem,
        re.IGNORECASE,
    )
    if clock_match:
        hours = int(clock_match.group("hours"))
        minutes = int(clock_match.group("minutes"))
        seconds = float(clock_match.group("seconds"))
        return hours * 3600.0 + minutes * 60.0 + seconds
    raise ValueError(f"Could not extract timestamp from {frame_path.name}")


def find_person_crop(frame_number: int, person_number: int) -> Path:
    """Resolve corresponding person crop."""
    candidate_stems = (f"person_frame{frame_number}_person{person_number}", f"frame_{frame_number}_person_{person_number}")
    for stem in candidate_stems:
        for extension in CONFIG.image_extensions:
            candidate = CONFIG.cropped_persons_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No person crop found for frame {frame_number}, person {person_number}")


def detect_video_name() -> str:
    """Infer the source video when exactly one video exists in the project."""
    videos = sorted(
        path for path in CONFIG.base_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CONFIG.video_extensions
    )
    if CONFIG.video_path.exists():
        return CONFIG.video_path.name
    return videos[0].name if len(videos) == 1 else ""


def prepare_cache(paths: dict[str, Path]) -> None:
    """Create per-video cache directories."""
    paths["embeddings_dir"].mkdir(parents=True, exist_ok=True)
    paths["faiss_dir"].mkdir(parents=True, exist_ok=True)
    paths["faces_dir"].mkdir(parents=True, exist_ok=True)


def save_face_preview(face_path: Path, paths: dict[str, Path]) -> Path:
    """Persist a compact matched-face preview into the video cache."""
    destination = paths["faces_dir"] / face_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(face_path, destination)
    return destination


@contextmanager
def cache_lock(lock_path: Path, timeout_seconds: float = 120.0):
    """Use an atomic lock file to protect per-video cache writes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    handle: int | None = None
    while handle is None:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode("utf-8"))
        except FileExistsError:
            if time.perf_counter() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        if handle is not None:
            os.close(handle)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def atomic_pickle(path: Path, payload: Any) -> None:
    """Write pickle data via temporary replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_embedding_db(path: Path) -> list[dict[str, Any]]:
    """Load persisted embedding records."""
    if not path.exists():
        return []
    with path.open("rb") as file:
        records = pickle.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Embedding database is invalid: {path}")
    return records


def save_incremental_state(paths: dict[str, Path], records: list[dict[str, Any]], processed_faces: dict[str, Any]) -> None:
    """Persist embedding records, metadata, and processed-face manifest."""
    atomic_pickle(paths["embedding_db"], records)
    metadata = [record["metadata"] for record in records]
    save_json(paths["metadata"], metadata)
    save_json(paths["processed_faces"], processed_faces)


def records_to_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Convert persisted records to FAISS-ready arrays and metadata."""
    metadata = [record["metadata"] for record in records]
    if not records:
        return np.empty((0, CONFIG.embedding_dim), dtype=np.float32), metadata
    embeddings = np.vstack([np.asarray(record["embedding"], dtype=np.float32) for record in records])
    return np.ascontiguousarray(embeddings, dtype=np.float32), metadata


def load_detection_metadata() -> dict[str, dict[str, Any]]:
    """Load optional face detection sidecar written by face_cropping.py."""
    if not CONFIG.face_detection_metadata_file.exists():
        return {}
    with CONFIG.face_detection_metadata_file.open("r", encoding="utf-8") as file:
        entries = json.load(file)
    return {entry["face_image"]: entry for entry in entries if isinstance(entry, dict) and "face_image" in entry}


def build_records(
    face_paths: list[Path],
    frame_paths: list[Path],
    video_name: str,
    retinaface: Any,
    detection_metadata: dict[str, dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[tuple[str, str]]]:
    """Align face crops and build metadata records before batched embedding."""
    aligned_faces: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for face_path in tqdm(face_paths, desc="Aligning faces"):
        try:
            identifiers = parse_face_filename(face_path)
            frame_path = find_matching_frame(identifiers["frame"], frame_paths)
            person_path = find_person_crop(identifiers["frame"], identifiers["person"])
            image_bgr = image_to_bgr(face_path)
            aligned, landmark_detection = align_face(image_bgr, retinaface, CONFIG)
            sidecar = detection_metadata.get(face_path.name, {})

            aligned_faces.append(aligned)
            records.append(
                {
                    "face_image": face_path.name,
                    "face_image_path": str(face_path.resolve()),
                    "person_crop": person_path.name,
                    "person_crop_path": str(person_path.resolve()),
                    "frame_name": frame_path.name,
                    "frame_path": str(frame_path.resolve()),
                    "frame_number": identifiers["frame"],
                    "timestamp": timestamp_from_frame(frame_path),
                    "video_name": video_name,
                    "video_path": str((CONFIG.base_dir / video_name).resolve()) if video_name else "",
                    "person_id": identifiers["person"],
                    "track_id": identifiers["person"],
                    "face_id": identifiers["face"],
                    "bbox": sidecar.get("bbox_in_person_crop"),
                    "embedding_dimension": CONFIG.embedding_dim,
                    "embedding_model": f"InsightFace ArcFace {CONFIG.arcface_model_name}",
                    "confidence": float(sidecar.get("confidence", landmark_detection.confidence if landmark_detection else 0.0)),
                    "face_detector": sidecar.get("detector", landmark_detection.detector if landmark_detection else "unknown"),
                    "alignment_model": CONFIG.insightface_detection_model,
                    "alignment_detection": detection_to_dict(landmark_detection),
                }
            )
        except Exception as exc:
            failures.append((face_path.name, str(exc)))
            print(f"  FAILED {face_path.name}: {exc}")
    return aligned_faces, records, failures


def save_faiss_database(embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
    """Persist IndexFlatL2 and aligned metadata."""
    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    index = faiss.IndexFlatL2(CONFIG.embedding_dim)
    if len(embeddings):
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index.add(matrix)

    temporary_index = CONFIG.faiss_index_file.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(CONFIG.faiss_index_file)
    save_json(CONFIG.metadata_file, metadata)
    print(f"FAISS index saved: {CONFIG.faiss_index_file}")
    print(f"Metadata saved   : {CONFIG.metadata_file}")


def update_faiss_database(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    cache_index_path: Path,
    previous_count: int,
) -> None:
    """Create or append to the per-video FAISS index."""
    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    matrix = np.ascontiguousarray(embeddings.astype(np.float32))
    if len(matrix):
        faiss.normalize_L2(matrix)

    can_append = (
        cache_index_path.exists()
        and previous_count > 0
        and previous_count <= len(matrix)
    )
    if can_append:
        index = faiss.read_index(str(cache_index_path))
        if index.d == CONFIG.embedding_dim and index.ntotal == previous_count:
            new_vectors = matrix[previous_count:]
            if len(new_vectors):
                index.add(new_vectors)
                print(f"[FAISS UPDATE] appended {len(new_vectors)} vectors to {cache_index_path}")
            else:
                print(f"[FAISS UPDATE] no new vectors to append for {cache_index_path}")
        else:
            can_append = False

    if not can_append:
        index = faiss.IndexFlatL2(CONFIG.embedding_dim)
        if len(matrix):
            index.add(matrix)
        print(f"[FAISS UPDATE] rebuilt index with {index.ntotal} vectors at {cache_index_path}")

    temporary_index = cache_index_path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(cache_index_path)

    save_json(CONFIG.metadata_file, metadata)
    print(f"[FAISS UPDATE] index ready at {CONFIG.faiss_index_file}")
    print(f"[FAISS UPDATE] metadata ready at {CONFIG.metadata_file}")


def generate_incremental_embeddings(
    face_paths: list[Path],
    frame_paths: list[Path],
    video_name: str,
    arcface: Any,
    retinaface: Any,
    detection_metadata: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[str, str]], int]:
    """Generate, save, and resume embeddings one face at a time."""
    records = load_embedding_db(paths["embedding_db"])
    processed_faces = {
        record.get("metadata", {}).get("face_image"): {
            "saved_at": record.get("saved_at"),
            "video_hash": record.get("video_hash"),
        }
        for record in records
        if record.get("metadata", {}).get("face_image")
    }
    existing_count = len(records)
    failures: list[tuple[str, str]] = []

    print(f"Incremental embedding cache: {paths['embedding_db']}")
    print(f"Existing embeddings        : {existing_count}")

    for face_path in tqdm(face_paths, desc="Embedding faces"):
        if face_path.name in processed_faces:
            print(f"[EMBEDDING SKIP] already cached: {face_path.name}")
            continue

        try:
            aligned_faces, metadata, record_failures = build_records(
                [face_path],
                frame_paths,
                video_name,
                retinaface,
                detection_metadata,
            )
            if record_failures:
                failures.extend(record_failures)
                continue
            if not aligned_faces or not metadata:
                failures.append((face_path.name, "No aligned face generated."))
                continue

            embedding = embed_aligned_faces(aligned_faces, arcface, CONFIG)[0]
            preview_path = save_face_preview(face_path, paths)
            metadata[0]["matched_face_image"] = str(preview_path.resolve())
            metadata[0]["face_image_path"] = str(preview_path.resolve())
            record = {
                "embedding": np.asarray(embedding, dtype=np.float32),
                "metadata": metadata[0],
                "video_hash": paths["root"].name,
                "saved_at": time.time(),
            }
            records.append(record)
            processed_faces[face_path.name] = {
                "saved_at": record["saved_at"],
                "video_hash": record["video_hash"],
            }
            save_incremental_state(paths, records, processed_faces)
            print(f"[EMBEDDING SAVED] {face_path.name} -> {paths['embedding_db']}")
        except Exception as exc:
            failures.append((face_path.name, str(exc)))
            print(f"  FAILED {face_path.name}: {exc}")

    embeddings, metadata = records_to_arrays(records)
    return embeddings, metadata, failures, existing_count


def validate_saved_database(expected_count: int) -> None:
    """Reload saved files and validate index/metadata alignment."""
    index = faiss.read_index(str(CONFIG.faiss_index_file))
    with CONFIG.metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if index.d != CONFIG.embedding_dim or index.ntotal != expected_count:
        raise ValueError("Saved FAISS index dimension/count does not match generated data.")
    if not isinstance(metadata, list) or len(metadata) != expected_count:
        raise ValueError("Saved metadata count does not match generated data.")

    required = {
        "face_image", "face_image_path", "person_crop", "person_crop_path", "frame_name", "frame_path",
        "frame_number", "timestamp", "video_name", "person_id", "track_id", "face_id",
        "embedding_dimension", "embedding_model", "confidence",
    }
    for position, entry in enumerate(metadata):
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"Metadata entry {position} missing fields: {sorted(missing)}")
    print(f"Validated {index.ntotal} FAISS vectors with synchronized metadata.")


def print_statistics(total_faces: int, total_saved: int, failures: list[tuple[str, str]], elapsed: float) -> None:
    """Print final indexing statistics."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Total faces discovered : {total_faces}")
    print(f"Total embeddings saved : {total_saved}")
    print(f"Failed images          : {len(failures)}")
    print(f"End-to-end speed       : {total_faces / max(elapsed, 1e-9):.2f} faces/sec")
    if failures:
        for filename, reason in failures[:20]:
            print(f"  - {filename}: {reason}")
    print(sep)


def main() -> None:
    """Run the complete master embedding database workflow."""
    configure_logging()
    start = time.perf_counter()
    prepare_directories()
    face_paths = discover_images(CONFIG.cropped_faces_dir)
    frame_paths = discover_images(CONFIG.saved_frames_dir)
    print(f"Face crops discovered: {len(face_paths)}")
    print(f"Saved frames discovered: {len(frame_paths)}")
    paths = activate_video_cache(CONFIG.video_path)
    prepare_cache(paths)

    device = select_device()
    arcface = load_arcface(device)
    retinaface = load_retinaface(device)
    detection_metadata = load_detection_metadata()
    video_name = detect_video_name()

    embed_start = time.perf_counter()
    with cache_lock(paths["lock"]):
        embeddings, metadata, failures, existing_count = generate_incremental_embeddings(
            face_paths,
            frame_paths,
            video_name,
            arcface,
            retinaface,
            detection_metadata,
            paths,
        )
        print(f"Embedding generation/resume time: {time.perf_counter() - embed_start:.2f}s")

        index_start = time.perf_counter()
        update_faiss_database(embeddings, metadata, paths["faiss_index"], existing_count)
    print(f"FAISS indexing time     : {time.perf_counter() - index_start:.2f}s")
    stats = compute_threshold_stats(embeddings, metadata, CONFIG)
    save_json(CONFIG.threshold_stats_file, stats)
    print(f"Adaptive threshold      : {stats['adaptive_threshold']:.4f}")

    validate_saved_database(len(metadata))
    print_statistics(len(face_paths), len(metadata), failures, time.perf_counter() - start)


if __name__ == "__main__":
    main()
