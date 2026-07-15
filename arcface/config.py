"""Central configuration for the face re-identification pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineConfig:
    """Paths, thresholds, and model settings used by all pipeline stages."""

    base_dir: Path = BASE_DIR
    video_path: Path = BASE_DIR / "WhatsApp Video 2026-06-02 at 9.09.36 PM.mp4"
    saved_frames_dir: Path = BASE_DIR / "saved_frames"
    cropped_persons_dir: Path = BASE_DIR / "cropped_persons"
    cropped_faces_dir: Path = BASE_DIR / "cropped_faces"
    video_cache_dir: Path = BASE_DIR / "video_cache"

    yolo_person_model: Path = BASE_DIR / "yolo11n.pt"
    yolo_face_model: Path = BASE_DIR / "yolov8n-face-lindevs.pt"
    query_image_path: Path = BASE_DIR / "query10.png"
    query_face_crop_path: Path = BASE_DIR / "query_face_crop.png"

    faiss_index_file: Path = BASE_DIR / "video_cache" / "active" / "faiss" / "index.faiss"
    metadata_file: Path = BASE_DIR / "video_cache" / "active" / "metadata.json"
    threshold_stats_file: Path = BASE_DIR / "video_cache" / "active" / "threshold_stats.json"
    face_detection_metadata_file: Path = BASE_DIR / "cropped_faces" / "face_detection_metadata.json"
    query_preview_path: Path = BASE_DIR / "query_face_preview.png"
    visualization_dir: Path = BASE_DIR / "video_cache" / "active" / "visualizations"

    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm")

    arcface_model_name: str = "w600k_r50"
    arcface_candidates: tuple[Path, ...] = field(
        default_factory=lambda: (
            Path.home() / ".insightface" / "models" / "w600k_r50" / "model.onnx",
            Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
            Path.home() / ".insightface" / "models" / "arcface_w600k_r50" / "model.onnx",
            Path.home() / ".insightface" / "models" / "arcface_r50_v1" / "model.onnx",
            Path.home() / ".insightface" / "models" / "arcface_r100_v1" / "model.onnx",
        )
    )
    insightface_detection_model: str = "buffalo_l"

    embedding_dim: int = 512
    aligned_face_size: int = 112
    embedding_batch_size: int = 32

    person_confidence: float = 0.20
    face_confidence: float = 0.35
    yolo_low_confidence: float = 0.45
    retinaface_confidence: float = 0.45
    face_padding_ratio: float = 0.15
    min_crop_size: int = 5

    top_k: int = 20
    default_similarity_threshold: float = 0.80
    adaptive_threshold_margin: float = 0.02

    frame_sample_seconds: float = 0.5


CONFIG = PipelineConfig()


def compute_video_hash(video_path: Path) -> str:
    """Create a stable SHA-256 cache key from the uploaded video bytes."""
    resolved = video_path.resolve()
    if not resolved.exists():
        payload = str(resolved).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_cache_paths(video_path: Path | None = None, config: PipelineConfig = CONFIG) -> dict[str, Path]:
    """Return the single embedding/index storage location for a video."""
    target_video = video_path or config.video_path
    video_hash = compute_video_hash(target_video) if target_video.exists() else "unknown_video"
    root = config.video_cache_dir / video_hash
    return {
        "root": root,
        "embeddings_dir": root / "embeddings",
        "embedding_db": root / "embeddings" / "embeddings.pkl",
        "faiss_dir": root / "faiss",
        "faiss_index": root / "faiss" / "index.faiss",
        "metadata": root / "metadata.json",
        "status": root / "processing_status.json",
        "threshold_stats": root / "threshold_stats.json",
        "processed_faces": root / "processed_faces.json",
        "faces_dir": root / "faces",
        "visualization_dir": root / "visualizations",
        "evaluation_metrics": root / "evaluation_metrics.json",
        "lock": root / ".cache.lock",
    }


def activate_video_cache(video_path: Path | None = None, config: PipelineConfig = CONFIG) -> dict[str, Path]:
    """Point compatibility config fields at the resolved video cache paths."""
    paths = video_cache_paths(video_path, config)
    object.__setattr__(config, "faiss_index_file", paths["faiss_index"])
    object.__setattr__(config, "metadata_file", paths["metadata"])
    object.__setattr__(config, "threshold_stats_file", paths["threshold_stats"])
    object.__setattr__(config, "visualization_dir", paths["visualization_dir"])
    return paths
