"""Shared detection, alignment, embedding, and threshold utilities."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import faiss
import numpy as np
import torch
from PIL import Image

from config import CONFIG, PipelineConfig, activate_video_cache


LOGGER = logging.getLogger("face_reid")


def configure_logging() -> None:
    """Configure concise stage logging once per script."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass(frozen=True)
class FaceDetection:
    """One detected face and the information needed for auditability."""

    bbox: tuple[int, int, int, int]
    confidence: float
    detector: str
    landmarks: list[list[float]] | None = None


def select_device() -> torch.device:
    """Use CUDA automatically when available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Compute device: %s", device)
    return device


def load_arcface(device: torch.device, config: PipelineConfig = CONFIG) -> Any:
    """Load InsightFace ArcFace ONNX recognition weights."""
    try:
        from insightface.model_zoo import ArcFaceONNX
    except ImportError as exc:
        raise ImportError("Install insightface and onnxruntime to use ArcFace embeddings.") from exc

    for candidate in config.arcface_candidates:
        if candidate.exists():
            LOGGER.info("Loading ArcFace model: %s", candidate)
            model = ArcFaceONNX(model_file=str(candidate))
            model.prepare(ctx_id=-1 if device.type == "cpu" else 0)
            return model

    raise RuntimeError(
        "No ArcFace ONNX model found. Place InsightFace weights in ~/.insightface/models "
        f"(searched: {[str(path) for path in config.arcface_candidates]})."
    )


def load_retinaface(device: torch.device, config: PipelineConfig = CONFIG) -> Any:
    """Load InsightFace RetinaFace detector for fallback and landmarks."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError("Install insightface to use RetinaFace fallback and landmarks.") from exc

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=config.insightface_detection_model, providers=providers)
    app.prepare(ctx_id=0 if device.type == "cuda" else -1, det_size=(640, 640))
    return app


def crop_with_padding(image: np.ndarray, bbox: Iterable[float], padding_ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop an image region with proportional padding and clipped coordinates."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    pad = max(4, int(padding_ratio * max(x2 - x1, y2 - y1)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def detect_faces_yolo(image_bgr: np.ndarray, yolo_model: Any, config: PipelineConfig = CONFIG) -> list[FaceDetection]:
    """Detect faces with YOLOv8-face."""
    detections: list[FaceDetection] = []
    results = yolo_model.predict(source=image_bgr, conf=config.face_confidence, device="cpu", verbose=False)
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            detections.append(FaceDetection((x1, y1, x2, y2), confidence, "yolov8-face"))
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def detect_faces_retinaface(image_bgr: np.ndarray, retinaface: Any, config: PipelineConfig = CONFIG) -> list[FaceDetection]:
    """Detect faces and landmarks with InsightFace RetinaFace."""
    detections: list[FaceDetection] = []
    for face in retinaface.get(image_bgr):
        score = float(getattr(face, "det_score", 0.0))
        if score < config.retinaface_confidence:
            continue
        bbox = tuple(int(round(v)) for v in face.bbox.tolist())
        landmarks = getattr(face, "kps", None)
        landmark_list = landmarks.astype(float).tolist() if landmarks is not None else None
        detections.append(FaceDetection(bbox, score, "retinaface", landmark_list))
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def detect_faces_with_fallback(
    image_bgr: np.ndarray,
    yolo_model: Any,
    retinaface: Any,
    config: PipelineConfig = CONFIG,
) -> list[FaceDetection]:
    """Use YOLO first, and call RetinaFace only when YOLO misses or is weak."""
    yolo_faces = detect_faces_yolo(image_bgr, yolo_model, config)
    if yolo_faces and yolo_faces[0].confidence >= config.yolo_low_confidence:
        return yolo_faces
    retina_faces = detect_faces_retinaface(image_bgr, retinaface, config)
    return retina_faces or yolo_faces


def align_face(image_bgr: np.ndarray, retinaface: Any, config: PipelineConfig = CONFIG) -> tuple[np.ndarray, FaceDetection | None]:
    """Align a cropped face to ArcFace's 112x112 canonical view."""
    try:
        from insightface.utils.face_align import norm_crop
    except ImportError as exc:
        raise ImportError("InsightFace face_align is required for landmark-based alignment.") from exc

    faces = detect_faces_retinaface(image_bgr, retinaface, config)
    if faces and faces[0].landmarks is not None:
        kps = np.asarray(faces[0].landmarks, dtype=np.float32)
        aligned = norm_crop(image_bgr, landmark=kps, image_size=config.aligned_face_size)
        return aligned, faces[0]

    LOGGER.warning("No landmarks found; using resized unaligned crop as fallback.")
    resized = cv2.resize(image_bgr, (config.aligned_face_size, config.aligned_face_size), interpolation=cv2.INTER_LINEAR)
    return resized, None


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a finite L2-normalized float32 embedding."""
    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinity.")
    norm = float(np.linalg.norm(embedding))
    if norm <= 0.0:
        raise ValueError("Embedding norm is zero.")
    return embedding / norm


def embed_aligned_faces(
    aligned_faces_bgr: list[np.ndarray],
    arcface: Any,
    config: PipelineConfig = CONFIG,
) -> np.ndarray:
    """Generate L2-normalized ArcFace embeddings in batches."""
    if not aligned_faces_bgr:
        return np.empty((0, config.embedding_dim), dtype=np.float32)

    vectors: list[np.ndarray] = []
    for start in range(0, len(aligned_faces_bgr), config.embedding_batch_size):
        batch = np.stack(aligned_faces_bgr[start:start + config.embedding_batch_size]).astype(np.uint8)
        try:
            raw = arcface.get_feat(batch)
        except Exception:
            raw = [arcface.get_feat(face) for face in batch]

        raw_array = np.asarray(raw, dtype=np.float32)
        batch_size = len(batch)
        if raw_array.ndim == 1:
            raw_array = raw_array.reshape(1, -1)
        elif raw_array.ndim > 2:
            raw_array = raw_array.reshape(raw_array.shape[0], -1)

        if raw_array.shape[0] != batch_size and raw_array.size == batch_size * config.embedding_dim:
            raw_array = raw_array.reshape(batch_size, config.embedding_dim)
        elif raw_array.shape[-1] != config.embedding_dim and raw_array.size % config.embedding_dim == 0:
            raw_array = raw_array.reshape(-1, config.embedding_dim)

        for row in raw_array:
            embedding = np.asarray(row, dtype=np.float32).reshape(-1)
            if embedding.shape[0] != config.embedding_dim:
                raise ValueError(f"Expected {config.embedding_dim}-D embedding, got {embedding.shape[0]}.")
            vectors.append(l2_normalize(embedding))
    return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


def squared_l2_to_cosine(distance: float) -> float:
    """Convert FAISS squared L2 distance between unit vectors to cosine similarity."""
    return float(1.0 - (distance / 2.0))


def confidence_from_similarity(similarity: float, threshold: float) -> float:
    """Map similarity to a human-readable confidence score in [0, 1]."""
    if similarity <= threshold:
        return max(0.0, similarity / max(threshold, 1e-6)) * 0.5
    return min(1.0, 0.5 + ((similarity - threshold) / max(1.0 - threshold, 1e-6)) * 0.5)


def compute_threshold_stats(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    config: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Estimate an adaptive threshold from same/different person similarities."""
    stats: dict[str, Any] = {
        "default_threshold": config.default_similarity_threshold,
        "adaptive_threshold": config.default_similarity_threshold,
        "genuine_count": 0,
        "impostor_count": 0,
    }
    if len(embeddings) < 2:
        return stats

    matrix = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(matrix)
    sims = matrix @ matrix.T
    genuine: list[float] = []
    impostor: list[float] = []
    labels = [str(item.get("track_id", item.get("person_id", ""))) for item in metadata]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] and labels[i] == labels[j]:
                genuine.append(float(sims[i, j]))
            else:
                impostor.append(float(sims[i, j]))

    stats["genuine_count"] = len(genuine)
    stats["impostor_count"] = len(impostor)
    if genuine:
        stats["genuine_mean"] = float(np.mean(genuine))
        stats["genuine_p05"] = float(np.percentile(genuine, 5))
    if impostor:
        stats["impostor_mean"] = float(np.mean(impostor))
        stats["impostor_p95"] = float(np.percentile(impostor, 95))

    if genuine and impostor:
        threshold = (float(np.percentile(genuine, 5)) + float(np.percentile(impostor, 95))) / 2.0
        stats["adaptive_threshold"] = float(np.clip(threshold + config.adaptive_threshold_margin, 0.35, 0.95))
    elif impostor:
        stats["adaptive_threshold"] = float(np.clip(np.percentile(impostor, 99) + config.adaptive_threshold_margin, 0.35, 0.95))
    return stats


def save_json(path: Path, data: Any) -> None:
    """Write JSON atomically enough for this local pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def load_threshold(config: PipelineConfig = CONFIG) -> float:
    """Load the calibrated threshold, falling back to the configured default."""
    activate_video_cache(config.video_path, config)
    if config.threshold_stats_file.exists():
        with config.threshold_stats_file.open("r", encoding="utf-8") as file:
            stats = json.load(file)
        return float(stats.get("adaptive_threshold", config.default_similarity_threshold))
    return config.default_similarity_threshold


def image_to_bgr(path: Path) -> np.ndarray:
    """Read an image as BGR with a clear error."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to OpenCV BGR."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def now_seconds() -> float:
    """Small wrapper to make timing calls easy to read."""
    return time.perf_counter()


def detection_to_dict(detection: FaceDetection | None) -> dict[str, Any] | None:
    """Serialize a detection dataclass for metadata."""
    return asdict(detection) if detection is not None else None
