import asyncio
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.services.job_manager import JobManager


logger = logging.getLogger(__name__)


def _ensure_project_root_in_path() -> Path:
    """Ensure the repository root is on sys.path and return the project root."""
    here = Path(__file__).resolve()
    # web/app -> parents[2] -> project root (NSG AI Surveillance Dashboard)
    project_root = here.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Also ensure arcface folder itself is on sys.path so arcface modules can do `from config import CONFIG`.
    arcface_dir = project_root / "arcface"
    if str(arcface_dir) not in sys.path:
        sys.path.insert(0, str(arcface_dir))
    return project_root


def _load_arcface_config():
    """Load the same CONFIG module name used by the ArcFace scripts."""
    _ensure_project_root_in_path()
    from config import CONFIG

    return CONFIG


def _activate_arcface_cache(video_path: Path | None = None) -> dict[str, Path]:
    """Activate the same video cache used by embedding generation and query matching."""
    _ensure_project_root_in_path()
    from config import activate_video_cache

    return activate_video_cache(video_path)


def _write_cache_status(paths: dict[str, Path], status: str, message: str, **extra: Any) -> None:
    """Persist lightweight cache status metadata for the current video."""
    payload = {
        "video_hash": paths["root"].name,
        "status": status,
        "message": message,
        **extra,
    }
    paths["status"].parent.mkdir(parents=True, exist_ok=True)
    with paths["status"].open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _cache_artifacts_complete(paths: dict[str, Path]) -> bool:
    """Determine whether the cache already contains the persisted embedding artifacts."""
    return all(
        [
            paths["embedding_db"].exists(),
            paths["faiss_index"].exists(),
            paths["metadata"].exists(),
            paths["processed_faces"].exists(),
        ]
    )


def _active_video_hash_file(config: Any) -> Path:
    return config.video_cache_dir / "active_video.json"


def _read_active_video_hash(config: Any) -> str | None:
    active_file = _active_video_hash_file(config)
    if not active_file.exists():
        return None
    try:
        with active_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return str(payload.get("video_hash")) if payload.get("video_hash") is not None else None
    except Exception:
        pass
    return None


def _write_active_video_hash(config: Any, video_hash: str) -> None:
    active_file = _active_video_hash_file(config)
    active_file.parent.mkdir(parents=True, exist_ok=True)
    with active_file.open("w", encoding="utf-8") as fh:
        json.dump({"video_hash": video_hash}, fh, indent=2, ensure_ascii=False)


def _cleanup_temp_processing_dirs(config: Any) -> None:
    """Remove temporary processing folders for a previous video run."""
    cleanup_paths = [
        config.saved_frames_dir,
        config.cropped_persons_dir,
        config.cropped_faces_dir,
        config.base_dir / "aligned_faces",
        config.base_dir / "detected_faces",
    ]
    for path in cleanup_paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def _cleanup_previous_temp_dirs_if_new_video(config: Any, current_video_hash: str, cache_paths: dict[str, Path]) -> None:
    """Delete old temporary folders only when a different video begins processing."""
    previous_hash = _read_active_video_hash(config)
    current_is_cached = _cache_artifacts_complete(cache_paths)

    if previous_hash is None and current_is_cached:
        # If we already have a cache for this hash, keep temp folders for reuse.
        _write_active_video_hash(config, current_video_hash)
        return

    if previous_hash is not None and previous_hash == current_video_hash:
        _write_active_video_hash(config, current_video_hash)
        return

    # New video upload detected; clear old temporary processing folders.
    _cleanup_temp_processing_dirs(config)
    _write_active_video_hash(config, current_video_hash)


def save_query_file(file: UploadFile, job_id: str) -> Path:
    """Save uploaded query image to arcface expected location (CONFIG.query_image_path)."""
    CONFIG = _load_arcface_config()

    target = CONFIG.base_dir / CONFIG.query_image_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    file.file.seek(0)
    with target.open("wb") as out:
        out.write(file.file.read())
    return target


def save_video_files(files: list[UploadFile], job_id: str) -> list[Path]:
    """Save uploaded videos to the arcface base directory."""
    CONFIG = _load_arcface_config()

    saved: list[Path] = []
    for upload in files:
        target = CONFIG.base_dir / upload.filename
        upload.file.seek(0)
        with target.open("wb") as out:
            out.write(upload.file.read())
        saved.append(target)
    return saved


def _write_results_to_web_app(job_id: str, results: Any) -> Path:
    """Persist results into web/app/data/results_<job_id>.json for the API to read."""
    project_root = _ensure_project_root_in_path()
    data_dir = project_root / "web" / "app" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"results_{job_id}.json"
    import json

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    return out_path


def _update_job(
    job_manager: JobManager,
    job_id: str,
    *,
    status: str = "RUNNING",
    progress: int,
    stage: str,
    message: str,
    **extra: Any,
) -> None:
    job_manager.update_job(
        job_id,
        status=status,
        progress=progress,
        stage=stage,
        message=message,
        **extra,
    )


def run_job_sync(job_id: str, job_manager: JobManager) -> None:
    """Synchronous runner for the ArcFace pipeline with per-video caching and cleanup."""
    _ensure_project_root_in_path()
    try:
        from arcface import video_to_frames, person_cropping, face_cropping, master_embedding_database, query_matching
        from arcface.face_recognition_utils import configure_logging, load_threshold
        CONFIG = _load_arcface_config()

        configure_logging()

        job_state = job_manager.get_job(job_id) or {}
        video_paths = job_state.get("video_paths") or []
        if not video_paths:
            raise FileNotFoundError("No uploaded video path found for this job.")

        first_video = Path(video_paths[0]).expanduser()
        if not first_video.exists():
            raise FileNotFoundError(f"Uploaded video file not found: {first_video}")

        object.__setattr__(CONFIG, "video_path", first_video.resolve())
        cache_paths = _activate_arcface_cache(CONFIG.video_path)
        cache_paths["root"].mkdir(parents=True, exist_ok=True)
        cache_paths["embeddings_dir"].mkdir(parents=True, exist_ok=True)
        cache_paths["faiss_dir"].mkdir(parents=True, exist_ok=True)
        _cleanup_previous_temp_dirs_if_new_video(CONFIG, cache_paths["root"].name, cache_paths)
        _write_cache_status(
            cache_paths,
            "processing",
            "Preparing video cache.",
            video_name=first_video.name,
            video_path=str(first_video.resolve()),
        )
        logger.info("[PIPELINE START] job_id=%s video_path=%s query_image=%s", job_id, CONFIG.video_path, CONFIG.query_image_path)

        for folder in (
            CONFIG.saved_frames_dir,
            CONFIG.cropped_persons_dir,
            CONFIG.cropped_faces_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        if _cache_artifacts_complete(cache_paths):
            _update_job(
                job_manager,
                job_id,
                progress=20,
                stage="CACHE_REUSE",
                message="Reusing cached embeddings and FAISS index for this video.",
            )
            _write_cache_status(
                cache_paths,
                "ready",
                "Cached artifacts found; skipping frame extraction and embedding generation.",
                video_name=first_video.name,
            )
        else:
            _update_job(
                job_manager,
                job_id,
                progress=10,
                stage="PROCESSING_VIDEO",
                message="Building or resuming the video cache.",
            )

            if not any(CONFIG.saved_frames_dir.iterdir()):
                _update_job(
                    job_manager,
                    job_id,
                    progress=15,
                    stage="FRAME_EXTRACTION",
                    message="Extracting frames from uploaded video.",
                )
                video_to_frames.main()
            else:
                logger.info("[FRAME EXTRACTION SKIPPED] job_id=%s cache already has saved frames", job_id)

            if not any(CONFIG.cropped_persons_dir.iterdir()):
                _update_job(
                    job_manager,
                    job_id,
                    progress=35,
                    stage="PERSON_DETECTION",
                    message="Detecting persons.",
                )
                _update_job(
                    job_manager,
                    job_id,
                    progress=50,
                    stage="PERSON_CROPPING",
                    message="Cropping detected persons.",
                )
                person_cropping.main()
                logger.info("[DETECTION COMPLETE] job_id=%s", job_id)
            else:
                logger.info("[PERSON CROPPING SKIPPED] job_id=%s cache already has person crops", job_id)

            if not any(CONFIG.cropped_persons_dir.iterdir()):
                raise FileNotFoundError(f"No person crops generated in {CONFIG.cropped_persons_dir}")

            if not any(CONFIG.cropped_faces_dir.iterdir()):
                _update_job(
                    job_manager,
                    job_id,
                    progress=65,
                    stage="FACE_DETECTION",
                    message="Detecting faces.",
                )
                _update_job(
                    job_manager,
                    job_id,
                    progress=80,
                    stage="FACE_ALIGNMENT",
                    message="Aligning faces.",

                )
                face_cropping.main()
                logger.info("[ALIGNMENT COMPLETE] job_id=%s", job_id)
            else:
                logger.info("[FACE CROPPING SKIPPED] job_id=%s cache already has face crops", job_id)

            if not any(CONFIG.cropped_faces_dir.iterdir()):
                raise FileNotFoundError(f"No face crops generated in {CONFIG.cropped_faces_dir}")

            _update_job(
                job_manager,
                job_id,
                progress=90,
                stage="EMBEDDING_GENERATION",
                message="Generating ArcFace embeddings.",
            )
            master_embedding_database.main()
            logger.info("[EMBEDDING COMPLETE] job_id=%s", job_id)
            logger.info("[FAISS COMPLETE] job_id=%s", job_id)
            _write_cache_status(
                cache_paths,
                "completed",
                "Embeddings and FAISS index created successfully.",
                video_name=first_video.name,
            )

        _update_job(
            job_manager,
            job_id,
            progress=95,
            stage="FAISS_MATCHING",
            message="Matching query face against the FAISS index.",
        )
        arcface_model, yolo, retinaface = query_matching.load_models()
        index, metadata = query_matching.load_search_database()
        threshold = load_threshold()

        from arcface.face_recognition_utils import image_to_bgr

        query_bgr = image_to_bgr(CONFIG.query_image_path)
        query_face, query_detection = query_matching.detect_query_face(query_bgr, yolo, retinaface)
        query_embedding = query_matching.embed_query_face(query_face, arcface_model, retinaface)

        results = query_matching.search(query_embedding, index, metadata, top_k=CONFIG.top_k)
        matched = query_matching.apply_adaptive_threshold(results, threshold)
        best_similarity = max((item.get("cosine_similarity", 0.0) for item in results), default=0.0)

        _write_cache_status(
            cache_paths,
            "completed",
            "Video cache is ready for future uploads.",
            video_name=first_video.name,
        )

        _update_job(
            job_manager,
            job_id,
            progress=98,
            stage="SAVING_RESULTS",
            message="Finalizing results.",
        )
        output = {
            "job_id": job_id,
            "status": "completed",
            "best_similarity": best_similarity,
            "results": results,
            "matched": matched,
            "query_detection": query_detection,
            "query_face_crop_path": query_detection.get("crop_path"),
            "query_face_preview_path": query_detection.get("preview_path"),
        }
        out_path = _write_results_to_web_app(job_id, output)
        job_manager.update_job(
            job_id,
            status="COMPLETED",
            progress=100,
            stage="COMPLETED",
            message="Pipeline completed successfully.",
            result=output,
            results_ready=1,
            results_path=str(out_path.resolve()),
        )
        logger.info("[JOB FINISHED] job_id=%s", job_id)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("[JOB FINISHED] job_id=%s failed", job_id)
        job_manager.update_job(
            job_id,
            status="FAILED",
            stage="FAILED",
            message=str(exc),
            error=tb,
        )


async def run_job(job_id: str, job_manager: JobManager) -> None:
    """Async wrapper that runs the synchronous pipeline in a thread."""
    job_manager.update_job(
        job_id,
        status="QUEUED",
        progress=5,
        stage="QUEUED",
        message="Upload complete. Pipeline queued.",
    )
    await asyncio.sleep(0.1)
    job_manager.update_job(
        job_id,
        status="RUNNING",
        progress=8,
        stage="STARTING",
        message="Pipeline starting.",
    )
    await asyncio.to_thread(run_job_sync, job_id, job_manager)
