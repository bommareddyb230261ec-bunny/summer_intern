import asyncio
import logging
from pathlib import Path
from urllib.parse import quote_plus

from authlib.integrations.base_client.errors import OAuthError
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    OAuthProfileError,
    create_access_token,
    get_current_user,
    normalize_google_profile,
    oauth,
    upsert_google_user,
)
from app.config import settings
from app.database import get_db
from app.models import User
from app.schemas import (
    AuthResponse,
    ErrorResponse,
    JobResponse,
    MessageResponse,
    ProcessStartRequest,
    ProcessStatusResponse,
    ResultItem,
    ResultsResponse,
    UserResponse,
)
from app.pipeline import save_query_file, save_video_files, run_job
from app.services.job_manager import job_manager

router = APIRouter()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCFACE_DIR = PROJECT_ROOT / "arcface"
FACE_SEARCH_DIRS = (
    ARCFACE_DIR / "detected_faces",
    ARCFACE_DIR / "cropped_faces",
    ARCFACE_DIR / "aligned_faces",
    ARCFACE_DIR / "video_cache",
    ARCFACE_DIR / "results" / "faces",
    ARCFACE_DIR / "output" / "faces",
)


def job_not_found_response(job_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "code": "JOB_NOT_FOUND",
                "message": "Job not found.",
                "job_id": job_id,
            }
        },
    )


def resolve_pipeline_image_url(filename_or_path: str | None) -> str | None:
    """Resolve an ArcFace output filename/path to the mounted static URL."""

    if not filename_or_path:
        return None

    normalized = str(filename_or_path).replace("\\", "/").strip()
    if normalized.startswith(("http://", "https://", "/static/")):
        return normalized

    candidate = Path(normalized)
    if candidate.is_absolute() and candidate.exists():
        try:
            relative = candidate.resolve().relative_to(ARCFACE_DIR.resolve())
            return f"/static/pipeline/{relative.as_posix()}"
        except ValueError:
            logger.warning("Matched image is outside static pipeline directory: %s", candidate)
            return None

    for directory in FACE_SEARCH_DIRS:
        direct = directory / Path(normalized).name
        if direct.exists():
            relative = direct.resolve().relative_to(ARCFACE_DIR.resolve())
            return f"/static/pipeline/{relative.as_posix()}"

    if ARCFACE_DIR.exists():
        matches = list(ARCFACE_DIR.rglob(Path(normalized).name))
        if matches:
            relative = matches[0].resolve().relative_to(ARCFACE_DIR.resolve())
            return f"/static/pipeline/{relative.as_posix()}"

    logger.warning("Matched face image not found for filename/path: %s", filename_or_path)
    return None


@router.get("/", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def root() -> MessageResponse:
    """Health-style root endpoint that avoids exposing secrets or connection URLs."""

    return MessageResponse(message="Google OAuth FastAPI service is running.")


@router.get(
    "/login",
    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    responses={307: {"description": "Redirect to Google's OAuth consent screen."}},
)
async def login(request: Request) -> RedirectResponse:
    """Start Google OAuth and let Authlib handle the state/nonce exchange."""

    google_client = getattr(oauth, "google")
    return await google_client.authorize_redirect(request, settings.REDIRECT_URI)


@router.get(
    "/auth/callback",
    response_model=AuthResponse,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def auth_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Handle Google's callback, persist the user, and issue a JWT."""

    if request.query_params.get("error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google login failed: {request.query_params.get('error')}",
        )

    if not request.query_params.get("code"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code from Google.",
        )

    try:
        google_client = getattr(oauth, "google")
        token = await google_client.authorize_access_token(request)
        user_info = token.get("userinfo")
        if user_info is None:
            user_info = await google_client.parse_id_token(request, token)

        profile = normalize_google_profile(dict(user_info))
        user = await upsert_google_user(db, profile)
    except OAuthProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except OAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Google OAuth response: {exc.error}",
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this Google account or email already exists.",
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while processing Google login.",
        ) from exc

    access_token = create_access_token(user)
    response_payload = AuthResponse(
        message="Login successful.",
        user=UserResponse.model_validate(user),
        access_token=access_token,
        token_type="bearer",
    )

    accept_header = request.headers.get("accept", "").lower()
    if "text/html" in accept_header:
        redirect_url = (
            f"{settings.FRONTEND_URL}/auth/callback?access_token={quote_plus(access_token)}"
        )
        return RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)

    return JSONResponse(content=response_payload.model_dump(mode="json"))


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout() -> JSONResponse:
    """Clear the JWT cookie so the browser session is removed."""

    response = JSONResponse(content=MessageResponse(message="Logout successful.").model_dump())
    response.delete_cookie(key="access_token", path="/")
    return response


@router.get(
    "/profile",
    response_model=UserResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def profile(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user's profile from PostgreSQL."""

    return UserResponse.model_validate(current_user)


@router.post(
    "/upload/query",
    response_model=JobResponse,
    responses={401: {"model": ErrorResponse}},
)
async def upload_query(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    if file.filename == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query face file is required.",
        )

    job = job_manager.create_job(
        stage="QUERY_UPLOAD",
        message="Query face upload received.",
    )
    job_id = job["job_id"]
    try:
        saved = save_query_file(file, job_id)
        job_manager.update_job(
            job_id,
            progress=10,
            stage="QUERY_UPLOADED",
            message=f"Query face uploaded and saved: {saved.name}",
            query_path=str(saved),
        )
    except Exception as exc:
        job_manager.mark_failed(job_id, "Failed to save query face.", str(exc))
        raise

    return JobResponse(job_id=job_id, message=f"Query face uploaded and saved: {saved.name}")


@router.post(
    "/upload/videos",
    response_model=JobResponse,
    responses={401: {"model": ErrorResponse}},
)
async def upload_videos(
    job_id: str = Form(...),
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    if job_manager.get_job(job_id) is None:
        return job_not_found_response(job_id)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one video file is required.",
        )
    try:
        saved_paths = save_video_files(files, job_id)
        job_manager.update_job(
            job_id,
            progress=25,
            stage="VIDEOS_UPLOADED",
            message=f"Videos uploaded: {[p.name for p in saved_paths]}",
            video_uploaded=1,
            video_paths=[str(p) for p in saved_paths],
        )
    except Exception as exc:
        job_manager.mark_failed(job_id, "Failed to save uploaded videos.", str(exc))
        raise

    return JobResponse(job_id=job_id, message=f"Videos uploaded: {[p.name for p in saved_paths]}")


@router.post(
    "/process/start",
    response_model=JobResponse,
    responses={401: {"model": ErrorResponse}},
)
async def start_processing(
    payload: ProcessStartRequest,
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    job = job_manager.get_job(payload.job_id)
    if job is None:
        return job_not_found_response(payload.job_id)

    if not job.get("video_uploaded"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload videos before starting processing.",
        )
    job_manager.update_job(
        payload.job_id,
        status="QUEUED",
        progress=5,
        stage="QUEUED",
        message="Processing job accepted and queued.",
        results_ready=0,
        result=None,
    )
    asyncio.create_task(run_job(payload.job_id, job_manager))

    return JobResponse(job_id=payload.job_id, message="Processing started in background.")


@router.get(
    "/process/status/{job_id}",
    response_model=ProcessStatusResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def process_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> ProcessStatusResponse:
    payload = job_manager.status_payload(job_id)
    if payload is None:
        return job_not_found_response(job_id)
    return ProcessStatusResponse(**payload)


@router.get(
    "/results/{job_id}",
    response_model=ResultsResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_results(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> ResultsResponse:
    job = job_manager.get_job(job_id)
    if job is None or not job.get("results_ready"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results are not ready for the specified job.",
        )
    # Prefer results persisted by the pipeline
    results_path = job.get("results_path")
    results_list = []
    if results_path:
        import json

        try:
            with open(results_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            matches = payload.get("matched", []) or payload.get("results", [])
            for item in matches:
                # Map arcface item to ResultItem schema
                face_id = item.get("face_image") or item.get("face_id") or str(item.get("rank", ""))
                label = item.get("video_name") or str(item.get("person_id", ""))
                similarity = float(item.get("cosine_similarity") or item.get("similarity") or 0.0)
                timestamp = str(item.get("timestamp", ""))
                matched_face_image = resolve_pipeline_image_url(
                    item.get("matched_face_image")
                    or item.get("face_image_path")
                    or item.get("face_image")
                    or item.get("face_image_url")
                    or item.get("image_path")
                    or item.get("crop_path")
                    or item.get("person_crop_path")
                    or face_id
                )
                results_list.append({
                    "face_id": face_id,
                    "label": label,
                    "similarity": similarity,
                    "timestamp": timestamp,
                    "matched_face_image": matched_face_image,
                    "frame_name": item.get("frame_name") or item.get("frame_id") or item.get("frame_image") or timestamp,
                    "bounding_box": item.get("bounding_box") or item.get("bbox") or item.get("box") or item.get("face_bbox"),
                })
        except Exception:
            # Fall back to in-memory results if read fails
            result_payload = job.get("result") or {}
            results_list = result_payload.get("matched", []) or result_payload.get("results", [])

    return ResultsResponse(job_id=job_id, status=job.get("status", "completed"), results=results_list)
