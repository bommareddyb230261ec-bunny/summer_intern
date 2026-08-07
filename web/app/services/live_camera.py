import base64
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from app.pipeline import _ensure_project_root_in_path


def _load_arcface_modules() -> tuple[Any, Any]:
    _ensure_project_root_in_path()
    from arcface import query_matching
    from arcface.config import CONFIG
    from arcface.face_recognition_utils import (
        align_face,
        configure_logging,
        crop_with_padding,
        detect_faces_with_fallback,
        image_to_bgr,
    )

    return query_matching, CONFIG, align_face, configure_logging, crop_with_padding, detect_faces_with_fallback, image_to_bgr


class LiveCameraManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._query_embedding: np.ndarray | None = None
        self._query_preview_path: Path | None = None
        self._status = "idle"
        self._person_status = "idle"
        self._face_status = "idle"
        self._match_status = "waiting"
        self._similarity = 0.0
        self._timestamp = ""
        self._last_frame_b64: str | None = None
        self._history: list[dict[str, Any]] = []
        self._arcface = None
        self._yolo_person = None
        self._yolo_face = None
        self._retinaface = None
        self._index = None
        self._metadata = None
        self._threshold = 0.0
        self._query_uploaded = False
        self._error: str | None = None
        self._frame_interval = 0.2
        self._detector_busy = False

    def save_query_file(self, file: Any) -> Path:
        _ensure_project_root_in_path()
        _, CONFIG, _, _, _, _, _ = _load_arcface_modules()
        target = CONFIG.base_dir / "live_query.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        file.file.seek(0)
        with target.open("wb") as out:
            out.write(file.file.read())

        self._query_preview_path = CONFIG.base_dir / "live_query_preview.png"
        self._query_uploaded = False
        self._query_embedding = None
        self._status = "query_ready"
        self._error = None
        self._query_image_path = target
        self._detect_query_face_and_embed(target)
        return target

    def _detect_query_face_and_embed(self, query_image_path: Path) -> None:
        query_matching, CONFIG, align_face, configure_logging, crop_with_padding, detect_faces_with_fallback, image_to_bgr = _load_arcface_modules()
        configure_logging()
        self._load_models_if_needed(reload_search=False)

        query_bgr = image_to_bgr(query_image_path)
        detections = detect_faces_with_fallback(query_bgr, self._yolo_face, self._retinaface, CONFIG)
        if not detections:
            raise RuntimeError("No face detected in query image.")

        best = max(detections, key=lambda detection: (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]))
        crop, _ = crop_with_padding(query_bgr, best.bbox, CONFIG.face_padding_ratio)
        if crop is None or crop.size == 0:
            raise RuntimeError("Query face crop is empty.")

        aligned, _ = align_face(crop, self._retinaface, CONFIG)
        self._query_embedding = query_matching.embed_aligned_faces([aligned], self._arcface, CONFIG)[0]
        if self._query_preview_path is not None:
            cv2.imwrite(str(self._query_preview_path), crop)

        self._query_uploaded = True
        self._status = "query_uploaded"
        self._match_status = "waiting"
        self._similarity = 0.0
        self._timestamp = ""
        self._history.clear()

    def _load_models_if_needed(self, reload_search: bool = True) -> None:
        query_matching, CONFIG, _, _, _, _, _ = _load_arcface_modules()
        if self._arcface is None or self._yolo_face is None or self._retinaface is None:
            self._arcface, self._yolo_face, self._retinaface = query_matching.load_models()

        if self._yolo_person is None:
            self._yolo_person = YOLO(str(CONFIG.yolo_person_model))

        if reload_search or self._index is None or self._metadata is None:
            self._index, self._metadata = query_matching.load_search_database()
            self._threshold = query_matching.load_threshold()

    def start_camera(self, source: str | int = 0) -> None:
        with self._lock:
            if not self._query_uploaded or self._query_embedding is None:
                raise RuntimeError("Upload a query face before starting the live camera.")
            if self._thread and self._thread.is_alive():
                return

            self._load_models_if_needed()
            self._stop_event.clear()
            self._capture = cv2.VideoCapture(int(source) if isinstance(source, str) and source.isdigit() else source)
            if not self._capture.isOpened():
                raise RuntimeError("Failed to open the default camera. Ensure the webcam is accessible.")

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._status = "running"
            self._error = None

    def stop_camera(self) -> None:
        with self._lock:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
            self._thread = None
            self._capture = None
            self._status = "stopped"

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._capture_and_preview()
            except Exception as exc:
                self._error = str(exc)
                self._status = "error"
                traceback.print_exc()
                break
            time.sleep(self._frame_interval)

    def _capture_and_preview(self) -> None:
        if self._capture is None:
            raise RuntimeError("Camera is not initialized.")

        ret, frame = self._capture.read()
        if not ret or frame is None:
            self._status = "error"
            self._error = "Unable to read frame from camera."
            return

        self._status = "running"
        self._timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._person_status = "processing"
        self._face_status = "processing"
        self._match_status = "processing"
        self._similarity = 0.0

        self._draw_overlay(frame)
        self._encode_frame(frame)

        if not self._detector_busy:
            self._detector_busy = True
            detection_frame = frame.copy()
            threading.Thread(target=self._detect_and_match, args=(detection_frame,), daemon=True).start()

    def _detect_and_match(self, frame: np.ndarray) -> None:
        try:
            query_matching, CONFIG, _, _, crop_with_padding, detect_faces_with_fallback, _ = _load_arcface_modules()
            self._load_models_if_needed(reload_search=False)

            person_matches = []
            self._face_status = "none"
            self._match_status = "no_match"
            best_similarity = 0.0

            results = self._yolo_person(frame, conf=CONFIG.person_confidence, imgsz=640, verbose=False)
            boxes = getattr(results[0], "boxes", None) if results else None
            if boxes is None or len(boxes) == 0:
                self._person_status = "none"
                self._draw_overlay(frame)
                self._encode_frame(frame)
                return

            self._person_status = "detected"
            person_boxes = self._extract_box_data(boxes)
            for box in person_boxes:
                x1, y1, x2, y2, score, cls_id = box[:6]
                if int(cls_id) != 0 or score < CONFIG.person_confidence:
                    continue
                x1i = max(0, int(round(x1)))
                y1i = max(0, int(round(y1)))
                x2i = min(frame.shape[1], int(round(x2)))
                y2i = min(frame.shape[0], int(round(y2)))
                person_crop = frame[y1i:y2i, x1i:x2i]
                if person_crop is None or person_crop.size == 0:
                    continue

                detections = detect_faces_with_fallback(person_crop, self._yolo_face, self._retinaface, CONFIG)
                if not detections:
                    continue
                self._face_status = "detected"
                for detection in detections:
                    face_crop, local_bbox = crop_with_padding(person_crop, detection.bbox, CONFIG.face_padding_ratio)
                    if face_crop is None or face_crop.size == 0:
                        continue
                    live_embedding = query_matching.embed_query_face(face_crop, self._arcface, self._retinaface)
                    if self._query_embedding is None:
                        continue
                    similarity = float(np.dot(self._query_embedding, live_embedding))
                    best_similarity = max(best_similarity, similarity)
                    if similarity >= self._threshold:
                        self._match_status = "match_found"
                        self._timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        global_bbox = (
                            x1i + local_bbox[0],
                            y1i + local_bbox[1],
                            x1i + local_bbox[2],
                            y1i + local_bbox[3],
                        )
                        face_data = self._encode_face_crop(face_crop)
                        person_matches.append({
                            "bbox": global_bbox,
                            "similarity": similarity,
                            "label": "Live match",
                            "timestamp": self._timestamp,
                            "face_data": face_data,
                        })

            if not person_matches:
                full_frame_detections = detect_faces_with_fallback(frame, self._yolo_face, self._retinaface, CONFIG)
                if full_frame_detections:
                    self._face_status = "detected"
                    for detection in full_frame_detections:
                        face_crop, local_bbox = crop_with_padding(frame, detection.bbox, CONFIG.face_padding_ratio)
                        if face_crop is None or face_crop.size == 0:
                            continue
                        live_embedding = query_matching.embed_query_face(face_crop, self._arcface, self._retinaface)
                        if self._query_embedding is None:
                            continue
                        similarity = float(np.dot(self._query_embedding, live_embedding))
                        best_similarity = max(best_similarity, similarity)
                        if similarity >= self._threshold:
                            self._match_status = "match_found"
                            self._timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            face_data = self._encode_face_crop(face_crop)
                            person_matches.append({
                                "bbox": (
                                    int(local_bbox[0]),
                                    int(local_bbox[1]),
                                    int(local_bbox[2]),
                                    int(local_bbox[3]),
                                ),
                                "similarity": similarity,
                                "label": "Live match",
                                "timestamp": self._timestamp,
                                "face_data": face_data,
                            })

            if person_matches:
                best_match = max(person_matches, key=lambda item: item["similarity"])
                self._similarity = best_match["similarity"]
                self._match_status = "match_found"
                self._history.insert(0, best_match)
                self._history = self._history[:10]
                self._draw_matches(frame, person_matches)
            else:
                self._similarity = best_similarity
                self._draw_overlay(frame)

            self._encode_frame(frame)
        except Exception as exc:
            self._error = str(exc)
            self._status = "error"
            traceback.print_exc()
        finally:
            self._detector_busy = False

    def _extract_box_data(self, boxes: Any) -> np.ndarray:
        try:
            return boxes.data.cpu().numpy()
        except Exception:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            return np.column_stack([xyxy, confs, cls_ids])

    def _draw_overlay(self, frame: np.ndarray) -> None:
        status_text = "MATCH FOUND" if self._match_status == "match_found" else "NO MATCH"
        color = (20, 200, 30) if self._match_status == "match_found" else (220, 40, 40)
        cv2.putText(frame, "Live Camera", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Person: {self._person_status}", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Face: {self._face_status}", (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Result: {status_text}", (14, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sim: {self._similarity:.3f}", (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Time: {self._timestamp}", (14, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    def _draw_matches(self, frame: np.ndarray, matches: list[dict[str, Any]]) -> None:
        for match in matches:
            x1, y1, x2, y2 = [int(round(v)) for v in match["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 200, 30), 3)
            cv2.putText(
                frame,
                f"MATCH {match['similarity']:.3f}",
                (max(8, x1), max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 200, 30),
                2,
                cv2.LINE_AA,
            )
        self._draw_overlay(frame)

    def _encode_frame(self, frame: np.ndarray) -> None:
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            self._last_frame_b64 = None
            return
        self._last_frame_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _encode_face_crop(self, face_crop: np.ndarray) -> str | None:
        success, encoded = cv2.imencode(".jpg", face_crop)
        if not success:
            return None
        return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('utf-8')}"

    def status_payload(self) -> dict[str, Any]:
        return {
            "camera_status": self._status,
            "person_status": self._person_status,
            "face_status": self._face_status,
            "match_status": self._match_status,
            "similarity": round(self._similarity, 4),
            "timestamp": self._timestamp,
            "query_uploaded": self._query_uploaded,
            "error": self._error,
            "history": self._history,
        }

    def frame_payload(self) -> dict[str, Any]:
        return {
            **self.status_payload(),
            "frame_data": f"data:image/jpeg;base64,{self._last_frame_b64}" if self._last_frame_b64 else None,
        }


live_camera_manager = LiveCameraManager()
