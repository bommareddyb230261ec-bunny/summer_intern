"""Evaluate face retrieval quality from the saved FAISS database."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import faiss
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CONFIG, activate_video_cache
from face_recognition_utils import load_threshold, save_json


def load_embeddings_and_metadata() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load all vectors and metadata from the saved database."""
    activate_video_cache(CONFIG.video_path)
    index = faiss.read_index(str(CONFIG.faiss_index_file))
    with CONFIG.metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata counts do not match.")

    embeddings = np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings, metadata


def pairwise_scores(embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pairwise same-person labels, cosine similarities, and Euclidean distances."""
    labels: list[int] = []
    similarities: list[float] = []
    distances: list[float] = []
    identities = [str(item.get("track_id", item.get("person_id", ""))) for item in metadata]

    for i, j in combinations(range(len(metadata)), 2):
        same = int(bool(identities[i]) and identities[i] == identities[j])
        similarity = float(np.dot(embeddings[i], embeddings[j]))
        labels.append(same)
        similarities.append(similarity)
        distances.append(float(np.linalg.norm(embeddings[i] - embeddings[j])))

    return np.asarray(labels), np.asarray(similarities), np.asarray(distances)


def evaluate() -> dict[str, Any]:
    """Compute classification and retrieval-threshold metrics."""
    embeddings, metadata = load_embeddings_and_metadata()
    y_true, similarities, distances = pairwise_scores(embeddings, metadata)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation needs at least one positive and one negative pair.")

    threshold = load_threshold(CONFIG)
    y_pred = (similarities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, similarities)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, similarities)

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, similarities)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "far": float(fp / max(fp + tn, 1)),
        "frr": float(fn / max(fn + tp, 1)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "cosine_similarity_distribution": {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
        },
        "euclidean_distance_distribution": {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
        },
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "precision_recall_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
    }
    return metrics


def main() -> None:
    """Run evaluation and write metrics JSON."""
    activate_video_cache(CONFIG.video_path)
    paths = activate_video_cache(CONFIG.video_path)
    metrics = evaluate()
    output = paths["evaluation_metrics"]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, metrics)
    print(json.dumps({key: metrics[key] for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "far", "frr")}, indent=2))
    print(f"Saved full metrics: {output}")


if __name__ == "__main__":
    main()
