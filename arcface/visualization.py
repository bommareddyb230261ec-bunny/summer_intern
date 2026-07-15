"""Generate research-style plots for the face retrieval database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve

from config import CONFIG, activate_video_cache
from evaluation import pairwise_scores
from face_recognition_utils import save_json


def load_database() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load embeddings and metadata."""
    index = faiss.read_index(str(CONFIG.faiss_index_file))
    with CONFIG.metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    embeddings = np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings, metadata


def save_current_plot(name: str) -> Path:
    """Save and close the active Matplotlib figure."""
    CONFIG.visualization_dir.mkdir(parents=True, exist_ok=True)
    path = CONFIG.visualization_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_roc(y_true: np.ndarray, similarities: np.ndarray) -> Path:
    """Generate ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, similarities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    return save_current_plot("roc_curve.png")


def plot_precision_recall(y_true: np.ndarray, similarities: np.ndarray) -> Path:
    """Generate precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, similarities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    return save_current_plot("precision_recall_curve.png")


def plot_similarity_histogram(y_true: np.ndarray, similarities: np.ndarray) -> Path:
    """Generate same/different similarity histogram."""
    plt.figure(figsize=(7, 5))
    plt.hist(similarities[y_true == 1], bins=30, alpha=0.7, label="Same person")
    plt.hist(similarities[y_true == 0], bins=30, alpha=0.7, label="Different person")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Pair Count")
    plt.title("Similarity Histogram")
    plt.legend()
    return save_current_plot("similarity_histogram.png")


def plot_embedding_projection(embeddings: np.ndarray, metadata: list[dict[str, Any]], method: str) -> Path:
    """Generate PCA or t-SNE embedding projection."""
    labels = [str(item.get("track_id", item.get("person_id", ""))) for item in metadata]
    if method == "pca":
        points = PCA(n_components=2).fit_transform(embeddings)
        filename = "embedding_pca.png"
        title = "Embedding PCA"
    else:
        perplexity = max(2, min(30, len(embeddings) - 1))
        points = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(embeddings)
        filename = "embedding_tsne.png"
        title = "Embedding t-SNE"

    plt.figure(figsize=(7, 6))
    for label in sorted(set(labels)):
        mask = np.asarray(labels) == label
        plt.scatter(points[mask, 0], points[mask, 1], s=28, label=label)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if len(set(labels)) <= 12:
        plt.legend(fontsize=8)
    return save_current_plot(filename)


def plot_top_k_retrieval(query_path: Path, matches: list[dict[str, Any]], output_name: str = "top_k_retrieval.png") -> Path:
    """Create a contact sheet for a query and its ranked matches."""
    images: list[tuple[str, np.ndarray]] = []
    query = cv2.imread(str(query_path))
    if query is not None:
        images.append(("Query", cv2.cvtColor(query, cv2.COLOR_BGR2RGB)))
    for match in matches[:CONFIG.top_k]:
        image = cv2.imread(str(match["face_image_path"]))
        if image is not None:
            title = f"#{match.get('rank', '?')} sim={match.get('cosine_similarity', 0):.3f}"
            images.append((title, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    if not images:
        raise ValueError("No images available for Top-K retrieval figure.")

    cols = min(5, len(images))
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, (title, image) in enumerate(images, start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    return save_current_plot(output_name)


def main() -> None:
    """Generate database-level visualizations."""
    activate_video_cache(CONFIG.video_path)
    embeddings, metadata = load_database()
    y_true, similarities, _ = pairwise_scores(embeddings, metadata)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Visualization needs at least one positive and one negative pair.")

    outputs = [
        plot_roc(y_true, similarities),
        plot_precision_recall(y_true, similarities),
        plot_similarity_histogram(y_true, similarities),
        plot_embedding_projection(embeddings, metadata, "pca"),
    ]
    if len(embeddings) > 2:
        outputs.append(plot_embedding_projection(embeddings, metadata, "tsne"))

    save_json(CONFIG.visualization_dir / "visualization_outputs.json", [str(path) for path in outputs])
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
