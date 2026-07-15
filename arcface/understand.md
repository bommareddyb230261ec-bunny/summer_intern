# Face Re-Identification Project Audit

## Project Overview

This repository implements a face re-identification pipeline that starts from a video file and ends with a FAISS-based face retrieval system. The pipeline is designed to extract frames from a video, detect people, crop person regions, detect faces, align faces, generate ArcFace embeddings, index them in FAISS, and retrieve matching identities for a query image.

The implementation is a local, file-based prototype rather than a production-ready deployment system. It uses Python scripts, OpenCV, InsightFace, Ultralytics YOLO, PyTorch, and FAISS.

## Objectives

The repository aims to:

- build a searchable database of face embeddings from a video source;
- match a query face against the database;
- reject unknown identities using an adaptive similarity threshold;
- produce evaluation metrics and visualization plots for analysis.

## Features

The current code implements the following capabilities:

- video frame extraction with fixed sampling interval;
- person detection using YOLO11;
- face detection using YOLOv8-face with RetinaFace fallback;
- face alignment using RetinaFace landmarks when available;
- ArcFace embedding generation in batches;
- FAISS indexing with L2-normalized embeddings;
- metadata persistence for each embedding;
- threshold calibration from same/different person similarity statistics;
- query-image retrieval with Top-K ranking;
- unknown-person rejection through thresholding;
- evaluation metrics and visualization plots.

## Folder Structure

The repository layout is as follows:

- [config.py](config.py): central configuration dataclass and default paths.
- [video_to_frames.py](video_to_frames.py): extracts sampled frames from the source video.
- [person_cropping.py](person_cropping.py): crops detected person regions from frames.
- [face_cropping.py](face_cropping.py): crops detected faces inside person crops and writes metadata.
- [face_recognition_utils.py](face_recognition_utils.py): shared face detection, alignment, embedding, and threshold utilities.
- [master_embedding_database.py](master_embedding_database.py): builds the FAISS index and metadata database.
- [query_matching.py](query_matching.py): runs retrieval against the FAISS database for a query image.
- [evaluation.py](evaluation.py): computes evaluation metrics from the stored embeddings.
- [visualization.py](visualization.py): generates ROC, precision-recall, histogram, and embedding projection plots.
- [requirements.txt](requirements.txt): Python dependencies.
- [saved_frames/](saved_frames): extracted frames from the video.
- [cropped_persons/](cropped_persons): person crops produced by YOLO11.
- [cropped_faces/](cropped_faces): face crops produced by the face detector.
- [embeddings/](embeddings): FAISS index, metadata JSON, threshold stats, evaluation metrics, and visualization outputs.

## Models Used

The project uses five major model components:

### YOLO11

- Purpose: person detection in each frame.
- Why selected: the code uses it as the first-stage detector to isolate person regions before face analysis.
- Implementation: loaded from [config.py](config.py) through the file [person_cropping.py](person_cropping.py).

### YOLOv8-Face

- Purpose: primary face detection inside person crops.
- Why selected: the code uses it as a fast first-pass detector for faces.
- Implementation: loaded through [face_cropping.py](face_cropping.py) and [face_recognition_utils.py](face_recognition_utils.py).

### RetinaFace

- Purpose: fallback face detector and landmark provider.
- Why selected: the code uses it when YOLO misses or produces weak detections, and it provides 5-point landmarks for alignment.
- Implementation: loaded via InsightFace FaceAnalysis in [face_recognition_utils.py](face_recognition_utils.py).

### ArcFace

- Purpose: face embedding generation.
- Why selected: the code uses ArcFace to produce discriminative 512-dimensional embeddings for re-identification.
- Implementation: loaded from ONNX weights through InsightFace in [face_recognition_utils.py](face_recognition_utils.py).

### FAISS

- Purpose: fast nearest-neighbor retrieval over the embedding database.
- Why selected: it provides efficient similarity search over large embedding sets.
- Implementation: [master_embedding_database.py](master_embedding_database.py) builds an IndexFlatL2 database; [query_matching.py](query_matching.py) performs search.

## Complete Pipeline

The implemented pipeline is:

```text
Input video
   |
   v
Frame extraction (video_to_frames.py)
   |
   v
Person detection (person_cropping.py)
   |
   v
Face detection and crop saving (face_cropping.py)
   |
   v
Face alignment + ArcFace embedding (master_embedding_database.py)
   |
   v
FAISS index + metadata JSON + threshold stats
   |
   v
Query image -> detect face -> embed -> FAISS search -> threshold filter
   |
   v
Retrieval result / unknown-person decision
```

### Stage-by-stage explanation

1. Frame extraction
   - The script [video_to_frames.py](video_to_frames.py) reads the input video and writes frames at a fixed interval.
   - The sampling interval is controlled by the config value `frame_sample_seconds`.

2. Person cropping
   - [person_cropping.py](person_cropping.py) loads YOLO11 and runs inference on each extracted frame.
   - Detections with class 0 and confidence above the configured threshold are saved as person crops.

3. Face cropping
   - [face_cropping.py](face_cropping.py) loads YOLOv8-face and RetinaFace.
   - It processes each person crop and saves face crops with padding.
   - Detection metadata is stored in [cropped_faces/face_detection_metadata.json](cropped_faces/face_detection_metadata.json).

4. Alignment and embedding
   - [master_embedding_database.py](master_embedding_database.py) parses face crop filenames to recover frame, person, and face indexes.
   - It aligns each face crop using RetinaFace landmarks when available.
   - It generates ArcFace embeddings in batches and stores them in a FAISS index.

5. Query retrieval
   - [query_matching.py](query_matching.py) loads the FAISS index and metadata.
   - It detects the best face from the query image, embeds it, performs similarity search, and applies the adaptive threshold.

## Data Flow

The data flow is file-based and deterministic for this prototype:

1. A video file is read and sampled into JPEG frames.
2. Frames are passed to the YOLO11 person detector.
3. Person crops are written to [cropped_persons/](cropped_persons).
4. Person crops are passed to the face detector and face crops are written to [cropped_faces/](cropped_faces).
5. Face crops are aligned and embedded.
6. Embeddings are stored in a FAISS index, while metadata is stored in JSON.
7. At query time, a query image follows the same detection/alignment/embedding route.
8. The query embedding is compared to the FAISS database and ranked by similarity.

## Metadata Structure

Each entry in [embeddings/master_face_metadata.json](embeddings/master_face_metadata.json) contains the following fields:

- `face_image`: file name of the stored face crop;
- `face_image_path`: absolute path to the face crop;
- `person_crop`: file name of the corresponding person crop;
- `person_crop_path`: absolute path to the person crop;
- `frame_name`: name of the matched frame from the video;
- `frame_path`: absolute path to that frame;
- `frame_number`: frame index inferred from the crop filename;
- `timestamp`: timestamp in seconds inferred from the frame name;
- `video_name`: source video file name;
- `video_path`: absolute path to the source video;
- `person_id`: person index from the crop filename;
- `track_id`: same as `person_id` in the current implementation;
- `face_id`: face index from the crop filename;
- `bbox`: face bounding box inside the person crop when available;
- `embedding_dimension`: expected embedding size, currently 512;
- `embedding_model`: ArcFace model description;
- `confidence`: face detector confidence from sidecar metadata when available;
- `face_detector`: detector used for the crop;
- `alignment_model`: the RetinaFace model name used for alignment;
- `alignment_detection`: detection payload, including landmarks when present.

The metadata file is meant to remain synchronized with the FAISS index by row order.

## FAISS Database

The FAISS database is built in [master_embedding_database.py](master_embedding_database.py).

Implementation details:

- The index type is `IndexFlatL2`.
- Each embedding is L2-normalized before being added to the index.
- Query embeddings are also normalized before search.
- Similarity is derived from squared L2 distance using:

  $$\text{cosine similarity} = 1 - \frac{d^2}{2}$$

  for unit-normalized vectors.

The code explicitly validates that:

- the index dimension matches the configured embedding size;
- the number of FAISS vectors equals the number of metadata entries;
- the metadata contains the required fields.

## Query Retrieval

The retrieval process in [query_matching.py](query_matching.py) is:

1. load the FAISS index and metadata;
2. read the query image;
3. detect faces in the query image using YOLOv8-face with RetinaFace fallback;
4. select the largest face region as the query face;
5. crop with padding and write a preview image;
6. align the face and generate an ArcFace embedding;
7. search the FAISS index for Top-K nearest neighbors;
8. convert FAISS distances to cosine similarity;
9. apply the adaptive threshold to decide whether the query is a known person or an unknown person.

The script prints ranked results and shows a side-by-side visualization for the top matches.

## Adaptive Threshold

The adaptive threshold is computed in [face_recognition_utils.py](face_recognition_utils.py) and persisted to [embeddings/threshold_stats.json](embeddings/threshold_stats.json).

The implementation:

- forms pairwise cosine similarities between all embeddings;
- groups them into same-person and different-person pairs using the `track_id` field;
- computes the 5th percentile of genuine similarities and the 95th percentile of impostor similarities;
- uses their midpoint and adds the configured margin;
- clips the result to the range $[0.35, 0.95]$.

In the current generated statistics, the adaptive threshold is approximately 0.4726, which is lower than the default threshold of 0.80.

This threshold is then used to classify a query as a match or as an unknown identity.

## Evaluation Metrics

The evaluation script [evaluation.py](evaluation.py) computes the following metrics from pairwise embeddings:

- accuracy;
- precision;
- recall;
- F1 score;
- ROC AUC;
- PR AUC;
- false acceptance rate (FAR);
- false rejection rate (FRR);
- confusion matrix;
- similarity distribution statistics;
- Euclidean distance distribution statistics;
- ROC curve data;
- precision-recall curve data.

The evaluation is based on pairwise same/different-person labels derived from the `track_id` values in the metadata.

The current saved results are stored in [embeddings/evaluation_metrics.json](embeddings/evaluation_metrics.json).

## Visualization

The visualization script [visualization.py](visualization.py) generates the following outputs in [embeddings/visualizations/](embeddings/visualizations):

- ROC curve: [embeddings/visualizations/roc_curve.png](embeddings/visualizations/roc_curve.png)
- precision-recall curve: [embeddings/visualizations/precision_recall_curve.png](embeddings/visualizations/precision_recall_curve.png)
- similarity histogram: [embeddings/visualizations/similarity_histogram.png](embeddings/visualizations/similarity_histogram.png)
- PCA projection of embeddings: [embeddings/visualizations/embedding_pca.png](embeddings/visualizations/embedding_pca.png)
- t-SNE projection of embeddings: [embeddings/visualizations/embedding_tsne.png](embeddings/visualizations/embedding_tsne.png)

These outputs are useful for inspecting how separable the embeddings are.

## Configuration

The central configuration is implemented in [config.py](config.py) using a frozen dataclass named `PipelineConfig`.

Important configurable settings include:

- base directories for inputs and outputs;
- paths to the video, query image, model weights, FAISS index, metadata, threshold stats, and visualization directory;
- model names such as `w600k_r50` and `buffalo_l`;
- embedding dimension 512;
- alignment size 112;
- batch size 32;
- detection confidence thresholds;
- padding ratio and minimum crop size;
- Top-K retrieval size;
- default similarity threshold 0.80;
- adaptive threshold margin 0.02;
- frame sampling interval.

The configuration is used by every major stage and makes the pipeline easy to rerun under different settings.

## Running the Project

The intended execution order is:

```powershell
pip install -r requirements.txt
python video_to_frames.py
python person_cropping.py
python face_cropping.py
python master_embedding_database.py
python query_matching.py
python evaluation.py
python visualization.py
```

Each stage writes outputs into the project directories and relies on the previous stage’s artifacts.

## Expected Outputs

The project produces the following outputs:

- [saved_frames/](saved_frames): sampled frame images from the input video.
- [cropped_persons/](cropped_persons): person crop images detected by YOLO11.
- [cropped_faces/](cropped_faces): face crop images and face detection metadata.
- [embeddings/master_face_embeddings.faiss](embeddings/master_face_embeddings.faiss): FAISS index of normalized embeddings.
- [embeddings/master_face_metadata.json](embeddings/master_face_metadata.json): metadata synchronized with the FAISS index.
- [embeddings/threshold_stats.json](embeddings/threshold_stats.json): adaptive threshold statistics.
- [embeddings/evaluation_metrics.json](embeddings/evaluation_metrics.json): evaluation report.
- [embeddings/visualizations/](embeddings/visualizations): plots for analysis.

## Current Limitations

The current implementation is functional as a prototype, but it has several limitations:

- It is not a full tracking system. Identity is inferred from filename-based person indices rather than robust multi-frame tracking.
- The evaluation is performed on the same data used to build the database, so it is not a true independent test protocol.
- The query pipeline selects the largest detected face region rather than using a more principled selection strategy.
- The system depends on external model files and local InsightFace weights, which are not bundled with the repository.
- The metadata stores absolute paths, which makes the project less portable across machines.
- The pipeline is largely script-driven and not packaged as a reusable library or service.
- The current adaptive threshold appears to produce a high false acceptance rate in the generated evaluation metrics, which suggests that the thresholding strategy is not yet well calibrated for the available data.

## Future Improvements

Realistic next steps would include:

- adding robust person tracking across frames to stabilize identity assignment;
- using a proper train/validation/test split for evaluation;
- replacing the simple largest-face query selection with a more robust face selection strategy;
- adding a small re-ranking stage after FAISS retrieval;
- supporting batch query processing and video-level retrieval;
- making the metadata path handling relative and portable;
- adding a command-line interface for fully automated runs.

## Script-Level Audit

### [config.py](config.py)

- Purpose: centralize all config values used by the pipeline.
- Input: none; it defines defaults and paths.
- Output: a frozen `PipelineConfig` object and a module-level `CONFIG` instance.
- Algorithms used: none; this file is configuration-only.
- Models used: none directly, but it stores model paths and names.
- Important functions: `PipelineConfig` dataclass and module-level constants.
- Execution flow: imported by every script; no runtime logic.
- Dependencies: standard library only.
- Time complexity: $O(1)$.
- Space complexity: $O(1)$.

### [video_to_frames.py](video_to_frames.py)

- Purpose: sample frames from the source video.
- Input: the configured video file.
- Output: JPEG images in [saved_frames/](saved_frames).
- Algorithms used: simple sequential video decoding and frame sampling.
- Models used: none.
- Important functions: `format_timestamp`, `main`.
- Execution flow: open video, read frames, save every $N$-th frame, and print progress.
- Dependencies: OpenCV, tqdm, config.
- Time complexity: $O(F)$ over the number of frames read.
- Space complexity: $O(1)$ aside from the current frame buffer.

### [person_cropping.py](person_cropping.py)

- Purpose: crop person regions from saved frames using YOLO11.
- Input: extracted frames from [saved_frames/](saved_frames).
- Output: person crops in [cropped_persons/](cropped_persons).
- Algorithms used: YOLO object detection with confidence filtering and rectangular cropping.
- Models used: YOLO11 person detector from `yolo11n.pt`.
- Important functions: `setup`, `load_model`, `extract_frame_number`, `get_box_data`, `crop_persons`, `process_frames`, `main`.
- Execution flow: discover image files, load the model, run detection per frame, save valid person crops.
- Dependencies: OpenCV, NumPy, Ultralytics, tqdm, config.
- Time complexity: approximately $O(N \cdot D)$ where $N$ is the number of frames and $D$ is YOLO inference cost.
- Space complexity: $O(1)$ per image plus output storage.

### [face_cropping.py](face_cropping.py)

- Purpose: detect and crop faces from person crops.
- Input: person crops from [cropped_persons/](cropped_persons).
- Output: face crops in [cropped_faces/](cropped_faces) and a metadata JSON file.
- Algorithms used: YOLOv8-face detection, RetinaFace fallback, padding-based cropping, metadata serialization.
- Models used: YOLOv8-face and RetinaFace.
- Important functions: `setup`, `load_models`, `crop_faces_from_image`, `detect_and_crop`, `main`.
- Execution flow: load detectors, process each person crop, save face crops and metadata, and print a summary.
- Dependencies: OpenCV, tqdm, Ultralytics, config, shared utilities.
- Time complexity: roughly $O(M \cdot D_f)$ over the number of person crops and detector cost.
- Space complexity: $O(K)$ for metadata and output image storage, where $K$ is the number of faces.

### [face_recognition_utils.py](face_recognition_utils.py)

- Purpose: shared utility module for all face-processing stages.
- Input: images, detection objects, and configuration values.
- Output: detections, aligned crops, embeddings, threshold statistics, and JSON persistence helpers.
- Algorithms used: detection wrappers, bounding-box padding, landmark-based alignment, embedding normalization, threshold estimation, JSON save/load helpers.
- Models used: RetinaFace, ArcFace, PyTorch device selection.
- Important functions: `select_device`, `load_arcface`, `load_retinaface`, `crop_with_padding`, `detect_faces_yolo`, `detect_faces_retinaface`, `detect_faces_with_fallback`, `align_face`, `l2_normalize`, `embed_aligned_faces`, `compute_threshold_stats`, `save_json`, `load_threshold`.
- Execution flow: these helpers are called by the face-cropping, embedding, and query-retrieval scripts.
- Dependencies: OpenCV, NumPy, PyTorch, PIL, FAISS, InsightFace, config.
- Time complexity: depends on the underlying model inference and the batch size.
- Space complexity: proportional to the number of aligned faces in a batch and the embedding dimension.

### [master_embedding_database.py](master_embedding_database.py)

- Purpose: build the master database of ArcFace embeddings and the matching FAISS index.
- Input: face crops, saved frames, and detection metadata.
- Output: [embeddings/master_face_embeddings.faiss](embeddings/master_face_embeddings.faiss), [embeddings/master_face_metadata.json](embeddings/master_face_metadata.json), and [embeddings/threshold_stats.json](embeddings/threshold_stats.json).
- Algorithms used: filename parsing, frame lookup, face alignment, batched embedding, FAISS index creation, metadata validation.
- Models used: RetinaFace for alignment and ArcFace for embeddings.
- Important functions: `prepare_directories`, `discover_images`, `parse_face_filename`, `extract_frame_number`, `find_matching_frame`, `timestamp_from_frame`, `find_person_crop`, `detect_video_name`, `load_detection_metadata`, `build_records`, `save_faiss_database`, `validate_saved_database`, `main`.
- Execution flow: discover images, infer metadata from file names, align and embed each face, save the FAISS index and metadata, compute threshold statistics, and validate the resulting database.
- Dependencies: OpenCV, NumPy, FAISS, tqdm, config, shared utilities.
- Time complexity: $O(F \cdot E + F \cdot A)$, where $F$ is the number of face crops, $E$ is embedding cost, and $A$ is alignment cost.
- Space complexity: $O(F \cdot 512)$ for embeddings plus metadata storage.

### [query_matching.py](query_matching.py)

- Purpose: retrieve likely matches for a query image from the FAISS database.
- Input: query image, FAISS index, metadata, and model weights.
- Output: ranked retrieval results, a query preview, and console output for match or unknown-person decisions.
- Algorithms used: face detection, cropping, alignment, embedding, FAISS search, thresholding, confidence mapping.
- Models used: YOLOv8-face, RetinaFace, ArcFace.
- Important functions: `setup`, `load_models`, `load_search_database`, `detect_query_face`, `embed_query_face`, `search`, `apply_adaptive_threshold`, `show_match`, `print_ranked_results`, `print_match_results`, `main`.
- Execution flow: load the database, detect a face in the query image, embed it, query FAISS, apply the threshold, and print results.
- Dependencies: OpenCV, FAISS, Matplotlib, Pillow, Ultralytics, config, shared utilities.
- Time complexity: dominated by query face detection and embedding plus FAISS search, roughly $O(1)$ for the search step relative to the database index size because FAISS is optimized internally.
- Space complexity: $O(K)$ for Top-K results and the query embedding.

### [evaluation.py](evaluation.py)

- Purpose: evaluate the quality of the stored database using pairwise similarity metrics.
- Input: the saved FAISS index and metadata.
- Output: [embeddings/evaluation_metrics.json](embeddings/evaluation_metrics.json).
- Algorithms used: pairwise similarity computation, threshold-based classification, ROC and PR curve computations, confusion matrix analysis.
- Models used: none directly.
- Important functions: `load_embeddings_and_metadata`, `pairwise_scores`, `evaluate`, `main`.
- Execution flow: load the embeddings, construct pairwise same/different labels, evaluate thresholding performance, and save metrics.
- Dependencies: FAISS, NumPy, scikit-learn, config, shared utilities.
- Time complexity: $O(N^2)$ for pairwise comparisons over $N$ embeddings.
- Space complexity: $O(N^2)$ for storing the full pairwise similarity arrays in the current implementation.

### [visualization.py](visualization.py)

- Purpose: generate research-style plots for the embedding database.
- Input: the saved FAISS index and metadata.
- Output: PNG plots and a JSON manifest in [embeddings/visualizations/](embeddings/visualizations).
- Algorithms used: pairwise similarity computations, ROC/PR curves, histogram generation, PCA, and t-SNE.
- Models used: none directly.
- Important functions: `load_database`, `save_current_plot`, `plot_roc`, `plot_precision_recall`, `plot_similarity_histogram`, `plot_embedding_projection`, `plot_top_k_retrieval`, `main`.
- Execution flow: load embeddings, compute pairwise labels and similarities, create plots, and save them.
- Dependencies: FAISS, NumPy, scikit-learn, Matplotlib, OpenCV, config, evaluation utilities.
- Time complexity: dominated by pairwise evaluation and dimensionality reduction.
- Space complexity: depends on the number of embeddings and the chosen projection method.

## Technical Assessment

### Is this pipeline technically correct for Face Re-Identification?

Yes, at a prototype level, the pipeline is technically reasonable for a face re-identification workflow. It follows the standard ordering of detection, cropping, alignment, embedding, and similarity search. The use of ArcFace embeddings with FAISS indexing is a standard approach for face retrieval and is appropriate for a research prototype.

However, it is not yet a strong or fully robust re-identification system. The implementation is more of a file-based demonstration than a production-grade face recognition pipeline.

### Is this pipeline suitable for an internship project?

Yes. It is suitable as an internship project because it demonstrates multiple relevant topics: computer vision, face detection, embedding generation, vector search, evaluation, and visualization. It is also a good educational project for learning how such systems are assembled.

### Is this pipeline suitable for publication after improvements?

Not in its current form. The architecture is promising, but the pipeline would need stronger identity tracking, a better evaluation protocol, better threshold calibration, and stronger experimental rigor before it could be considered publication-ready.

### Scorecard

- Architecture: 6/10
- Code Quality: 6/10
- Face Recognition: 7/10
- Retrieval: 7/10
- Scalability: 5/10
- Research Quality: 5/10
- Documentation: 7/10

### Recommended improvements

Only the improvements that would meaningfully strengthen the project should be prioritized:

1. add robust person tracking across frames;
2. replace filename-derived identity labels with a more reliable identity assignment mechanism;
3. use a proper train/validation/test evaluation protocol;
4. improve the thresholding strategy and report confidence intervals and error analysis;
5. add a re-ranking stage after FAISS retrieval.
