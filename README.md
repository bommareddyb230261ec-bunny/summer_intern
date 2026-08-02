# AI-Based Face Re-Identification from Video Sequences

An end-to-end AI-powered surveillance system that identifies a person from one or more surveillance videos using a single query face image.

This project combines computer vision, deep learning, vector similarity search, and full-stack web development into a practical application for automated person search in large video collections.

---

# Project Overview

Face re-identification is an important capability in modern surveillance and security systems. In many real-world environments, operators must search through thousands of hours of footage to find a specific person, which is slow, costly, and often unreliable when performed manually. Small details, changing viewpoints, varying lighting, and large volumes of data make this task difficult even for experienced analysts.

This project addresses that problem by providing an end-to-end AI workflow that accepts a query face image, analyzes surveillance videos, detects people and faces, generates feature embeddings, and returns ranked matches with timestamps and visual previews. The system is designed to reduce the time required to locate a person in video data while improving consistency and scalability.

The application can be used in security monitoring, forensic review, access control investigations, public safety operations, and research environments where efficient video search is required.

---

# Why This Project Exists

Large surveillance systems often produce thousands of hours of video footage every day. Manually inspecting this content is not practical for most organizations, especially when the search target is a single individual. Human review introduces fatigue, delays, and occasional errors, while traditional keyword-based or manual search methods are inefficient for visual evidence.

This project exists to automate the search process using AI. Instead of requiring an operator to scan hours of video manually, the system can process the footage automatically, detect relevant face instances, compare them against a query image, and present the most likely matches in a structured and interpretable way.

---

# Key Objectives

The project is built around the following objectives:

- Detect people automatically in video frames
- Detect faces within those person regions
- Generate discriminative embeddings for each face
- Search efficiently over large embedding collections
- Return timestamps and matched frames for review
- Provide visualization for results and analysis
- Build an end-to-end AI application that is usable from a web dashboard

---

# Key Features

- Google OAuth login with JWT authentication
- Upload a query face image
- Upload multiple surveillance videos
- Automatic frame extraction
- Person detection using YOLO
- Face detection and cropping
- ArcFace embedding generation
- FAISS-based similarity search
- Interactive React dashboard
- FastAPI backend with REST APIs
- Processing status tracking
- Match visualization with timestamps
- Result management through the dashboard

---

# Complete System Workflow

The complete workflow of the system is designed to move from user interaction to intelligent matching in a clear and structured way.

1. Login

   Users authenticate through the application to access the dashboard securely. Authentication is handled through the platform’s backend and protected routes.

2. Upload Query Image

   A user provides a reference face image representing the person of interest. This image becomes the search query for the system.

3. Upload Videos

   One or more surveillance videos are uploaded for analysis. These videos are processed offline by the backend pipeline.

4. Backend

   The backend receives the uploaded files, stores them, manages the job lifecycle, and coordinates the execution of the AI pipeline.

5. AI Pipeline

   The system extracts frames, detects persons, detects faces, aligns the face regions, generates embeddings, and performs similarity search against the query image.

6. Results

   The pipeline produces ranked matches with similarity scores, timestamps, and preview images that can be reviewed by the user.

7. Dashboard

   The dashboard displays the results in a user-friendly interface so that the operator can inspect the matches quickly and understand the output.

---

# AI Pipeline

The AI pipeline is the core of the system. Each stage is designed to transform raw video content into searchable, interpretable results.

## 1. Frame Extraction

- Purpose: Convert videos into a sequence of individual frames that can be processed by computer vision models.
- Input: Uploaded video files.
- Output: Image frames, each associated with its timestamp.
- Reason: Video is a temporal medium, but most vision models operate on images. Frame extraction makes the content processable while preserving timing information for later result visualization.

## 2. YOLO Person Detection

- Purpose: Detect persons in each frame and localize their positions.
- Input: Extracted video frames.
- Output: Bounding boxes and cropped person regions.
- Reason: Running face detection over the entire frame would be inefficient and less accurate. Person detection narrows the search area to relevant regions first.

## 3. Face Detection

- Purpose: Identify the face regions within the detected person crops.
- Input: Cropped person images.
- Output: Face bounding boxes and face crops.
- Reason: This step isolates the facial region and avoids unnecessary processing of the rest of the body or background.

## 4. Face Alignment

- Purpose: Normalize detected faces before feature extraction.
- Input: Detected face crops.
- Output: Aligned face images with consistent orientation and geometry.
- Reason: Alignment helps the embedding model focus on meaningful facial structure and reduces variability caused by pose or scale differences.

## 5. ArcFace Embedding Generation

- Purpose: Convert every aligned face into a dense feature vector.
- Input: Aligned face images.
- Output: High-dimensional face embeddings.
- Reason: ArcFace produces representations that capture identity-related facial characteristics, making it possible to compare faces numerically and robustly.

## 6. FAISS Similarity Search

- Purpose: Search the embedding database quickly and efficiently.
- Input: Query embedding and the database of extracted face embeddings.
- Output: Similarity-ranked candidate matches.
- Reason: FAISS enables fast nearest-neighbor search, which is essential when the system processes many frames and faces.

## 7. Cosine Similarity

- Purpose: Measure how similar the query face is to each candidate embedding.
- Input: Query embedding and candidate embeddings.
- Output: Similarity scores for ranking.
- Reason: Cosine similarity provides a meaningful measure of directional similarity between feature vectors, which is well suited for face embeddings.

## 8. Threshold Decision

- Purpose: Decide whether a candidate should be treated as a valid match.
- Input: Similarity scores and configured thresholds.
- Output: Accepted or rejected matches.
- Reason: The threshold prevents weak or ambiguous matches from being treated as true positives and improves the reliability of the output.

## 9. Result Visualization

- Purpose: Present matched results clearly to the user.
- Input: Accepted matches, timestamps, and preview images.
- Output: Visual previews, match metadata, and dashboard-ready results.
- Reason: A useful AI system should not only identify matches, but also make them understandable and inspectable for the end user.

---

# System Architecture

## Overall System Architecture

![Overall Architecture](overall1.png)

The overall architecture is organized around a user-facing web application and a backend processing engine. The frontend allows users to log in, upload images and videos, and review results. The backend manages authentication, file storage, job execution, and API communication. The AI component processes the uploaded media, generates embeddings, performs similarity matching, and returns structured results that are displayed in the dashboard.

## AI Processing Pipeline

![ML Pipeline](ml1.png)

The pipeline begins with video input, then moves through frame extraction, person detection, face detection, face alignment, embedding generation, and similarity matching. The output is a set of candidate matches that are ranked and presented to the user with associated timestamps and previews. This flow reflects the end-to-end design of the application, from raw media input to actionable results.

---

# Technology Stack

## Artificial Intelligence

- Python
- PyTorch
- ArcFace
- InsightFace
- OpenCV
- YOLO
- FAISS

## Backend

- FastAPI
- SQLAlchemy
- JWT Authentication
- Google OAuth
- PostgreSQL
- Pydantic

## Frontend

- React
- Vite
- Axios
- React Router
- HTML
- CSS
- JavaScript

## Database

- SQLite database used by the web application

## Authentication

- Google OAuth login
- JWT-based session authentication

## Deployment

- Local development workflow with backend and frontend services

## Development Tools

- Git
- GitHub
- VS Code

---

# Project Structure

```text
NSG AI Surveillance Dashboard/

├── arcface/
│   Core AI processing modules for frame handling, person detection,
│   face detection, embedding generation, and query matching.
│
├── web/
│   Backend and frontend application code.
│
│   ├── app/
│   │   FastAPI backend, authentication logic, database models,
│   │   API routes, and processing pipeline orchestration.
│   │
│   ├── frontend/
│   │   React-based dashboard interface for user interaction.
│   │
│   └── requirements.txt
│
├── Results/
│   Output results generated by the processing pipeline.
│
└── README.md
```

---

# Authentication Flow

```text
Google Login

↓

Google OAuth

↓

FastAPI

↓

JWT Generation

↓

Dashboard Access
```

Only authenticated users can access the dashboard. The authentication layer protects the application and ensures that the processing workflow is available only to authorized users.

---

# Backend Workflow

```text
Upload Query

↓

Upload Videos

↓

Save Files

↓

Run Pipeline

↓

Generate Results

↓

Return JSON

↓

Dashboard
```

The backend coordinates everything from request handling to processing execution and result generation. It ensures that the AI pipeline is invoked in a consistent and trackable manner.

---

# Output

After processing is complete, the system generates both structured data and visual outputs for easy inspection.

## Results Folder

All generated outputs are stored in the results/ directory.

The folder may contain:

- Matched face images
- Cropped face images
- Matched video frames
- Timestamp information
- Similarity scores
- JSON result files
- Visualization images generated during processing

Each JSON file contains metadata such as:

- Query image details
- Matched video name
- Timestamp
- Similarity score
- Match confidence
- File paths to the generated images

The generated images allow users to visually verify the identified person without manually searching through the original video.

## Dashboard Results

The same information is also displayed through the React dashboard, where users can:

- View the query image
- Inspect matched frames
- Review timestamps
- Compare similarity scores
- Open generated result images
- Monitor processing status

---

# Documentation

## Project Report

For readers interested in implementation details and design decisions, please refer to the accompanying project report. It provides a more detailed explanation of the AI models, pipeline design, FAISS indexing approach, ArcFace integration, architecture, and implementation choices used in the project.

---

# Installation

## Clone Repository

```bash
git clone <repository-url>

cd "NSG AI Surveillance Dashboard"
```

---

## Backend

```bash
cd web

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Frontend

```bash
cd frontend

npm install
```

---

## Start Backend

```bash
uvicorn app.main:app --reload
```

---

## Start Frontend

```bash
npm run dev
```

---

# Future Improvements

The system has a strong foundation for practical use, and several improvements can further increase its value. Future work may include real-time CCTV support, multi-camera tracking, distributed processing for larger workloads, cloud deployment, GPU acceleration, more advanced person re-identification methods, and automated reporting features. These enhancements would improve scalability, robustness, and usability for production environments.

---

# Skills Demonstrated

This project demonstrates practical experience in:

- Computer Vision
- Deep Learning
- Face Recognition
- Embedding Models
- Similarity Search
- REST APIs
- JWT
- OAuth
- React
- FastAPI
- Database Design
- Software Engineering
- AI System Design
- Deep Learning
- Face Recognition
- REST API Development
- Full Stack Web Development
- Authentication
- Database Design
- Software Architecture
- Modular AI Pipeline Design
- Vector Similarity Search
- End-to-End AI System Integration

---

# About

This project was developed as part of an AI surveillance research initiative focused on automating face re-identification from surveillance video sequences.

It demonstrates how modern AI models can be integrated with scalable backend services and interactive web applications to solve real-world surveillance and security problems.
