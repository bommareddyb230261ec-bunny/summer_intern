# AI-Based Face Re-Identification from Video Sequences

## Project Overview

This project is an end-to-end AI surveillance dashboard for face re-identification from video sequences. It combines computer vision, deep learning, and web application development to detect people in video frames, extract faces, generate embeddings, match identities, and display results through an interactive dashboard.

The system is designed to help a user upload a query face image and one or more video files, process them through an AI pipeline, and retrieve the most similar faces from the indexed video data.

---

## What This Project Does

The pipeline performs the following steps:

1. Extracts frames from uploaded video files.
2. Detects persons in the video frames using YOLO.
3. Detects and crops faces from the detected person regions.
4. Aligns faces and generates ArcFace embeddings.
5. Stores and searches embeddings using FAISS for fast similarity matching.
6. Displays results through a React + FastAPI dashboard with authentication.

This makes the project a strong example of real-world AI application development, covering:

- computer vision
- deep learning inference
- vector similarity search
- backend APIs
- frontend dashboard UI
- authentication and job-based processing

---

## Key Features

- Face detection and person detection from video sequences
- ArcFace-based embedding generation
- FAISS-powered face similarity search
- Adaptive similarity thresholding for unknown-person rejection
- Job-based video processing pipeline
- Google OAuth authentication
- Interactive dashboard for uploading queries and videos
- Result visualization and output storage

---

## Technologies Used

### AI and Computer Vision

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- InsightFace
- ArcFace embeddings
- FAISS for similarity search
- scikit-learn for evaluation metrics
- matplotlib and seaborn for visualization

### Backend

- FastAPI
- SQLAlchemy
- Pydantic
- JWT authentication
- Authlib for Google OAuth

### Frontend

- React
- Vite
- React Router
- Axios
- Framer Motion
- Lucide icons

### Database and Storage

- SQLite for local development
- JSON-based result and metadata storage
- File-based caching for processed video artifacts

---

## Project Structure

- arcface/: AI pipeline for frame extraction, detection, embedding generation, and matching
- web/app/: FastAPI backend, authentication, routes, database, and job processing
- web/frontend/: React dashboard UI
- web/requirements.txt: Python dependencies
- web/frontend/package.json: frontend dependencies

---

## How to Clone the Project

Run the following commands:

```bash
git clone <your-repository-url>
cd "NSG AI Surveillance Dashboard"
```

If you already have the repository locally, make sure you are in the project root before running the setup steps below.

---

## Setup Instructions

### 1. Prerequisites

Make sure you have installed:

- Python 3.10 or newer
- Node.js 18 or newer
- Git

### 2. Create a Python Environment

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r web/requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd web/frontend
npm install
```

### 5. Configure Environment Variables

Create a file named `.env` inside the web folder and fill in the required values.

Example:

```env
DATABASE_URL=sqlite+aiosqlite:///./app/data/jobs.sqlite3
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
SECRET_KEY=your_secret_key
REDIRECT_URI=http://localhost:8000/auth/callback
FRONTEND_URL=http://localhost:5173
```

> If you are using a different database, replace the `DATABASE_URL` value accordingly.

### 6. Run the Backend

From the project root:

```bash
cd web
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Run the Frontend

In a second terminal:

```bash
cd web/frontend
npm run dev
```

Then open:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

---

## How the System Works

### Workflow

1. Upload a query face image.
2. Upload one or more video files.
3. Start processing.
4. The backend runs the ArcFace pipeline.
5. The system generates matches and stores the results.
6. The React dashboard displays the processed outputs and match results.

### Output

The project stores:

- extracted frames
- detected persons
- cropped faces
- embeddings and metadata
- FAISS index files
- evaluation metrics and visualization outputs

---

## Why This Project Is Impressive for Interviews

This project shows that you can build more than a simple demo. It demonstrates:

- end-to-end AI pipeline development
- real computer vision implementation
- practical use of deep learning models in production-like workflows
- integration of AI with a full-stack web application
- strong understanding of software engineering concepts like modular design, caching, job processing, and authentication

If an interviewer asks, you can confidently say:

> This project is an AI-based surveillance system that performs video-to-face re-identification using deep learning, similarity search, and a web-based interface.

---

## Future Improvements

Possible improvements for the project include:

- deployment to cloud platforms
- support for real-time video streams
- better UI/UX for monitoring and analytics
- database migration from SQLite to PostgreSQL
- model optimization for lower latency
- multi-camera tracking and identity continuity

---

## Summary

This repository is a strong portfolio project because it combines:

- artificial intelligence
- computer vision
- backend engineering
- frontend design
- modern software architecture

It is a great example of how AI can be transformed into a usable application that looks professional and interview-ready.
