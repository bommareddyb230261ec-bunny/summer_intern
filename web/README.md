# Web Dashboard for AI Face Re-Identification

This folder contains the web application for the AI-based face re-identification system.

## What it provides

- FastAPI backend for processing and serving results
- Google OAuth authentication
- React dashboard for uploading images and videos
- job-based pipeline execution

## Run locally

### Backend

```bash
cd web
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd web/frontend
npm install
npm run dev
```

## Notes

- The backend expects Google OAuth credentials and a valid database URL.
- The first run may download model assets used by the face recognition pipeline.
- The UI is designed to show pipeline progress, job status, and matched results.
