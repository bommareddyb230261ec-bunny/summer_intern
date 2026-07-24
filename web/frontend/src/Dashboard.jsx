import { useEffect, useMemo, useRef, useState } from "react";
import api from "./api/api";
import ErrorState from "./components/ErrorState";
import LoadingOverlay from "./components/LoadingOverlay";
import Navbar from "./components/Navbar";
import PipelineProgress from "./components/PipelineProgress";
import ResultDrawer from "./components/ResultDrawer";
import ResultsTable from "./components/ResultsTable";
import Sidebar from "./components/Sidebar";
import SummaryCards from "./components/SummaryCards";
import UploadPanel from "./components/UploadPanel";
import "./Dashboard.css";

const STATUS_LABELS = {
  idle: "Waiting for uploads",
  QUEUED: "Queued",
  RUNNING: "Analysis in progress",
  COMPLETED: "Analysis complete",
  FAILED: "Analysis failed",
};

function Dashboard() {
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("idle");
  const [stage, setStage] = useState("IDLE");
  const [message, setMessage] = useState("Waiting for uploads");
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [queryFile, setQueryFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [queryPreview, setQueryPreview] = useState("");
  const pollRef = useRef(null);
  const completionTimerRef = useRef(null);

  const canStart = jobId && stage === "VIDEOS_UPLOADED";

  const progressLabel = useMemo(() => {
    return message || STATUS_LABELS[status] || "Preparing...";
  }, [message, status]);

  const matchesCount = results.length;
  const peopleCount = useMemo(
    () =>
      new Set(results.map((item) => item.label).filter(Boolean)).size ||
      results.length,
    [results],
  );

  useEffect(() => {
    api
      .get("/profile")
      .then(({ data }) => setProfile(data))
      .catch(() => {
        localStorage.removeItem("access_token");
        window.location.replace("/");
      });

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
      if (completionTimerRef.current) {
        clearTimeout(completionTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!queryFile) {
      setQueryPreview("");
      return undefined;
    }

    const previewUrl = URL.createObjectURL(queryFile);
    setQueryPreview(previewUrl);
    return () => URL.revokeObjectURL(previewUrl);
  }, [queryFile]);

  const handleLogout = async () => {
    setLoading(true);
    try {
      await api.post("/logout");
    } finally {
      localStorage.removeItem("access_token");
      window.location.replace("/");
    }
  };

  const handleQueryChange = (event) => {
    setQueryFile(event.target.files?.[0] ?? null);
  };

  const handleVideosChange = (event) => {
    setVideoFiles(Array.from(event.target.files ?? []));
  };

  const uploadQuery = async () => {
    if (!queryFile) {
      setError("Please select a query face file.");
      return;
    }

    setError("");
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", queryFile);

      const response = await api.post("/upload/query", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setJobId(response.data.job_id);
      setStatus("QUEUED");
      setStage("QUERY_UPLOADED");
      setMessage(response.data.message || "Query face uploaded.");
      setProgress(10);
    } catch (err) {
      setError("Failed to upload query face. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const uploadVideos = async () => {
    if (!jobId) {
      setError("Upload the query face first.");
      return;
    }
    if (!videoFiles.length) {
      setError("Please select one or more videos.");
      return;
    }

    setError("");
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("job_id", jobId);
      videoFiles.forEach((file) => {
        formData.append("files", file);
      });

      await api.post("/upload/videos", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setStatus("QUEUED");
      setStage("VIDEOS_UPLOADED");
      setMessage("Videos uploaded. Ready to start analysis.");
      setProgress(25);
    } catch (err) {
      setError("Failed to upload videos. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const startProcessing = async () => {
    if (!canStart) {
      return;
    }

    setError("");
    setLoading(true);

    try {
      const response = await api.post("/process/start", { job_id: jobId });
      const startedJobId = response.data.job_id;
      setJobId(startedJobId);
      setStatus("QUEUED");
      setStage("QUEUED");
      setMessage(response.data.message || "Processing started.");
      setProgress(5);
      if (completionTimerRef.current) {
        clearTimeout(completionTimerRef.current);
      }
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
      await fetchStatus(startedJobId);
      pollRef.current = setInterval(() => fetchStatus(startedJobId), 2000);
    } catch (err) {
      setError("Failed to start processing.");
      setLoading(false);
    }
  };

  const fetchStatus = async (targetJobId = jobId) => {
    if (!targetJobId) return;

    try {
      const response = await api.get(`/process/status/${targetJobId}`);
      setStatus(response.data.status);
      setStage(response.data.stage);
      setMessage(response.data.message);
      setProgress(response.data.progress);

      if (response.data.status === "COMPLETED") {
        if (pollRef.current) {
          clearInterval(pollRef.current);
        }
        await loadResults(targetJobId);
        completionTimerRef.current = setTimeout(() => {
          setLoading(false);
        }, 2000);
      }

      if (response.data.status === "FAILED") {
        if (pollRef.current) {
          clearInterval(pollRef.current);
        }
        setError(response.data.message || "Processing failed.");
        setLoading(false);
      }
    } catch (err) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
      setError("Unable to fetch process status.");
      setLoading(false);
    }
  };

  const loadResults = async (targetJobId = jobId) => {
    if (!targetJobId) return;

    try {
      const response = await api.get(`/results/${targetJobId}`);
      setResults(response.data.results);
    } catch (err) {
      setError("Unable to load results.");
    }
  };

  return (
    <div className="dashboard-shell">
      <Sidebar profile={profile} onLogout={handleLogout} />

      <main className="dashboard-main">
        <div className="dashboard-container">
          <Navbar profile={profile} status={status} message={progressLabel} />

          {status === "COMPLETED" ? (
            <section className="success-banner" aria-live="polite">
              <strong>Analysis completed</strong>
              <span>
                {matchesCount} matches found. Execution status is stored under
                job {jobId}.
              </span>
            </section>
          ) : null}

          <ErrorState message={error} onRetry={() => setError("")} />

          <SummaryCards
            queryFile={queryFile}
            queryPreview={queryPreview}
            videoCount={videoFiles.length}
            peopleCount={peopleCount}
            matchesCount={matchesCount}
          />

          <section className="workspace-grid">
            <UploadPanel
              queryFile={queryFile}
              videoFiles={videoFiles}
              uploading={uploading}
              loading={loading}
              jobId={jobId}
              canStart={canStart}
              onQueryChange={handleQueryChange}
              onVideosChange={handleVideosChange}
              onUploadQuery={uploadQuery}
              onUploadVideos={uploadVideos}
              onStartProcessing={startProcessing}
            />

            <PipelineProgress
              status={status}
              stage={stage}
              message={progressLabel}
              progress={progress}
              videoCount={videoFiles.length}
              peopleCount={peopleCount}
            />
          </section>

          <ResultsTable
            results={results}
            queryPreview={queryPreview}
            onSelectResult={setSelectedResult}
          />
        </div>
      </main>

      <ResultDrawer
        result={selectedResult}
        queryPreview={queryPreview}
        onClose={() => setSelectedResult(null)}
      />
      <LoadingOverlay
        show={loading}
        message={progressLabel}
        progress={progress}
        stage={stage}
        status={status}
      />
    </div>
  );
}

export default Dashboard;
