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
import LiveCameraPanel from "./components/LiveCameraPanel";
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
  const livePollRef = useRef(null);

  const [liveQueryFile, setLiveQueryFile] = useState(null);
  const [liveQueryPreview, setLiveQueryPreview] = useState("");
  const [liveQueryUploaded, setLiveQueryUploaded] = useState(false);
  const [liveCameraRunning, setLiveCameraRunning] = useState(false);
  const [liveFrame, setLiveFrame] = useState(null);
  const [liveStatus, setLiveStatus] = useState("idle");
  const [livePersonStatus, setLivePersonStatus] = useState("idle");
  const [liveFaceStatus, setLiveFaceStatus] = useState("idle");
  const [liveMatchStatus, setLiveMatchStatus] = useState("waiting");
  const [liveSimilarity, setLiveSimilarity] = useState(0.0);
  const [liveTimestamp, setLiveTimestamp] = useState("");
  const [liveHistory, setLiveHistory] = useState([]);
  const [liveError, setLiveError] = useState("");

  const canStart = jobId && stage === "VIDEOS_UPLOADED";

  const progressLabel = useMemo(() => {
    return message || STATUS_LABELS[status] || "Preparing...";
  }, [message, status]);

  const liveResults = useMemo(
    () =>
      liveHistory.map((item, index) => ({
        face_id: item.label || `Live-${index + 1}`,
        label: item.label || "Live match",
        similarity: Number(item.similarity || 0),
        timestamp: item.timestamp || "",
        matched_face_image: item.face_data,
        frame_name: item.timestamp || `Live-${index + 1}`,
        bounding_box: item.bbox || item.bounding_box || null,
        source: "live",
      })),
    [liveHistory],
  );

  const mergedResults = useMemo(
    () => [...liveResults, ...results],
    [liveResults, results],
  );
  const matchesCount = mergedResults.length;
  const peopleCount = useMemo(
    () =>
      new Set(mergedResults.map((item) => item.label).filter(Boolean)).size ||
      mergedResults.length,
    [mergedResults],
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
      if (livePollRef.current) {
        clearInterval(livePollRef.current);
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

  useEffect(() => {
    if (!liveQueryFile) {
      setLiveQueryPreview("");
      return undefined;
    }

    const previewUrl = URL.createObjectURL(liveQueryFile);
    setLiveQueryPreview(previewUrl);
    return () => URL.revokeObjectURL(previewUrl);
  }, [liveQueryFile]);

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

  const handleLiveQueryChange = (event) => {
    setLiveQueryFile(event.target.files?.[0] ?? null);
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

  const uploadLiveQuery = async () => {
    if (!liveQueryFile) {
      setLiveError("Please select a query face file.");
      return;
    }

    setLiveError("");
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", liveQueryFile);
      await api.post("/live/query", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setLiveQueryUploaded(true);
      setLiveStatus("query_uploaded");
    } catch (err) {
      setLiveError("Failed to upload live query face. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const startLiveCamera = async () => {
    if (!liveQueryUploaded) {
      setLiveError("Upload the live query face before starting the camera.");
      return;
    }
    setLiveError("");
    setLoading(true);

    try {
      await api.post("/live/start");
      setLiveCameraRunning(true);
      setLiveStatus("running");
      if (livePollRef.current) {
        clearInterval(livePollRef.current);
      }
      await fetchLiveStatus();
      livePollRef.current = setInterval(fetchLiveStatus, 1500);
    } catch (err) {
      setLiveError("Failed to start live camera.");
      setLiveCameraRunning(false);
    } finally {
      setLoading(false);
    }
  };

  const stopLiveCamera = async () => {
    setLoading(true);

    try {
      await api.post("/live/stop");
      setLiveCameraRunning(false);
      setLiveStatus("stopped");
      if (livePollRef.current) {
        clearInterval(livePollRef.current);
      }
    } catch (err) {
      setLiveError("Failed to stop live camera.");
    } finally {
      setLoading(false);
    }
  };

  const fetchLiveStatus = async () => {
    try {
      const response = await api.get("/live/status");
      setLiveStatus(response.data.camera_status || "idle");
      setLivePersonStatus(response.data.person_status || "idle");
      setLiveFaceStatus(response.data.face_status || "idle");
      setLiveMatchStatus(response.data.match_status || "waiting");
      setLiveSimilarity(response.data.similarity || 0.0);
      setLiveTimestamp(response.data.timestamp || "");
      setLiveHistory(response.data.history || []);

      const frameResponse = await api.get("/live/frame");
      setLiveFrame(frameResponse.data.frame_data || null);
    } catch (err) {
      setLiveError("Unable to fetch live camera status.");
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

            <LiveCameraPanel
              liveQueryFile={liveQueryFile}
              liveQueryPreview={liveQueryPreview}
              liveQueryUploaded={liveQueryUploaded}
              liveCameraRunning={liveCameraRunning}
              liveLoading={loading}
              liveError={liveError}
              liveFrame={liveFrame}
              liveStatus={liveStatus}
              livePersonStatus={livePersonStatus}
              liveFaceStatus={liveFaceStatus}
              liveMatchStatus={liveMatchStatus}
              liveSimilarity={liveSimilarity}
              liveTimestamp={liveTimestamp}
              onLiveQueryChange={handleLiveQueryChange}
              onUploadLiveQuery={uploadLiveQuery}
              onStartLiveCamera={startLiveCamera}
              onStopLiveCamera={stopLiveCamera}
            />
          </section>

          <ResultsTable
            results={mergedResults}
            queryPreview={queryPreview}
            liveQueryPreview={liveQueryPreview}
            onSelectResult={setSelectedResult}
          />
        </div>
      </main>

      <ResultDrawer
        result={selectedResult}
        queryPreview={queryPreview}
        liveQueryPreview={liveQueryPreview}
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
