import { memo, useState } from "react";
import { Camera, Clock, Play, StopCircle, UploadCloud, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

function LiveCameraPanel({
  liveQueryFile,
  liveQueryPreview,
  liveQueryUploaded,
  liveCameraRunning,
  liveLoading,
  liveError,
  liveFrame,
  liveStatus,
  livePersonStatus,
  liveFaceStatus,
  liveMatchStatus,
  liveSimilarity,
  liveTimestamp,
  onLiveQueryChange,
  onUploadLiveQuery,
  onStartLiveCamera,
  onStopLiveCamera,
}) {
  const [previewImages, setPreviewImages] = useState({ query: "", live: "" });

  return (
    <motion.article
      className="panel live-camera-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="panel__header">
        <div>
          <span>Live Camera</span>
          <h2>Live Face Search</h2>
        </div>
        <Camera size={22} aria-hidden="true" />
      </div>

      <label className="dropzone" htmlFor="live-query-upload">
        <input
          id="live-query-upload"
          type="file"
          accept="image/*"
          onChange={onLiveQueryChange}
        />
        <span className="dropzone__icon">
          <UploadCloud size={22} aria-hidden="true" />
        </span>
        <strong>Upload Query Face</strong>
        <small>Upload a face image to match against live camera input.</small>
      </label>
      <div className="file-list" aria-live="polite">
        <span>
          {liveQueryFile ? liveQueryFile.name : "No query face selected"}
        </span>
      </div>

      <div className="live-camera-actions">
        <motion.button
          type="button"
          disabled={!liveQueryFile || liveLoading}
          onClick={onUploadLiveQuery}
          whileTap={{ scale: 0.98 }}
        >
          Upload Face
        </motion.button>
        <motion.button
          type="button"
          disabled={!liveQueryUploaded || liveCameraRunning || liveLoading}
          onClick={onStartLiveCamera}
          whileTap={{ scale: 0.98 }}
          className="button-primary"
        >
          <Play size={16} aria-hidden="true" />
          Start Camera
        </motion.button>
        <motion.button
          type="button"
          disabled={!liveCameraRunning || liveLoading}
          onClick={onStopLiveCamera}
          whileTap={{ scale: 0.98 }}
        >
          <StopCircle size={16} aria-hidden="true" />
          Stop Camera
        </motion.button>
      </div>

      <div className="camera-preview-card">
        {liveFrame ? (
          <img
            className="camera-preview-image"
            src={liveFrame}
            alt="Live camera preview"
          />
        ) : (
          <div className="camera-preview-placeholder">
            <Clock size={40} aria-hidden="true" />
            <span>Live preview will appear after camera starts.</span>
          </div>
        )}
      </div>

      <div className="live-camera-metrics">
        <div>
          <strong>Camera Status</strong>
          <span>{liveStatus}</span>
        </div>
        <div>
          <strong>Person Detection</strong>
          <span>{livePersonStatus}</span>
        </div>
        <div>
          <strong>Face Detection</strong>
          <span>{liveFaceStatus}</span>
        </div>
        <div>
          <strong>Match Status</strong>
          <span>{liveMatchStatus}</span>
        </div>
        <div>
          <strong>Similarity Score</strong>
          <span>{liveSimilarity.toFixed(3)}</span>
        </div>
        <div>
          <strong>Timestamp</strong>
          <span>{liveTimestamp || "N/A"}</span>
        </div>
      </div>

      {liveError ? (
        <div className="error-card" style={{ marginTop: 16 }}>
          <strong>Camera error</strong>
          <span>{liveError}</span>
        </div>
      ) : null}

      <AnimatePresence>
        {previewImages.query || previewImages.live ? (
          <motion.div
            className="image-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Preview image zoom"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <button
              className="image-modal__backdrop"
              type="button"
              onClick={() => setPreviewImages({ query: "", live: "" })}
              aria-label="Close image preview"
            />
            <motion.div
              className="image-modal__content"
              initial={{ opacity: 0, y: 18, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 18, scale: 0.98 }}
            >
              <button
                className="icon-button"
                type="button"
                onClick={() => setPreviewImages({ query: "", live: "" })}
                aria-label="Close image preview"
              >
                <X size={18} aria-hidden="true" />
              </button>
              <div className="live-camera-preview-grid">
                {previewImages.query ? (
                  <div className="live-camera-preview-card">
                    <img src={previewImages.query} alt="Query face preview" />
                    <small>Query</small>
                  </div>
                ) : null}
                {previewImages.live ? (
                  <div className="live-camera-preview-card">
                    <img src={previewImages.live} alt="Live face preview" />
                    <small>Live</small>
                  </div>
                ) : null}
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.article>
  );
}

export default memo(LiveCameraPanel);
