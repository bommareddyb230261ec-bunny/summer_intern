import { memo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

const STAGE_LABELS = {
  QUEUED: "Uploading Files...",
  STARTING: "Preparing Pipeline...",
  FRAME_EXTRACTION: "Extracting Frames...",
  PERSON_DETECTION: "Detecting Persons...",
  FACE_ALIGNMENT: "Detecting Faces...",
  EMBEDDING_GENERATION: "Generating Embeddings...",
  QUERY_MATCHING: "Matching Query Face...",
  COMPLETED: "Analysis Completed Successfully",
  FAILED: "Analysis Failed",
};

function LoadingOverlay({
  show,
  message,
  progress = 0,
  stage = "QUEUED",
  status = "QUEUED",
}) {
  const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
  const isComplete = status === "COMPLETED" || safeProgress >= 100;
  const isFailed = status === "FAILED";
  const stageLabel = STAGE_LABELS[stage] || message || "Processing...";

  return (
    <AnimatePresence>
      {show ? (
        <motion.div
          className="loading-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          aria-live="polite"
        >
          <motion.div
            className={`loading-overlay__card ${isComplete ? "is-complete" : ""} ${isFailed ? "is-failed" : ""}`}
            initial={{ y: 18, scale: 0.98 }}
            animate={{ y: 0, scale: 1 }}
            exit={{ y: 18, scale: 0.98 }}
          >
            <div
              className="loading-progress-ring"
              style={{ "--progress": `${safeProgress * 3.6}deg` }}
              aria-label={`${safeProgress}% complete`}
            >
              {isComplete ? (
                <CheckCircle2 size={38} aria-hidden="true" />
              ) : null}
              {isFailed ? <AlertTriangle size={38} aria-hidden="true" /> : null}
              {!isComplete && !isFailed ? (
                <strong>{safeProgress}%</strong>
              ) : null}
            </div>

            <div className="loading-overlay__copy">
              <strong>
                {isComplete
                  ? "Analysis Completed Successfully"
                  : isFailed
                    ? "Analysis Failed"
                    : stageLabel}
              </strong>
              <p>{isComplete ? "Refreshing results table..." : message}</p>
            </div>

            <div className="loading-linear" aria-hidden="true">
              <motion.span
                animate={{ width: `${safeProgress}%` }}
                transition={{ type: "spring", stiffness: 100, damping: 22 }}
              />
            </div>

            <span className="loading-overlay__percent">{safeProgress}%</span>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

export default memo(LoadingOverlay);
