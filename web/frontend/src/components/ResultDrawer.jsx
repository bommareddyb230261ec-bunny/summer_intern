import { memo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Box, Crosshair, ImageIcon, MapPinned, ShieldCheck, X } from "lucide-react";
import StatusBadge from "./StatusBadge";
import { PlaceholderAvatar, SmartImage, imageCandidates } from "./ResultsTable";

function fieldValue(result, keys, fallback = "Unavailable") {
  const value = keys.map((key) => result?.[key]).find((item) => item !== undefined && item !== null && item !== "");
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object" && value) return JSON.stringify(value);
  return value || fallback;
}

function ResultDrawer({ result, queryPreview, onClose }) {
  const similarity = result ? Math.round(Number(result.similarity || 0) * 100) : 0;
  const imageSources = result?.imageSources?.length ? result.imageSources : imageCandidates(result || {});
  const status = similarity >= 75 ? "COMPLETED" : "QUEUED";

  return (
    <AnimatePresence>
      {result ? (
        <>
          <motion.button
            className="drawer-backdrop"
            type="button"
            aria-label="Close result drawer"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
          <motion.aside
            className="result-drawer"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 240 }}
            aria-label="Result details"
          >
            <div className="result-drawer__header">
              <div>
                <span>Match Detail</span>
                <h2>{result.face_id}</h2>
              </div>
              <button className="icon-button" type="button" onClick={onClose} aria-label="Close details">
                <X size={18} aria-hidden="true" />
              </button>
            </div>

            <div className="drawer-images">
              <div>
                <span>Matched Face</span>
                <SmartImage
                  sources={imageSources}
                  alt={`Matched face ${result.face_id}`}
                  className="drawer-face-image"
                  fallbackLabel={result.face_id}
                  onClick={() => {}}
                />
              </div>
              <div>
                <span>Query Face</span>
                {queryPreview ? (
                  <img className="drawer-query-image" src={queryPreview} alt="Uploaded query face" loading="lazy" />
                ) : (
                  <PlaceholderAvatar label="Q" />
                )}
              </div>
            </div>

            <div className="drawer-score">
              <span>Similarity score</span>
              <strong>{similarity}%</strong>
              <div className="mini-meter">
                <span style={{ width: `${similarity}%` }} />
              </div>
            </div>

            <div className="metadata-list">
              <div>
                <MapPinned size={17} aria-hidden="true" />
                <span>Video name</span>
                <strong>{result.label || "Unknown video"}</strong>
              </div>
              <div>
                <ImageIcon size={17} aria-hidden="true" />
                <span>Frame name</span>
                <strong>{fieldValue(result, ["frame_name", "frame", "frame_id", "timestamp", "face_id"])}</strong>
              </div>
              <div>
                <ShieldCheck size={17} aria-hidden="true" />
                <span>Match status</span>
                <strong>
                  <StatusBadge status={status} label={similarity >= 75 ? "Match" : "Review"} />
                </strong>
              </div>
              <div>
                <Box size={17} aria-hidden="true" />
                <span>Confidence</span>
                <strong>{similarity}%</strong>
              </div>
              <div>
                <Crosshair size={17} aria-hidden="true" />
                <span>Bounding box</span>
                <strong>{fieldValue(result, ["bounding_box", "bbox", "box", "face_bbox"])}</strong>
              </div>
            </div>
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  );
}

export default memo(ResultDrawer);
