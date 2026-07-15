import { memo } from "react";
import { Check, Loader2, X } from "lucide-react";

const STAGE_ORDER = [
  "QUERY_UPLOADED",
  "VIDEOS_UPLOADED",
  "FRAME_EXTRACTION",
  "PERSON_DETECTION",
  "FACE_ALIGNMENT",
  "EMBEDDING_GENERATION",
  "QUERY_MATCHING",
  "COMPLETED",
];

const STAGE_LABELS = {
  QUERY_UPLOADED: "Upload",
  VIDEOS_UPLOADED: "Video Intake",
  FRAME_EXTRACTION: "Frame Extraction",
  PERSON_DETECTION: "Person Detection",
  FACE_ALIGNMENT: "Face Alignment",
  EMBEDDING_GENERATION: "Embedding Generation",
  QUERY_MATCHING: "FAISS Search",
  COMPLETED: "Results Ready",
};

function stageState(stage, status, item) {
  if (status === "FAILED" && stage === item) return "failed";
  if (status === "COMPLETED") return "completed";
  const currentIndex = STAGE_ORDER.indexOf(stage);
  const itemIndex = STAGE_ORDER.indexOf(item);
  if (itemIndex < currentIndex) return "completed";
  if (itemIndex === currentIndex && status === "RUNNING") return "running";
  if (itemIndex === currentIndex && stage !== "IDLE") return "completed";
  return "waiting";
}

function ProgressTimeline({ stage, status }) {
  return (
    <ol className="timeline" aria-label="AI pipeline stages">
      {STAGE_ORDER.map((item) => {
        const state = stageState(stage, status, item);
        return (
          <li className={`timeline__item timeline__item--${state}`} key={item}>
            <span className="timeline__marker">
              {state === "completed" ? <Check size={15} /> : null}
              {state === "running" ? <Loader2 size={15} className="spin" /> : null}
              {state === "failed" ? <X size={15} /> : null}
            </span>
            <div>
              <strong>{STAGE_LABELS[item]}</strong>
              <small>{state}</small>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export default memo(ProgressTimeline);
