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
  QUERY_UPLOADED: "Preparing Pipeline",
  VIDEOS_UPLOADED: "Extracting Frames",
  FRAME_EXTRACTION: "Detecting Persons",
  PERSON_DETECTION: "Cropping Persons",
  FACE_ALIGNMENT: "Detecting Faces",
  EMBEDDING_GENERATION: "Cropping Faces",
  QUERY_MATCHING: "Generating Face Embeddings",
  COMPLETED: "Face Retrieval Completed",
};

const STAGE_PERCENT = {
  QUERY_UPLOADED: "8%",
  VIDEOS_UPLOADED: "20%",
  FRAME_EXTRACTION: "40%",
  PERSON_DETECTION: "55%",
  FACE_ALIGNMENT: "70%",
  EMBEDDING_GENERATION: "80%",
  QUERY_MATCHING: "95%",
  COMPLETED: "100%",
};

function stageState(stage, status, item) {
  if (status === "FAILED" && stage === item) return "failed";
  if (status === "COMPLETED") return "completed";

  const currentIndex = STAGE_ORDER.indexOf(stage);
  const itemIndex = STAGE_ORDER.indexOf(item);

  if (currentIndex === -1) {
    return itemIndex === 0 ? "current" : "upcoming";
  }

  if (itemIndex < currentIndex) return "completed";
  if (itemIndex === currentIndex)
    return status === "RUNNING" ? "running" : "current";
  return "upcoming";
}

function ProgressTimeline({ stage, status }) {
  return (
    <div className="pipeline-timeline" aria-label="AI pipeline stages">
      <div className="pipeline-timeline__line" />
      {STAGE_ORDER.map((item) => {
        const state = stageState(stage, status, item);
        return (
          <div className={`timeline-step timeline-step--${state}`} key={item}>
            <span className="timeline-step__circle" aria-hidden="true">
              {state === "completed" ? <Check size={14} /> : null}
              {state === "running" ? (
                <Loader2 size={14} className="spin" />
              ) : null}
            </span>
            <span className="timeline-step__percent">
              {STAGE_PERCENT[item]}
            </span>
            <span className="timeline-step__label">{STAGE_LABELS[item]}</span>
          </div>
        );
      })}
    </div>
  );
}

export default memo(ProgressTimeline);
