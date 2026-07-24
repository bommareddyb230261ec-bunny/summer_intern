import { memo } from "react";
import { motion } from "framer-motion";
import { Activity, Clock, Cpu, Gauge } from "lucide-react";
import ProgressTimeline from "./ProgressTimeline";
import StatusBadge from "./StatusBadge";

function PipelineProgress({
  status,
  stage,
  message,
  progress,
  videoCount,
  peopleCount,
}) {
  const isRunning = status === "RUNNING";
  const eta =
    status === "COMPLETED" ? "Complete" : isRunning ? "2-5 min" : "Pending";

  return (
    <motion.article
      className="panel pipeline-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="panel__header">
        <div>
          <span>Pipeline</span>
          <h2>AI Processing</h2>
        </div>
        <StatusBadge status={status} />
      </div>

      <div className="progress-hero">
        <div>
          <strong>{progress}%</strong>
          <span>{message}</span>
        </div>
        <div
          className="progress-ring"
          style={{ "--progress": `${progress * 3.6}deg` }}
          aria-label={`${progress}% complete`}
        >
          <span>{progress}</span>
        </div>
      </div>

      <div className="progress-track">
        <motion.div
          className="progress-track__bar"
          animate={{ width: `${progress}%` }}
          transition={{ type: "spring", stiffness: 100, damping: 24 }}
        />
      </div>

      <ProgressTimeline stage={stage} status={status} />

      <div className="metrics-grid">
        <div>
          <Clock size={16} aria-hidden="true" />
          <span>Estimated time</span>
          <strong>{eta}</strong>
        </div>
        <div>
          <Gauge size={16} aria-hidden="true" />
          <span>Current stage</span>
          <strong>{stage.replaceAll("_", " ")}</strong>
        </div>
        <div>
          <Activity size={16} aria-hidden="true" />
          <span>Detected people</span>
          <strong>{peopleCount}</strong>
        </div>
        <div>
          <Cpu size={16} aria-hidden="true" />
          <span>Processing speed</span>
          <strong>
            {isRunning ? `${Math.max(videoCount, 1)} stream/s` : "Idle"}
          </strong>
        </div>
      </div>
    </motion.article>
  );
}

export default memo(PipelineProgress);
