import { memo } from "react";

const STATUS_COPY = {
  idle: "Waiting",
  QUEUED: "Queued",
  RUNNING: "Running",
  COMPLETED: "Completed",
  FAILED: "Failed",
};

function StatusBadge({ status = "idle", label }) {
  return (
    <span className={`status-badge status-badge--${status.toLowerCase()}`}>
      <span className="status-badge__dot" />
      {label || STATUS_COPY[status] || status}
    </span>
  );
}

export default memo(StatusBadge);
