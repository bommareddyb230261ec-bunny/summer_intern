import { memo } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

function ErrorState({ message, onRetry }) {
  if (!message) return null;

  return (
    <div className="error-card" role="alert">
      <AlertTriangle size={20} aria-hidden="true" />
      <div>
        <strong>Pipeline alert</strong>
        <span>{message}</span>
      </div>
      <button type="button" onClick={onRetry}>
        <RotateCcw size={15} aria-hidden="true" />
        Retry
      </button>
    </div>
  );
}

export default memo(ErrorState);
