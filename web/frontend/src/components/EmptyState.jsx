import { memo } from "react";
import { FileSearch } from "lucide-react";

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-state__illustration">
        <FileSearch size={38} aria-hidden="true" />
      </div>
      <h3>Upload a query face and surveillance videos to begin analysis.</h3>
      <p>Results, match confidence, timestamps, and metadata will appear here after processing.</p>
    </div>
  );
}

export default memo(EmptyState);
