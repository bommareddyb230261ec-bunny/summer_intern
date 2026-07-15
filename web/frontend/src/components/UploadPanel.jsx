import { memo } from "react";
import { motion } from "framer-motion";
import { ImagePlus, Play, UploadCloud, Video } from "lucide-react";

function UploadDropzone({ id, title, description, accept, multiple, onChange, icon: Icon }) {
  return (
    <label className="dropzone" htmlFor={id}>
      <input id={id} type="file" accept={accept} multiple={multiple} onChange={onChange} />
      <span className="dropzone__icon">
        <Icon size={22} aria-hidden="true" />
      </span>
      <strong>{title}</strong>
      <small>{description}</small>
    </label>
  );
}

function UploadPanel({
  queryFile,
  videoFiles,
  uploading,
  loading,
  jobId,
  canStart,
  onQueryChange,
  onVideosChange,
  onUploadQuery,
  onUploadVideos,
  onStartProcessing,
}) {
  return (
    <motion.article
      className="panel upload-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="panel__header">
        <div>
          <span>Input</span>
          <h2>Upload Panel</h2>
        </div>
        <UploadCloud size={22} aria-hidden="true" />
      </div>

      <UploadDropzone
        id="query-upload"
        title="Upload Query Face"
        description="Drag or select a clear face image"
        accept="image/*"
        onChange={onQueryChange}
        icon={ImagePlus}
      />
      <div className="file-list" aria-live="polite">
        <span>{queryFile ? queryFile.name : "No query face selected"}</span>
      </div>

      <UploadDropzone
        id="video-upload"
        title="Upload Videos"
        description="Select one or more surveillance videos"
        accept="video/*"
        multiple
        onChange={onVideosChange}
        icon={Video}
      />
      <div className="file-list file-list--stack" aria-live="polite">
        {videoFiles.length ? (
          videoFiles.map((file) => <span key={`${file.name}-${file.size}`}>{file.name}</span>)
        ) : (
          <span>No videos selected</span>
        )}
      </div>

      <div className="upload-panel__actions">
        <motion.button type="button" disabled={uploading} onClick={onUploadQuery} whileTap={{ scale: 0.98 }}>
          Upload Face
        </motion.button>
        <motion.button type="button" disabled={uploading || !jobId} onClick={onUploadVideos} whileTap={{ scale: 0.98 }}>
          Upload Videos
        </motion.button>
        <motion.button
          className="button-primary"
          type="button"
          disabled={!canStart || loading}
          onClick={onStartProcessing}
          whileTap={{ scale: 0.98 }}
        >
          <Play size={17} aria-hidden="true" />
          Start Analysis
        </motion.button>
      </div>
    </motion.article>
  );
}

export default memo(UploadPanel);
