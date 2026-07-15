import { memo, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Eye, ImageIcon, Search, SlidersHorizontal, X } from "lucide-react";
import EmptyState from "./EmptyState";
import StatusBadge from "./StatusBadge";

const PAGE_SIZE = 8;
const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const IMAGE_EXTENSIONS = /\.(avif|gif|jpe?g|png|webp)$/i;

function resultSimilarity(item) {
  return Number(item.similarity || 0);
}

function normalizePath(value) {
  return typeof value === "string" ? value.trim().replaceAll("\\", "/") : "";
}

function isImagePath(value) {
  return IMAGE_EXTENSIONS.test(value.split("?")[0]);
}

function absoluteUrl(path) {
  if (!path) return "";
  if (/^(https?:|blob:|data:)/i.test(path)) return path;
  if (path.startsWith("/")) return `${API_BASE_URL}${path}`;
  return `${API_BASE_URL}/${path}`;
}

function imageCandidates(item) {
  const rawFields = [
    item.matched_face_image,
    item.image_url,
    item.matched_image,
    item.matched_image_url,
    item.face_image,
    item.face_image_url,
    item.image_path,
    item.face_path,
    item.crop_path,
    item.cropped_face_path,
    item.thumbnail,
    isImagePath(normalizePath(item.face_id)) ? item.face_id : "",
  ]
    .map(normalizePath)
    .filter(Boolean);

  const candidates = new Set();
  rawFields.forEach((path) => {
    candidates.add(absoluteUrl(path));
    const filename = path.split("/").pop();
    if (filename) {
      candidates.add(`${API_BASE_URL}/uploads/${filename}`);
      candidates.add(`${API_BASE_URL}/results/faces/${filename}`);
      candidates.add(`${API_BASE_URL}/cropped_faces/${filename}`);
      candidates.add(`${API_BASE_URL}/static/${filename}`);
    }
  });
  return [...candidates];
}

function PlaceholderAvatar({ label }) {
  return (
    <div className="image-placeholder" aria-label="No matched image available">
      <ImageIcon size={20} aria-hidden="true" />
      <span>No Image Available</span>
    </div>
  );
}

function SmartImage({ sources, alt, className, fallbackLabel, onClick }) {
  const [sourceIndex, setSourceIndex] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const source = sources[sourceIndex];

  if (!source) {
    if (sources.length) {
      console.warn("Matched face image could not be loaded from candidate paths:", sources);
    }
    return <PlaceholderAvatar label={fallbackLabel} />;
  }

  return (
    <button className="image-button" type="button" onClick={() => onClick(source)} aria-label={`Preview ${alt}`}>
      {!loaded ? <span className="image-skeleton" /> : null}
      <motion.img
        key={source}
        className={className}
        src={source}
        alt={alt}
        loading="lazy"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: loaded ? 1 : 0, scale: loaded ? 1 : 0.96 }}
        onLoad={() => setLoaded(true)}
        onError={() => {
          setLoaded(false);
          setSourceIndex((current) => current + 1);
        }}
      />
      {sourceIndex >= sources.length ? <PlaceholderAvatar label={fallbackLabel} /> : null}
    </button>
  );
}

function ImagePreviewModal({ imageUrl, onClose }) {
  return (
    <AnimatePresence>
      {imageUrl ? (
        <motion.div
          className="image-modal"
          role="dialog"
          aria-modal="true"
          aria-label="Matched image preview"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <button className="image-modal__backdrop" type="button" onClick={onClose} aria-label="Close image preview" />
          <motion.div
            className="image-modal__content"
            initial={{ opacity: 0, y: 18, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 18, scale: 0.98 }}
          >
            <button className="icon-button" type="button" onClick={onClose} aria-label="Close image preview">
              <X size={18} aria-hidden="true" />
            </button>
            <img src={imageUrl} alt="Large matched face preview" />
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

function ResultsTable({ results, queryPreview, onSelectResult }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [sortDirection, setSortDirection] = useState("desc");
  const [page, setPage] = useState(1);
  const [previewImage, setPreviewImage] = useState("");

  const visibleResults = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return [...results]
      .filter((item) => {
        const haystack = `${item.face_id} ${item.label} ${item.timestamp} ${item.image_path || ""} ${item.face_image || ""}`.toLowerCase();
        const passesQuery = !normalizedQuery || haystack.includes(normalizedQuery);
        const passesFilter = filter === "all" || resultSimilarity(item) >= Number(filter);
        return passesQuery && passesFilter;
      })
      .sort((a, b) => {
        const diff = resultSimilarity(a) - resultSimilarity(b);
        return sortDirection === "asc" ? diff : -diff;
      });
  }, [filter, query, results, sortDirection]);

  const totalPages = Math.max(1, Math.ceil(visibleResults.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const paginated = visibleResults.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  return (
    <motion.article className="panel results-panel" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
      <div className="panel__header results-panel__header">
        <div>
          <span>Investigation</span>
          <h2>Results Table</h2>
        </div>
        <div className="results-tools">
          <label>
            <Search size={16} aria-hidden="true" />
            <input
              aria-label="Search results"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setPage(1);
              }}
              placeholder="Search results"
            />
          </label>
          <select
            aria-label="Filter by confidence"
            value={filter}
            onChange={(event) => {
              setFilter(event.target.value);
              setPage(1);
            }}
          >
            <option value="all">All confidence</option>
            <option value="0.9">90% and up</option>
            <option value="0.75">75% and up</option>
            <option value="0.5">50% and up</option>
          </select>
          <button type="button" onClick={() => setSortDirection((current) => (current === "desc" ? "asc" : "desc"))}>
            <SlidersHorizontal size={16} aria-hidden="true" />
            Sort {sortDirection === "desc" ? "High" : "Low"}
          </button>
        </div>
      </div>

      {!results.length ? (
        <EmptyState />
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Face Preview</th>
                  <th>Face ID</th>
                  <th>Similarity</th>
                  <th>Video Name</th>
                  <th>Matched Image</th>
                  <th>Confidence</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {paginated.map((item) => {
                  const confidence = Math.round(resultSimilarity(item) * 100);
                  const sources = imageCandidates(item);
                  const rowKey = `${item.face_id}-${item.label}-${item.timestamp}`;
                  return (
                    <tr key={rowKey}>
                      <td>
                        {queryPreview ? (
                          <button className="image-button" type="button" onClick={() => setPreviewImage(queryPreview)} aria-label="Preview query face">
                            <img className="query-thumb" src={queryPreview} alt="Uploaded query face" loading="lazy" />
                          </button>
                        ) : (
                          <PlaceholderAvatar label="Q" />
                        )}
                      </td>
                      <td>{item.face_id}</td>
                      <td>{confidence}%</td>
                      <td>{item.label}</td>
                      <td>
                        <SmartImage
                          sources={sources}
                          alt={`Matched face ${item.face_id}`}
                          className="matched-thumb"
                          fallbackLabel={item.face_id}
                          onClick={setPreviewImage}
                        />
                      </td>
                      <td>
                        <div className="mini-meter">
                          <span style={{ width: `${confidence}%` }} />
                        </div>
                      </td>
                      <td>
                        <StatusBadge status={confidence >= 75 ? "COMPLETED" : "QUEUED"} label={confidence >= 75 ? "Match" : "Review"} />
                      </td>
                      <td>
                        <button className="table-action" type="button" onClick={() => onSelectResult({ ...item, imageSources: sources })}>
                          <Eye size={16} aria-hidden="true" />
                          View Details
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <span>
              Page {safePage} of {totalPages}
            </span>
            <div>
              <button type="button" disabled={safePage === 1} onClick={() => setPage((current) => Math.max(1, current - 1))}>
                <ChevronLeft size={16} aria-hidden="true" />
              </button>
              <button type="button" disabled={safePage === totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))}>
                <ChevronRight size={16} aria-hidden="true" />
              </button>
            </div>
          </div>
        </>
      )}

      <ImagePreviewModal imageUrl={previewImage} onClose={() => setPreviewImage("")} />
    </motion.article>
  );
}

export { imageCandidates, SmartImage, PlaceholderAvatar };
export default memo(ResultsTable);
