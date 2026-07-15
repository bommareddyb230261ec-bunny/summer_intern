import { memo } from "react";
import { motion } from "framer-motion";
import { FileImage, SearchCheck, Users, Video } from "lucide-react";

function SummaryCards({ queryFile, queryPreview, videoCount, peopleCount, matchesCount }) {
  const cards = [
    {
      title: "Query Face",
      value: queryFile ? "Ready" : "None",
      meta: queryFile?.name || "Awaiting upload",
      icon: FileImage,
      preview: queryPreview,
    },
    {
      title: "Videos Uploaded",
      value: videoCount,
      meta: videoCount === 1 ? "1 file selected" : `${videoCount} files selected`,
      icon: Video,
    },
    {
      title: "People Detected",
      value: peopleCount,
      meta: "From completed matches",
      icon: Users,
    },
    {
      title: "Matches Found",
      value: matchesCount,
      meta: "High confidence results",
      icon: SearchCheck,
    },
  ];

  return (
    <section className="summary-grid" aria-label="Pipeline summary">
      {cards.map(({ title, value, meta, icon: Icon, preview }, index) => (
        <motion.article
          className="summary-card"
          key={title}
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.06 }}
          whileHover={{ y: -4 }}
        >
          <div className="summary-card__top">
            <span className="summary-card__icon">
              <Icon size={20} aria-hidden="true" />
            </span>
            {preview ? (
              <img className="summary-card__preview" src={preview} alt="Query face preview" />
            ) : null}
          </div>
          <span>{title}</span>
          <strong>{value}</strong>
          <small>{meta}</small>
        </motion.article>
      ))}
    </section>
  );
}

export default memo(SummaryCards);
