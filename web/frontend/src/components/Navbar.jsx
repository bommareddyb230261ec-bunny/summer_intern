import { memo } from "react";
import { Bell, Search } from "lucide-react";
import { motion } from "framer-motion";
import StatusBadge from "./StatusBadge";

function Navbar({ profile, status, message }) {
  const initials = profile?.name
    ?.split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <header className="topbar">
      <div className="topbar__title">
        <span>Enterprise AI Operations</span>
        <h1>Dashboard</h1>
      </div>

      <div className="topbar__status">
        <StatusBadge status={status} />
        <span>{message}</span>
      </div>

      <label className="topbar__search">
        <Search size={18} aria-hidden="true" />
        <input aria-label="Search dashboard" placeholder="Search jobs, faces, videos" />
      </label>

      <motion.button
        className="icon-button"
        type="button"
        aria-label="Notifications"
        whileTap={{ scale: 0.94 }}
      >
        <Bell size={18} aria-hidden="true" />
      </motion.button>

      <div className="topbar__avatar avatar">{initials || "AI"}</div>
    </header>
  );
}

export default memo(Navbar);
