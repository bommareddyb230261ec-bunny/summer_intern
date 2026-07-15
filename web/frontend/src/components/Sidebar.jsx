import { memo } from "react";
import {
  BarChart3,
  BriefcaseBusiness,
  Gauge,
  LogOut,
  Settings,
  Table2,
} from "lucide-react";
import { motion } from "framer-motion";

const NAV_ITEMS = [
  { label: "Dashboard", icon: Gauge, active: true },
  { label: "Jobs", icon: BriefcaseBusiness },
  { label: "Results", icon: Table2 },
  { label: "Analytics", icon: BarChart3 },
  { label: "Settings", icon: Settings },
];

function Sidebar({ profile, onLogout }) {
  const initials = profile?.name
    ?.split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <aside className="sidebar" aria-label="Primary">
      <div className="sidebar__brand">
        <div className="sidebar__mark">N</div>
        <div>
          <strong>NSG AI</strong>
          <span>Surveillance</span>
        </div>
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(({ label, icon: Icon, active }) => (
          <motion.a
            key={label}
            href="#"
            className={active ? "is-active" : ""}
            whileHover={{ x: 4 }}
            transition={{ type: "spring", stiffness: 320, damping: 24 }}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </motion.a>
        ))}
      </nav>

      <div className="sidebar__profile">
        <div className="avatar avatar--dark">{initials || "AI"}</div>
        <div>
          <strong>{profile?.name || "Loading user"}</strong>
          <span>{profile?.email || "Authenticated session"}</span>
        </div>
      </div>

      <button className="sidebar__logout" onClick={onLogout} type="button">
        <LogOut size={16} aria-hidden="true" />
        Logout
      </button>
    </aside>
  );
}

export default memo(Sidebar);
