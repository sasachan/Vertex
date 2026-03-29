import { Activity, FolderOpen, Settings, ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "../../lib/utils.ts";

const NAV = [
  { icon: Activity, label: "Live", path: "/" },
  { icon: FolderOpen, label: "Sessions", path: "/sessions" },
  { icon: Settings, label: "Settings", path: "/settings" },
] as const;

interface SidebarProps {
  active: string;
  onNavigate: (path: string) => void;
}

export function Sidebar({ active, onNavigate }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "flex flex-col bg-vertex-surface border-r border-vertex-border",
        "transition-[width] duration-200",
        collapsed ? "w-16" : "w-48",
      )}
    >
      {/* Logo area */}
      <div className="flex items-center gap-2 px-4 h-14 border-b border-vertex-border">
        <div className="w-7 h-7 rounded-md bg-vertex-accent flex items-center justify-center text-black font-bold text-sm flex-shrink-0">
          V
        </div>
        {!collapsed && (
          <span className="font-semibold text-sm tracking-wide text-white">
            VERTEX
          </span>
        )}
      </div>

      {/* Nav items */}
      <nav className="flex-1 py-3 flex flex-col gap-1 px-2">
        {NAV.map(({ icon: Icon, label, path }) => (
          <button
            key={path}
            onClick={() => onNavigate(path)}
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors",
              "hover:bg-white/5",
              active === path
                ? "bg-white/10 text-white"
                : "text-vertex-muted",
            )}
          >
            <Icon size={18} className="flex-shrink-0" />
            {!collapsed && label}
          </button>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center h-10 border-t border-vertex-border text-vertex-muted hover:text-white transition-colors"
      >
        {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      {/* Version */}
      {!collapsed && (
        <div className="px-4 pb-3 text-[10px] text-vertex-muted">v0.3.0</div>
      )}
    </aside>
  );
}
