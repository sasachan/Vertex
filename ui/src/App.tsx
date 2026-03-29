import { useState } from "react";
import { Sidebar } from "./components/layout/Sidebar.tsx";
import { LiveDashboard } from "./pages/LiveDashboard.tsx";
import { SessionsPage } from "./pages/Sessions.tsx";
import { SettingsPage } from "./pages/Settings.tsx";

export default function App() {
  const [page, setPage] = useState("/");

  return (
    <>
      <Sidebar active={page} onNavigate={setPage} />
      {page === "/" && <LiveDashboard />}
      {page === "/sessions" && <SessionsPage />}
      {page === "/settings" && <SettingsPage />}
    </>
  );
}
