// src/App.tsx

import { useState } from "react";
import { Sidebar, TopBar } from "@/components/Sidebar";
import { AuditPage } from "@/pages/AuditPage";
import { PipelinePage } from "@/pages/PipelinePage";
import { TriagePage } from "@/pages/TriagePage";
import { TelemetryPage } from "@/pages/TelemetryPage";
import { LoginPage } from "@/pages/LoginPage";
import type { PageKey } from "@/types";

function App() {
  const [page, setPage] = useState<PageKey>("audit");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    return Boolean(localStorage.getItem("token"));
  });

  const handleLoginSuccess = (token: string, user: any) => {
    localStorage.setItem("token", token);
    localStorage.setItem("user", JSON.stringify(user));
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setIsAuthenticated(false);
    setPage("audit");
  };

  if (!isAuthenticated) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#f8fafc]">
      <Sidebar
        page={page}
        setPage={setPage}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar
          onMenu={() => setSidebarOpen((o) => !o)}
          settingsOpen={settingsOpen}
          onOpenSettings={() => setSettingsOpen((o) => !o)}
          onCloseSettings={() => setSettingsOpen(false)}
          onLogout={handleLogout}
        />

        <main className="flex-1 overflow-y-auto px-5 lg:px-8 py-6">
          <div className={page === "audit" ? "block" : "hidden"}>
            <AuditPage />
          </div>

          <div className={page === "pipeline" ? "block" : "hidden"}>
            <PipelinePage />
          </div>

          <div className={page === "triage" ? "block" : "hidden"}>
            <TriagePage />
          </div>

          <div className={page === "telemetry" ? "block" : "hidden"}>
            <TelemetryPage />
          </div>
        </main>
      </div>
    </div>
  );
}

// MAKE SURE THIS LINE IS AT THE VERY BOTTOM OF App.tsx
export default App;