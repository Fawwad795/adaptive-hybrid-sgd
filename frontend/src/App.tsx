import { useEffect, useMemo, useState } from "react";
import { Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import Header from "./components/Header";
import RunHistoryRail from "./components/RunHistoryRail";
import { RunHistoryProvider, useRunHistory } from "./components/RunHistoryContext";
import BenchmarksPage from "./pages/BenchmarksPage";
import ComparePage from "./pages/ComparePage";
import RunStudioPage from "./pages/RunStudioPage";

function HealthBar() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const response = await fetch(`${api.baseUrl}/health`);
        if (!cancelled) {
          setOnline(response.ok);
        }
      } catch {
        if (!cancelled) {
          setOnline(false);
        }
      }
    }
    void check();
    const interval = window.setInterval(() => {
      void check();
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  const { runs } = useRunHistory();
  const activeCount = useMemo(
    () => runs.filter((run) => run.status === "running" || run.status === "queued").length,
    [runs],
  );

  return (
    <Header
      apiBase={api.baseUrl}
      apiOnline={online}
      runCount={runs.length}
      activeRunCount={activeCount}
    />
  );
}

function Layout() {
  const location = useLocation();
  const showRail = location.pathname === "/" || location.pathname.startsWith("/compare");
  return (
    <div className="flex min-h-screen flex-col bg-canvas text-zinc-200">
      <HealthBar />
      <div className="flex flex-1">
        {showRail ? <RunHistoryRail /> : null}
        <main className="flex min-w-0 flex-1 flex-col">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <RunHistoryProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<RunStudioPage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="benchmarks" element={<BenchmarksPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </RunHistoryProvider>
  );
}
