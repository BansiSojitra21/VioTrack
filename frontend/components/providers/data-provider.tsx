"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { apiBase, demoFallbackAllowed, loadLivePayload } from "@/lib/api";
import { cloneSample } from "@/lib/demo-data";
import type { AppPayload } from "@/lib/types";

interface DataContextValue {
  status: "loading" | "ready" | "error";
  data: AppPayload | null;
  error: string | null;
  refresh: () => Promise<void>;
}

const DataContext = createContext<DataContextValue | null>(null);

export function DataProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<DataContextValue["status"]>("loading");
  const [data, setData] = useState<AppPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!apiBase()) {
      if (!demoFallbackAllowed()) {
        setData(null);
        setError("Set NEXT_PUBLIC_API_URL to the FastAPI service. Demo fallback is turned off, so sample records are hidden.");
        setStatus("error");
        return;
      }
      setData(cloneSample());
      setError(null);
      setStatus("ready");
      return;
    }
    try {
      const live = await loadLivePayload();
      setData(live);
      setError(null);
      setStatus("ready");
    } catch (err) {
      const message = err instanceof Error ? err.message : "The inference API could not be reached.";
      if (!demoFallbackAllowed()) {
        setData(null);
        setError(message);
        setStatus("error");
        return;
      }
      const sample = cloneSample();
      sample.notice = `${message} Sample data is shown instead. No detection was run.`;
      setData(sample);
      setError(null);
      setStatus("ready");
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refresh();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  const hasActiveAnalysis = Boolean(
    data?.videos.some((video) =>
      video.latest_job && ["pending", "processing", "cancelling"].includes(video.latest_job.status),
    ),
  );

  useEffect(() => {
    if (!hasActiveAnalysis) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [hasActiveAnalysis, refresh]);

  return <DataContext.Provider value={{ status, data, error, refresh }}>{children}</DataContext.Provider>;
}

export function useTrafficData() {
  const value = useContext(DataContext);
  if (!value) throw new Error("useTrafficData must be used inside DataProvider");
  return value;
}
