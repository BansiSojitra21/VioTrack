import type {
  AnalysisConfig,
  AppPayload,
  HealthInfo,
  JobRecord,
  ModelInfo,
} from "@/lib/types";

export function apiBase() {
  const value = process.env.NEXT_PUBLIC_API_URL?.trim() ?? "";
  return value ? value.replace(/\/$/, "") : null;
}

export function demoFallbackAllowed() {
  return process.env.NEXT_PUBLIC_DEMO_MODE !== "false";
}

export function liveStreamUrl() {
  const value = process.env.NEXT_PUBLIC_LIVE_STREAM_URL?.trim() ?? "";
  return value || null;
}

export function mediaUrl(path: string | null | undefined) {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = apiBase();
  if (!base) return null;
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

async function readError(response: Response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) return payload.detail.map((item: { msg?: string }) => item.msg).join(" ");
  } catch {
    return response.statusText;
  }
  return response.statusText;
}

export async function apiSend<T>(path: string, init?: RequestInit): Promise<T> {
  const base = apiBase();
  if (!base) {
    throw new Error("NEXT_PUBLIC_API_URL is empty. The inference service was not called.");
  }
  const response = await fetch(`${base}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  if (response.status === 204) return undefined as T;
  const type = response.headers.get("content-type") ?? "";
  if (type.includes("application/json")) return response.json() as Promise<T>;
  return response as T;
}

export async function downloadApi(path: string, filename: string) {
  const base = apiBase();
  if (!base) throw new Error("Inference API URL is not configured.");
  const response = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await readError(response));
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function loadLivePayload(): Promise<AppPayload> {
  const [health, summary, traffic, videos, violations, detections, model, history] = await Promise.all([
    apiSend<HealthInfo>("/api/health"),
    apiSend<AppPayload["summary"]>("/api/analytics/summary"),
    apiSend<AppPayload["traffic"]>("/api/analytics/traffic"),
    apiSend<{ items: AppPayload["videos"] }>("/api/videos"),
    apiSend<{ items: AppPayload["violations"] }>("/api/violations"),
    apiSend<{ items: AppPayload["detections"] }>("/api/detections?limit=400"),
    apiSend<ModelInfo>("/api/model/info"),
    apiSend<{ items: AppPayload["history"] }>("/api/reports/history"),
  ]);
  return {
    source: "live",
    notice: null,
    summary,
    traffic,
    videos: videos.items,
    violations: violations.items,
    detections: detections.items,
    tracks: [],
    model,
    history: history.items,
    health,
  };
}

export async function startAnalysis(videoId: string, config: AnalysisConfig) {
  return apiSend<JobRecord>("/api/analysis/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, config }),
  });
}
