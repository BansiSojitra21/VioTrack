export interface Preferences {
  conf: number;
  iou: number;
  imgsz: number;
  frameStride: number;
  trackBuffer: number;
  trackerType: "bytetrack" | "botsort";
  debugMode: boolean;
  targetFps: number | null;
  maxFrames: number;
  saveAnnotated: boolean;
  accent: "violet" | "cyan" | "blue";
}

export const defaultPreferences: Preferences = {
  conf: 0.25,
  iou: 0.5,
  imgsz: 960,
  frameStride: 1,
  trackBuffer: 30,
  trackerType: "bytetrack",
  debugMode: false,
  targetFps: null,
  maxFrames: 20000,
  saveAnnotated: true,
  accent: "violet",
};

const KEY = "trafficai-preferences";

export function loadPreferences(): Preferences {
  if (typeof window === "undefined") return defaultPreferences;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return defaultPreferences;
    const stored = JSON.parse(raw) as Partial<Preferences>;
    const usedOldDetectionDefaults =
      (stored.imgsz == null || stored.imgsz === 960) &&
      stored.conf === 0.35 &&
      stored.iou === 0.45;
    return {
      ...defaultPreferences,
      ...stored,
      conf: usedOldDetectionDefaults ? defaultPreferences.conf : stored.conf ?? defaultPreferences.conf,
      iou: usedOldDetectionDefaults ? defaultPreferences.iou : stored.iou ?? defaultPreferences.iou,
      imgsz: usedOldDetectionDefaults ? defaultPreferences.imgsz : stored.imgsz ?? defaultPreferences.imgsz,
    };
  } catch {
    return defaultPreferences;
  }
}

export function savePreferences(preferences: Preferences) {
  window.localStorage.setItem(KEY, JSON.stringify(preferences));
  document.documentElement.dataset.accent = preferences.accent;
}
