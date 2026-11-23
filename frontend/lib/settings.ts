/** Fixed analysis defaults — not user-editable. */
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

/** Always returns the locked defaults (localStorage overrides are ignored). */
export function loadPreferences(): Preferences {
  return { ...defaultPreferences };
}
