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
  // Locked balance: enough recall for bikes/cars, still filters weak ghosts.
  conf: 0.32,
  iou: 0.5,
  imgsz: 640,
  frameStride: 2,
  trackBuffer: 30,
  trackerType: "bytetrack",
  debugMode: false,
  targetFps: null,
  maxFrames: 20000,
  saveAnnotated: false,
  accent: "violet",
};

/** Always returns the locked defaults (localStorage overrides are ignored). */
export function loadPreferences(): Preferences {
  return { ...defaultPreferences };
}
