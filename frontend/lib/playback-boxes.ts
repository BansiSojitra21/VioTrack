import type { DetectionRecord } from "./types";

export const DISPLAY_VEHICLE_CLASSES = new Set(["car", "motorcycle", "bus", "truck", "bicycle"]);
export const TWO_WHEELER_CLASSES = new Set(["motorcycle", "bicycle"]);

export function detectionIoU(a: number[], b: number[]): number {
  const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const intersection = width * height;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(areaA + areaB - intersection, 1);
}

function boxCenter(bbox: number[]): [number, number] {
  return [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5];
}

function boxSize(bbox: number[]): [number, number] {
  return [Math.max(1, bbox[2] - bbox[0]), Math.max(1, bbox[3] - bbox[1])];
}

function centerDistance(a: number[], b: number[]): number {
  const [ax, ay] = boxCenter(a);
  const [bx, by] = boxCenter(b);
  return Math.hypot(ax - bx, ay - by);
}

/** Confidence must be a real 0..1 value — never Infinity/NaN on screen. */
export function sanitizeConfidence(value: unknown): number | null {
  const conf = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(conf) || conf < 0 || conf > 1) return null;
  return conf;
}

/** Reject broken / full-frame / zero-size boxes that create corner Infinity overlays. */
export function isValidBBox(bbox: number[] | null | undefined, frameW = 0, frameH = 0): boolean {
  if (!bbox || bbox.length < 4) return false;
  if (!bbox.every((value) => Number.isFinite(value))) return false;
  const width = bbox[2] - bbox[0];
  const height = bbox[3] - bbox[1];
  if (width < 6 || height < 6) return false;
  if (width > 4000 || height > 4000) return false;
  if (frameW > 0 && frameH > 0) {
    const area = width * height;
    const frameArea = frameW * frameH;
    if (area / frameArea > 0.22) return false;
    if (width / frameW > 0.55 || height / frameH > 0.7) return false;
  }
  return true;
}

/**
 * True when a box is absurdly large for its class (e.g. bike covering half the frame).
 * Those boxes hide every other vehicle via NMS containment.
 */
export function isOversizedVehicleBox(
  bbox: number[],
  className: string,
  frameW = 0,
  frameH = 0,
): boolean {
  if (!isValidBBox(bbox) || frameW <= 0 || frameH <= 0) return false;
  const [w, h] = boxSize(bbox);
  const areaRatio = (w * h) / Math.max(frameW * frameH, 1);
  if (TWO_WHEELER_CLASSES.has(className)) {
    return areaRatio > 0.045 || w / frameW > 0.18 || h / frameH > 0.28;
  }
  return areaRatio > 0.2 || w / frameW > 0.5 || h / frameH > 0.65;
}

/**
 * Mild upward pad for short bike-body crops only.
 * Never scale by width — that previously turned medium boxes into full-frame covers.
 */
export function expandTwoWheelerBBox(
  bbox: number[],
  className: string,
  frameW = 0,
  frameH = 0,
): number[] {
  if (!TWO_WHEELER_CLASSES.has(className) || !isValidBBox(bbox)) return bbox.slice();
  if (isOversizedVehicleBox(bbox, className, frameW, frameH)) return bbox.slice();
  const [w, h] = boxSize(bbox);
  // Only nudge compact, short crops (typical rear bike without rider head).
  if (w > 130 || h > 160) return bbox.slice();
  if (h / Math.max(w, 1) >= 1.35) return bbox.slice();
  const padTop = Math.min(h * 0.3, 40);
  const padBottom = Math.min(h * 0.05, 8);
  const padX = Math.min(w * 0.06, 10);
  let x1 = bbox[0] - padX;
  let y1 = bbox[1] - padTop;
  let x2 = bbox[2] + padX;
  let y2 = bbox[3] + padBottom;
  if (frameW > 0) {
    x1 = Math.max(0, x1);
    x2 = Math.min(frameW, x2);
  }
  if (frameH > 0) {
    y1 = Math.max(0, y1);
    y2 = Math.min(frameH, y2);
  }
  const out = [x1, y1, x2, y2];
  if (!isValidBBox(out, frameW, frameH)) return bbox.slice();
  if (isOversizedVehicleBox(out, className, frameW, frameH)) return bbox.slice();
  return out;
}

export function displayClassLabel(className: string | null | undefined): string {
  const key = (className || "").toLowerCase();
  if (key === "motorcycle") return "Bike";
  if (key === "bicycle") return "Bicycle";
  if (key === "car") return "Car";
  if (key === "truck") return "Truck";
  if (key === "bus") return "Bus";
  if (!key) return "Vehicle";
  return key.charAt(0).toUpperCase() + key.slice(1);
}

export function sanitizeDetection(
  detection: DetectionRecord,
  frameW = 0,
  frameH = 0,
): DetectionRecord | null {
  const confidence = sanitizeConfidence(detection.confidence);
  if (confidence == null) return null;
  const rawBBox = detection.bbox.map((value) => Number(value));
  if (!isValidBBox(rawBBox, frameW, frameH)) return null;
  if (isOversizedVehicleBox(rawBBox, detection.class_name, frameW, frameH)) return null;
  const bbox = expandTwoWheelerBBox(rawBBox, detection.class_name, frameW, frameH);
  if (!isValidBBox(bbox, frameW, frameH)) return null;
  if (isOversizedVehicleBox(bbox, detection.class_name, frameW, frameH)) return null;
  return {
    ...detection,
    confidence,
    bbox,
    plate_number: displayPlate(detection.plate_number),
  };
}

/** Move a box by centroid velocity; keep width/height from the latest sample. */
export function translateBBox(
  earlier: number[],
  later: number[],
  fromTime: number,
  toTime: number,
  time: number,
): number[] {
  if (!isValidBBox(earlier) || !isValidBBox(later)) return later.slice();
  const span = toTime - fromTime;
  if (!Number.isFinite(span) || span <= 1e-4) return later.slice();
  const dt = time - toTime;
  if (dt <= 0) return later.slice();
  const [cx0, cy0] = boxCenter(earlier);
  const [cx1, cy1] = boxCenter(later);
  const [w, h] = boxSize(later);
  const vx = (cx1 - cx0) / span;
  const vy = (cy1 - cy0) / span;
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return later.slice();
  const cappedDt = Math.min(dt, Math.max(0.12, span));
  const cx = cx1 + vx * cappedDt;
  const cy = cy1 + vy * cappedDt;
  const out = [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5];
  return isValidBBox(out) ? out : later.slice();
}

/**
 * Human-like size change: trucks grow as they approach, but never jump
 * from a tiny distant box to a huge near box in one step.
 */
export function easeBBox(
  previous: number[],
  next: number[],
  dtSec: number,
  positionAlpha = 0.45,
): number[] {
  if (!isValidBBox(previous)) return next.slice();
  if (!isValidBBox(next)) return previous.slice();
  const dt = Math.max(1e-3, Math.min(dtSec, 0.5));
  const [cx0, cy0] = boxCenter(previous);
  const [cx1, cy1] = boxCenter(next);
  const [w0, h0] = boxSize(previous);
  const [w1, h1] = boxSize(next);

  // Cap growth/shrink so approach feels continuous, but let the center follow hard.
  const maxScale = 1 + 0.9 * dt;
  const minScale = 1 / maxScale;
  const targetScaleW = Math.min(maxScale, Math.max(minScale, w1 / w0));
  const targetScaleH = Math.min(maxScale, Math.max(minScale, h1 / h0));
  const sizeAlpha = Math.max(0, Math.min(1, positionAlpha));
  const w = w0 * (1 + (targetScaleW - 1) * sizeAlpha);
  const h = h0 * (1 + (targetScaleH - 1) * sizeAlpha);
  const a = Math.max(0, Math.min(1, positionAlpha));
  const cx = cx0 * (1 - a) + cx1 * a;
  const cy = cy0 * (1 - a) + cy1 * a;
  const out = [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5];
  return isValidBBox(out) ? out : previous.slice();
}

/** True timeline interpolation between two real detections — must reach the target at t=1. */
export function lerpBBox(earlier: number[], later: number[], progress: number): number[] {
  const t = Math.max(0, Math.min(1, progress));
  if (!isValidBBox(earlier)) return later.slice();
  if (!isValidBBox(later)) return earlier.slice();
  return [
    earlier[0] + (later[0] - earlier[0]) * t,
    earlier[1] + (later[1] - earlier[1]) * t,
    earlier[2] + (later[2] - earlier[2]) * t,
    earlier[3] + (later[3] - earlier[3]) * t,
  ];
}

/** Fraction of the smaller box covered by the larger (1 = fully nested). */
export function boxContainment(a: number[], b: number[]): number {
  if (!isValidBBox(a) || !isValidBBox(b)) return 0;
  const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const intersection = width * height;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const smaller = Math.min(areaA, areaB);
  return smaller > 0 ? intersection / smaller : 0;
}

/**
 * True when two boxes are the same physical vehicle.
 * Nested window/plate crops inside a full-car box have low IoU but high containment
 * (e.g. #2562 window vs #2648 body) — those must collapse to one box.
 */
export function boxesOverlapTooMuch(a: number[], b: number[]): boolean {
  if (!isValidBBox(a) || !isValidBBox(b)) return false;
  if (boxContainment(a, b) >= 0.5) return true;
  const overlap = detectionIoU(a, b);
  if (overlap >= 0.22) return true;
  const [aw, ah] = boxSize(a);
  const [bw, bh] = boxSize(b);
  const radius = Math.max(aw, ah, bw, bh) * 0.55;
  return overlap >= 0.08 && centerDistance(a, b) <= radius;
}

/** Reject associating two boxes when size differs too much (tiny ≠ approaching truck). */
export function sizesCompatible(a: number[], b: number[]): boolean {
  if (!isValidBBox(a) || !isValidBBox(b)) return false;
  const [aw, ah] = boxSize(a);
  const [bw, bh] = boxSize(b);
  const areaA = aw * ah;
  const areaB = bw * bh;
  const ratio = Math.max(areaA, areaB) / Math.max(Math.min(areaA, areaB), 1);
  return ratio <= 6;
}

const CLASS_RANK: Record<string, number> = {
  car: 4,
  bus: 3,
  truck: 2,
  motorcycle: 1,
  bicycle: 0,
};

/**
 * Keep one display box for a vehicle. Prefer car over truck/bike when YOLO
 * double-fires on the same hatchback; otherwise prefer body size / confidence.
 */
export function preferVehicleBox(a: DetectionRecord, b: DetectionRecord): DetectionRecord {
  const [aw, ah] = boxSize(a.bbox);
  const [bw, bh] = boxSize(b.bbox);
  const areaA = aw * ah;
  const areaB = bw * bh;
  const nested = boxContainment(a.bbox, b.bbox) >= 0.5;
  const confA = sanitizeConfidence(a.confidence) ?? 0;
  const confB = sanitizeConfidence(b.confidence) ?? 0;
  const rankA = CLASS_RANK[a.class_name] ?? 0;
  const rankB = CLASS_RANK[b.class_name] ?? 0;
  let winner = a;
  if (a.class_name !== b.class_name && minConfClose(confA, confB)) {
    winner = rankA >= rankB ? a : b;
  } else if (nested) {
    // Same physical vehicle: prefer higher-ranked class, then larger body.
    if (rankA !== rankB && minConfClose(confA, confB)) winner = rankA > rankB ? a : b;
    else winner = areaA >= areaB ? a : b;
  } else if (confB > confA) {
    winner = b;
  } else if (confB === confA && areaB > areaA) {
    winner = b;
  }
  const plate =
    displayPlate(winner.plate_number) ??
    displayPlate(a.plate_number) ??
    displayPlate(b.plate_number);
  return {
    ...winner,
    confidence: Math.max(confA, confB),
    plate_number: plate,
  };
}

function minConfClose(a: number, b: number): boolean {
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return hi <= 0 ? true : lo >= hi * 0.55;
}

/** Greedy NMS that collapses nested / overlapping IDs into one vehicle box. */
export function dedupeVehicleBoxes(detections: DetectionRecord[]): DetectionRecord[] {
  const sorted = [...detections].sort((a, b) => {
    const confA = sanitizeConfidence(a.confidence) ?? 0;
    const confB = sanitizeConfidence(b.confidence) ?? 0;
    if (confB !== confA) return confB - confA;
    const [aw, ah] = boxSize(a.bbox);
    const [bw, bh] = boxSize(b.bbox);
    return bw * bh - aw * ah;
  });
  const unique: DetectionRecord[] = [];
  for (const detection of sorted) {
    const conf = sanitizeConfidence(detection.confidence);
    if (conf == null || !isValidBBox(detection.bbox)) continue;
    const index = unique.findIndex((kept) => {
      if (
        detection.tracking_id != null &&
        kept.tracking_id != null &&
        Math.abs(detection.tracking_id) === Math.abs(kept.tracking_id)
      ) {
        return true;
      }
      return boxesOverlapTooMuch(kept.bbox, detection.bbox);
    });
    if (index < 0) {
      unique.push({ ...detection, confidence: conf, plate_number: displayPlate(detection.plate_number) });
      continue;
    }
    unique[index] = preferVehicleBox(unique[index], { ...detection, confidence: conf });
  }
  return unique;
}

export function groupDetections(items: DetectionRecord[]) {
  const ordered = [...items].sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  const byTrack = new Map<number, DetectionRecord[]>();
  const untracked: DetectionRecord[] = [];
  for (const raw of ordered) {
    const detection = sanitizeDetection(raw);
    if (!detection) continue;
    if (!DISPLAY_VEHICLE_CLASSES.has(detection.class_name)) continue;
    if (detection.tracking_id == null) {
      untracked.push(detection);
      continue;
    }
    const track = byTrack.get(detection.tracking_id);
    if (track) track.push(detection);
    else byTrack.set(detection.tracking_id, [detection]);
  }
  for (const [trackId, trackDetections] of byTrack) {
    const deduped: DetectionRecord[] = [];
    for (const detection of trackDetections) {
      const previous = deduped[deduped.length - 1];
      if (previous && Math.abs(previous.timestamp_sec - detection.timestamp_sec) < 1e-3) {
        if (detection.confidence >= previous.confidence) deduped[deduped.length - 1] = detection;
        continue;
      }
      deduped.push(detection);
    }
    byTrack.set(trackId, deduped);
  }
  return { byTrack, untracked };
}

export function medianGap(byTrack: Map<number, DetectionRecord[]>) {
  const gaps: number[] = [];
  for (const trackDetections of byTrack.values()) {
    for (let index = 1; index < trackDetections.length; index += 1) {
      const gap = trackDetections[index].timestamp_sec - trackDetections[index - 1].timestamp_sec;
      if (gap > 0 && gap < 10) gaps.push(gap);
    }
  }
  if (gaps.length === 0) return 0.5;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

export function isCompletePlate(raw: string | null | undefined): boolean {
  if (!raw) return false;
  const cleaned = raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
  if (cleaned.length < 6 || cleaned.length > 15) return false;
  const hasLetter = /[A-Z]/.test(cleaned);
  const hasDigit = /\d/.test(cleaned);
  return hasLetter && hasDigit;
}

export function displayPlate(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const cleaned = raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
  return isCompletePlate(cleaned) ? cleaned : null;
}

function plateNearTime(trackDetections: DetectionRecord[], _time: number, indexHint: number) {
  for (let offset = 0; offset < trackDetections.length; offset += 1) {
    const left = indexHint - 1 - offset;
    const right = indexHint + offset;
    if (left >= 0) {
      const plate = displayPlate(trackDetections[left].plate_number);
      if (plate) return plate;
    }
    if (right < trackDetections.length) {
      const plate = displayPlate(trackDetections[right].plate_number);
      if (plate) return plate;
    }
  }
  return null;
}

export function boxesAtPlayhead(
  byTrack: Map<number, DetectionRecord[]>,
  untracked: DetectionRecord[],
  time: number,
  detectionGap: number,
  minConfidence: number,
) {
  // Hold long enough to bridge detector gaps (no blink), coast with velocity so boxes still move.
  const holdWindow = Math.min(0.75, Math.max(0.45, detectionGap * 2.5));
  const lookAheadWindow = Math.min(0.12, Math.max(0.04, detectionGap * 0.5));
  const maxInterpolationGap = Math.max(3, detectionGap * 8);
  const active: DetectionRecord[] = [];

  for (const trackDetections of byTrack.values()) {
    if (trackDetections.length === 0) continue;

    let lo = 0;
    let hi = trackDetections.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (trackDetections[mid].timestamp_sec < time) lo = mid + 1;
      else hi = mid;
    }

    const before = lo > 0 ? trackDetections[lo - 1] : null;
    const after = lo < trackDetections.length ? trackDetections[lo] : null;
    const plate = plateNearTime(trackDetections, time, lo);
    let candidate: DetectionRecord | null = null;

    if (before && after) {
      const span = after.timestamp_sec - before.timestamp_sec;
      if (span <= maxInterpolationGap && Number.isFinite(span)) {
        if (span <= 1e-4) {
          candidate = after;
        } else if (sizesCompatible(before.bbox, after.bbox)) {
          const progress = Math.max(0, Math.min(1, (time - before.timestamp_sec) / span));
          const conf =
            sanitizeConfidence(
              before.confidence + (after.confidence - before.confidence) * progress,
            ) ?? before.confidence;
          candidate = {
            ...before,
            id: `${before.id}-i-${time.toFixed(3)}`,
            timestamp_sec: time,
            confidence: conf,
            bbox: lerpBBox(before.bbox, after.bbox, progress),
            plate_number: plate ?? displayPlate(before.plate_number) ?? displayPlate(after.plate_number),
          };
        } else {
          // Size jump (distant speck → near truck): stick to the nearer sample in time.
          candidate =
            time - before.timestamp_sec <= after.timestamp_sec - time ? before : after;
        }
      }
    }

    if (!candidate && before && time >= before.timestamp_sec && time - before.timestamp_sec <= holdWindow) {
      const prior = lo > 1 ? trackDetections[lo - 2] : null;
      if (
        prior &&
        before.timestamp_sec - prior.timestamp_sec > 1e-3 &&
        time > before.timestamp_sec &&
        sizesCompatible(prior.bbox, before.bbox)
      ) {
        candidate = {
          ...before,
          id: `${before.id}-x-${time.toFixed(3)}`,
          timestamp_sec: time,
          bbox: translateBBox(prior.bbox, before.bbox, prior.timestamp_sec, before.timestamp_sec, time),
          plate_number: plate ?? displayPlate(before.plate_number),
        };
      } else {
        candidate = {
          ...before,
          id: `${before.id}-h-${time.toFixed(3)}`,
          timestamp_sec: time,
          plate_number: plate ?? displayPlate(before.plate_number),
        };
      }
    }

    if (!candidate && !before && after && after.timestamp_sec - time <= lookAheadWindow) {
      candidate = {
        ...after,
        id: `${after.id}-h-${time.toFixed(3)}`,
        timestamp_sec: time,
        plate_number: plate ?? displayPlate(after.plate_number),
      };
    }

    const clean = candidate ? sanitizeDetection(candidate) : null;
    if (clean && clean.confidence >= minConfidence) active.push(clean);
  }

  for (const detection of untracked) {
    const clean = sanitizeDetection(detection);
    if (!clean || clean.confidence < minConfidence) continue;
    if (Math.abs(clean.timestamp_sec - time) > holdWindow) continue;
    if (active.some((box) => boxesOverlapTooMuch(box.bbox, clean.bbox))) continue;
    active.push(clean);
  }

  return dedupeVehicleBoxes(active);
}

export function overlayLabel(
  detection: Pick<DetectionRecord, "tracking_id" | "confidence" | "plate_number" | "class_name">,
) {
  const classLabel = displayClassLabel(detection.class_name);
  const trackLabel =
    detection.tracking_id != null
      ? detection.tracking_id < 0
        ? `#${Math.abs(detection.tracking_id)}`
        : `#${detection.tracking_id}`
      : "#—";
  const conf = sanitizeConfidence(detection.confidence);
  const percent = conf == null ? null : Math.round(conf * 100);
  const primary =
    percent == null ? `${classLabel} ${trackLabel}` : `${classLabel} ${trackLabel}  ${percent}%`;
  const plate = displayPlate(detection.plate_number);
  return { primary, plate };
}

export interface StableTrackState {
  key: number;
  box: DetectionRecord;
  lastSeenSec: number;
  hits: number;
  visible: boolean;
  /** Previous sample used to coast the box while a detection is briefly missing. */
  prevBBox?: number[];
  prevSeenSec?: number;
}

function trackKey(detection: DetectionRecord): number | null {
  if (detection.tracking_id == null) return null;
  return detection.tracking_id;
}

/**
 * Human tracker rules:
 * 1) one steady box per vehicle
 * 2) follow motion smoothly
 * 3) grow/shrink gradually as the vehicle approaches/leaves
 * 4) never show Infinity/NaN or full-frame garbage boxes
 * 5) brief occlusion keeps the last real box (no drifting ghosts)
 */
export function stabilizeOverlay(
  previous: Map<number, StableTrackState>,
  incoming: DetectionRecord[],
  time: number,
  options?: {
    persistSec?: number;
    alpha?: number;
    confirmHits?: number;
    minDisplayConf?: number;
    frameW?: number;
    frameH?: number;
  },
): { boxes: DetectionRecord[]; state: Map<number, StableTrackState> } {
  const persistSec = options?.persistSec ?? 0.55;
  // Follow detections closely — low alpha left boxes lagging behind moving cars.
  const baseAlpha = options?.alpha ?? 0.8;
  const confirmHits = options?.confirmHits ?? 2;
  const minDisplayConf = options?.minDisplayConf ?? 0.35;
  const frameW = options?.frameW ?? 0;
  const frameH = options?.frameH ?? 0;
  const nextState = new Map<number, StableTrackState>();
  const claimed = new Set<number>();
  // Already-visible tracks may dip slightly under threshold for a frame without disappearing.
  const holdDisplayConf = minDisplayConf * 0.75;

  const sorted = incoming
    .map((detection) => sanitizeDetection(detection, frameW, frameH))
    .filter((detection): detection is DetectionRecord => detection != null)
    .filter((detection) => detection.confidence >= holdDisplayConf)
    .sort((a, b) => b.confidence - a.confidence);

  for (const detection of sorted) {
    const key = trackKey(detection);
    if (key == null) continue;
    if (claimed.has(key)) continue;

    let matchedKey = key;
    let prior = previous.get(key);
    if (!prior) {
      for (const [existingKey, existing] of previous.entries()) {
        if (claimed.has(existingKey) || !existing.visible) continue;
        // Containment (window ⊂ car) counts even when area ratio is large.
        if (boxesOverlapTooMuch(existing.box.bbox, detection.bbox)) {
          matchedKey = existingKey;
          prior = existing;
          break;
        }
      }
    }

    const overlapping = [...nextState.values()].find(
      (state) => state.visible && boxesOverlapTooMuch(state.box.bbox, detection.bbox),
    );
    if (overlapping) {
      // Fold nested crop into the already-accepted vehicle box (keep plate).
      overlapping.box = preferVehicleBox(overlapping.box, detection);
      continue;
    }

    claimed.add(matchedKey);
    if (prior && isValidBBox(prior.box.bbox, frameW, frameH)) {
      const dt = Math.max(1e-3, time - prior.lastSeenSec);
      if (!sizesCompatible(prior.box.bbox, detection.bbox)) {
        // Nested window/plate crop of the same vehicle: keep the body box.
        if (boxContainment(prior.box.bbox, detection.bbox) >= 0.5) {
          const merged = preferVehicleBox(prior.box, detection);
          const cleanNested = sanitizeDetection(
            { ...merged, timestamp_sec: time },
            frameW,
            frameH,
          );
          if (cleanNested) {
            nextState.set(matchedKey, {
              key: matchedKey,
              lastSeenSec: time,
              hits: prior.hits + 1,
              visible: true,
              box: cleanNested,
              prevBBox: prior.box.bbox.slice(),
              prevSeenSec: prior.lastSeenSec,
            });
          }
          continue;
        }
        // Treat as a new object rather than morphing a speck into a truck.
        nextState.set(key, {
          key,
          lastSeenSec: time,
          hits: 1,
          visible: confirmHits <= 1,
          box: detection,
        });
        continue;
      }
      const alpha = Math.min(1, Math.max(0.55, baseAlpha));
      const smoothed = easeBBox(prior.box.bbox, detection.bbox, dt, alpha);
      const hits = prior.hits + 1;
      const clean = sanitizeDetection(
        {
          ...detection,
          tracking_id: prior.box.tracking_id ?? detection.tracking_id,
          bbox: smoothed,
          plate_number: displayPlate(detection.plate_number) ?? displayPlate(prior.box.plate_number),
          confidence: detection.confidence,
        },
        frameW,
        frameH,
      );
      if (!clean) continue;
      nextState.set(matchedKey, {
        key: matchedKey,
        lastSeenSec: time,
        hits,
        visible: hits >= confirmHits || prior.visible,
        box: clean,
        prevBBox: prior.box.bbox.slice(),
        prevSeenSec: prior.lastSeenSec,
      });
    } else {
      nextState.set(matchedKey, {
        key: matchedKey,
        lastSeenSec: time,
        hits: 1,
        visible: confirmHits <= 1,
        box: detection,
      });
    }
  }

  for (const [key, prior] of previous.entries()) {
    if (nextState.has(key) || claimed.has(key)) continue;
    if (!prior.visible) continue;
    const age = time - prior.lastSeenSec;
    if (age > persistSec) continue;
    // Coast with last motion instead of freezing — frozen holds looked stuck on moving cars.
    let coastBBox = prior.box.bbox;
    if (
      prior.prevBBox &&
      prior.prevSeenSec != null &&
      prior.lastSeenSec - prior.prevSeenSec > 1e-3 &&
      sizesCompatible(prior.prevBBox, prior.box.bbox)
    ) {
      coastBBox = translateBBox(
        prior.prevBBox,
        prior.box.bbox,
        prior.prevSeenSec,
        prior.lastSeenSec,
        time,
      );
    }
    const held = sanitizeDetection(
      { ...prior.box, timestamp_sec: time, bbox: coastBBox, confidence: prior.box.confidence },
      frameW,
      frameH,
    );
    if (!held || held.confidence < holdDisplayConf) continue;
    if (
      [...nextState.values()].some(
        (state) => state.visible && boxesOverlapTooMuch(state.box.bbox, held.bbox),
      )
    ) {
      continue;
    }
    nextState.set(key, { ...prior, box: held });
  }

  // Sticky display: once visible, keep showing through brief dips instead of blinking off.
  const boxes = dedupeVehicleBoxes(
    [...nextState.values()]
      .filter((state) => state.visible)
      .map((state) => state.box)
      .filter((box) => {
        const conf = sanitizeConfidence(box.confidence);
        if (conf == null || !isValidBBox(box.bbox, frameW, frameH)) return false;
        const wasVisible = [...previous.values()].some(
          (prior) =>
            prior.visible &&
            prior.box.tracking_id != null &&
            box.tracking_id != null &&
            Math.abs(prior.box.tracking_id) === Math.abs(box.tracking_id),
        );
        return conf >= (wasVisible ? holdDisplayConf : minDisplayConf);
      }),
  );
  return { boxes, state: nextState };
}
