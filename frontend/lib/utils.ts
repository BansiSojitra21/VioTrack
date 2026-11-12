import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

export function formatClock(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const totalMilliseconds = Math.round(Math.max(0, seconds) * 1000);
  const totalSeconds = Math.floor(totalMilliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const remainSeconds = totalSeconds % 60;
  const milliseconds = totalMilliseconds % 1000;
  return `${minutes}:${remainSeconds.toString().padStart(2, "0")}.${milliseconds.toString().padStart(3, "0")}`;
}

export function formatVideoTimestamp(seconds: number | null | undefined) {
  return formatClock(seconds);
}

export function formatWhen(iso: string | null | undefined) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const day = date.getUTCDate().toString().padStart(2, "0");
  const month = (date.getUTCMonth() + 1).toString().padStart(2, "0");
  const year = date.getUTCFullYear();
  const hours = date.getUTCHours().toString().padStart(2, "0");
  const minutes = date.getUTCMinutes().toString().padStart(2, "0");
  return `${day}/${month}/${year} ${hours}:${minutes} UTC`;
}

export function formatDate(iso: string | null | undefined) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const day = date.getUTCDate().toString().padStart(2, "0");
  const month = (date.getUTCMonth() + 1).toString().padStart(2, "0");
  return `${day}/${month}/${date.getUTCFullYear()}`;
}

export function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export const CLASS_COLORS: Record<string, string> = {
  car: "#38bdf8",
  motorcycle: "#fbbf24",
  bus: "#c4b5fd",
  truck: "#34d399",
  bicycle: "#fb7185",
  pedestrian: "#e2e8f0",
};

export function classColor(name: string) {
  return CLASS_COLORS[name] ?? "#22d3ee";
}
