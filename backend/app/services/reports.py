"""CSV and PDF exports built from stored analysis records."""

from __future__ import annotations

import csv
import io
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.services import repository


def sanitize_summary(summary: dict | None) -> dict | None:
    if summary is None:
        return None
    result = dict(summary)
    violation_counts = result.get("violation_counts")
    if isinstance(violation_counts, dict):
        counts = dict(violation_counts)
        removed_count = int(counts.pop("seatbelt", 0) or 0)
        result["violation_counts"] = counts
        if removed_count:
            result["violations_total"] = max(0, int(result.get("violations_total", 0)) - removed_count)
    if isinstance(result.get("modules"), list):
        result["modules"] = [module for module in result["modules"] if module.get("name") != "seatbelt"]
    return result


def violations_csv(rows: list[dict] | None = None) -> str:
    records = rows if rows is not None else repository.list_violations()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "violation_id",
            "video_id",
            "job_id",
            "tracking_id",
            "plate_number",
            "vehicle_category",
            "violation_type",
            "timestamp_sec",
            "frame_number",
            "confidence",
            "review_status",
            "rule",
            "explanation",
        ]
    )
    for row in records:
        evidence = row.get("evidence") or {}
        writer.writerow(
            [
                row["id"],
                row["video_id"],
                row["job_id"],
                row.get("tracking_id"),
                evidence.get("plate_number", ""),
                row["vehicle_category"],
                row["violation_type"],
                f"{row['timestamp_sec']:.3f}",
                row["frame_number"],
                f"{row['confidence']:.4f}",
                row["review_status"],
                evidence.get("rule", ""),
                evidence.get("explanation", ""),
            ]
        )
    return buffer.getvalue()


def detections_csv(job_id: str | None = None) -> str:
    rows = repository.query_detections(job_id=job_id, limit=100000, offset=0)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["detection_id", "job_id", "video_id", "frame_number", "timestamp_sec", "class_name", "confidence", "tracking_id", "plate_number", "x1", "y1", "x2", "y2"]
    )
    for row in rows:
        writer.writerow(
            [
                row["id"],
                row["job_id"],
                row["video_id"],
                row["frame_number"],
                f"{row['timestamp_sec']:.3f}",
                row["class_name"],
                f"{row['confidence']:.4f}",
                row.get("tracking_id"),
                row.get("plate_number") or "",
                f"{row['x1']:.2f}",
                f"{row['y1']:.2f}",
                f"{row['x2']:.2f}",
                f"{row['y2']:.2f}",
            ]
        )
    return buffer.getvalue()


def build_pdf(job_id: str | None = None) -> bytes:
    job = repository.get_job(job_id) if job_id else None
    video = repository.get_video(job["video_id"]) if job else None
    violations = repository.list_violations(video_id=video["id"] if video else None)
    if job:
        violations = [row for row in violations if row["job_id"] == job["id"]]
    summary = sanitize_summary(json.loads(job["summary_json"])) if job and job.get("summary_json") else {}
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="TrafficAI analysis report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("TrafficAI — traffic analysis report", styles["Title"]),
        Paragraph("Potential violations require human verification. This report does not issue penalties.", styles["Italic"]),
        Spacer(1, 12),
    ]
    if video:
        story.append(Paragraph(f"Video: {escape(video['original_name'])}", styles["Heading2"]))
        story.append(
            Paragraph(
                f"Duration {video.get('duration_sec') or 'unknown'} s · {video.get('width')}×{video.get('height')} · {video.get('fps')} fps",
                styles["BodyText"],
            )
        )
    if summary:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Detection summary", styles["Heading2"]))
        model = summary.get("model") or {}
        story.append(
            Paragraph(
                f"Vehicles tracked: {summary.get('vehicles_detected', 0)}. "
                f"Potential violations: {summary.get('violations_total', 0)}. "
                f"Frames processed: {summary.get('frames_processed', 0)} in {summary.get('processing_seconds', 0)} s.",
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                f"Model {escape(str(model.get('name', 'n/a')))} · conf {model.get('conf')} · IoU {model.get('iou')} · input {model.get('input_size')}",
                styles["BodyText"],
            )
        )
        for module in summary.get("modules") or []:
            story.append(
                Paragraph(
                    f"{escape(module['name'])}: {escape(module['status'])} — {escape(module['message'])} ({module['count']})",
                    styles["BodyText"],
                )
            )
    story.append(Spacer(1, 10))
    story.append(Paragraph("Violation records", styles["Heading2"]))
    table_data = [["Type", "Track", "Class", "Time (s)", "Confidence", "Review"]]
    for row in violations[:40]:
        table_data.append(
            [
                row["violation_type"],
                str(row.get("tracking_id") or "—"),
                row["vehicle_category"],
                f"{row['timestamp_sec']:.2f}",
                f"{row['confidence']:.2f}",
                row["review_status"],
            ]
        )
    if len(table_data) == 1:
        story.append(Paragraph("No potential violations were stored for this selection.", styles["BodyText"]))
    else:
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        story.append(table)
    doc.build(story)
    return buffer.getvalue()
