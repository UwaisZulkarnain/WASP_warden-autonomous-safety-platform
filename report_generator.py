"""
WASP Report Generator
Generates a PDF safety report with all violation screenshots on demand.
Auto-screenshots every new violation event via save_violation_screenshot().
Usage: imported by wasp_backend.py
"""

import os
import glob
import cv2
import json
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether, PageBreak
)

# ── Brand colours ──────────────────────────────────────────────────────────────
RED      = colors.HexColor("#dc2626")
DARK     = colors.HexColor("#0b1120")
DARKGRAY = colors.HexColor("#1e293b")
MID      = colors.HexColor("#334155")
LIGHT    = colors.HexColor("#e2e8f0")
WHITE    = colors.white
AMBER    = colors.HexColor("#f59e0b")
GREEN    = colors.HexColor("#22c55e")
ORANGE   = colors.HexColor("#f97316")

RISK_COLORS = {
    "LOW":      GREEN,
    "MEDIUM":   AMBER,
    "HIGH":     ORANGE,
    "CRITICAL": RED,
}


def _styles():
    def S(name, **kw):
        return ParagraphStyle(name, **kw)
    return {
        "cover_title": S("cover_title",
            fontName="Helvetica-Bold", fontSize=32, textColor=WHITE,
            alignment=TA_CENTER, spaceAfter=6),
        "cover_sub": S("cover_sub",
            fontName="Helvetica", fontSize=13, textColor=colors.HexColor("#fca5a5"),
            alignment=TA_CENTER, spaceAfter=4),
        "cover_meta": S("cover_meta",
            fontName="Helvetica", fontSize=10, textColor=LIGHT,
            alignment=TA_CENTER, spaceAfter=2),
        "section": S("section",
            fontName="Helvetica-Bold", fontSize=13, textColor=RED,
            spaceBefore=14, spaceAfter=6),
        "body": S("body",
            fontName="Helvetica", fontSize=9, textColor=DARK,
            leading=13, spaceAfter=4),
        "caption": S("caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER, spaceAfter=6),
        "table_header": S("table_header",
            fontName="Helvetica-Bold", fontSize=8, textColor=WHITE),
        "table_cell": S("table_cell",
            fontName="Helvetica", fontSize=8, textColor=DARK, leading=11),
    }


# ── Screenshot helpers ─────────────────────────────────────────────────────────

def save_violation_screenshot(frame, cv_state: dict, reports_dir: str) -> str | None:
    """
    Burn timestamp + per-person violation labels onto the frame and save as JPEG.
    Called automatically by cv_thread() on each new violation event.
    Returns the saved file path, or None if frame is unavailable.
    """
    if frame is None:
        return None

    img = frame.copy()
    h, w = img.shape[:2]
    persons = cv_state.get("persons", [])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top banner
    cv2.rectangle(img, (0, 0), (w, 28), (30, 15, 30), -1)
    cv2.putText(img, f"WASP VIOLATION CAPTURE  {ts}", (8, 18),
                font, 0.52, (252, 165, 165), 1, cv2.LINE_AA)

    # Per-person violation labels
    y_off = 50
    for p in persons:
        violations = p.get("violations", [])
        if violations:
            label = f"{p['person_id'].upper()}: {', '.join(violations)}"
            cv2.rectangle(img, (6, y_off - 14),
                          (6 + len(label) * 7, y_off + 4), (180, 30, 30), -1)
            cv2.putText(img, label, (8, y_off), font, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y_off += 20

    os.makedirs(reports_dir, exist_ok=True)
    fname = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(reports_dir, fname)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[SCREENSHOT] Saved: {path}")
    return path


# ── PDF sections ───────────────────────────────────────────────────────────────

def _cover_page(story, styles, generated_at: str, site: str, shot_count: int):
    cover_data = [[Paragraph("WASP", styles["cover_title"])]]
    cover_tbl = Table(cover_data, colWidths=[16 * cm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 4))

    sub_data = [[Paragraph("Warden Autonomous Safety Platform", styles["cover_sub"])]]
    sub_tbl = Table(sub_data, colWidths=[16 * cm])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#991b1b")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 18))

    story.append(Paragraph("<b>ON-DEMAND SAFETY REPORT</b>", styles["cover_meta"]))
    story.append(Paragraph(f"Generated: {generated_at}", styles["cover_meta"]))
    story.append(Paragraph(f"Location: {site}", styles["cover_meta"]))
    story.append(Paragraph(f"Violation screenshots captured this session: {shot_count}", styles["cover_meta"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=16))


def _summary_table(story, styles, sensor_data: dict, cv_state: dict, ml_result: dict | None):
    story.append(Paragraph("1. Site Snapshot", styles["section"]))

    temp     = sensor_data.get("temperature", 0.0)
    humid    = sensor_data.get("humidity", 0.0)
    gas      = sensor_data.get("air_quality", 0)
    motion   = "Detected" if sensor_data.get("motion") else "Clear"
    persons  = cv_state.get("person_count", 0)
    violations = sum(1 for p in cv_state.get("persons", []) if p.get("status") == "VIOLATION")
    ml_label = (ml_result or {}).get("label", "N/A")
    ml_score = (ml_result or {}).get("combined_score", 0.0)

    rows = [
        [Paragraph("<b>Parameter</b>", styles["table_header"]),
         Paragraph("<b>Value</b>",     styles["table_header"]),
         Paragraph("<b>Status</b>",    styles["table_header"])],
        ["Temperature",      f"{temp:.1f} C",    "HIGH" if temp > 38 else "OK"],
        ["Humidity",         f"{humid:.0f} %",   "HIGH" if humid > 80 else "OK"],
        ["Gas / Air Quality", str(gas),           "HIGH" if gas > 600 else ("ELEVATED" if gas > 450 else "OK")],
        ["Motion",           motion,             "-"],
        ["Workers Detected", str(persons),       "-"],
        ["PPE Violations",   str(violations),    "YES" if violations else "NONE"],
        ["ML Anomaly",       f"{ml_label} ({ml_score:.3f})", "ANOMALY" if ml_label == "ANOMALY" else "Normal"],
    ]

    col_w = [6 * cm, 5 * cm, 5 * cm]
    tbl = Table(rows, colWidths=col_w, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8fafc"), WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    for i, row in enumerate(rows[1:], start=1):
        status = row[2]
        if status in ("HIGH", "ELEVATED", "YES", "ANOMALY"):
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (2, i), (2, i), colors.HexColor("#fef2f2")),
                ("TEXTCOLOR",  (2, i), (2, i), RED),
                ("FONTNAME",   (2, i), (2, i), "Helvetica-Bold"),
            ]))
        elif status in ("OK", "NONE", "Normal"):
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (2, i), (2, i), colors.HexColor("#f0fdf4")),
                ("TEXTCOLOR",  (2, i), (2, i), GREEN),
                ("FONTNAME",   (2, i), (2, i), "Helvetica-Bold"),
            ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _ppe_table(story, styles, cv_state: dict):
    story.append(Paragraph("2. Per-Person PPE Status", styles["section"]))
    persons = cv_state.get("persons", [])
    if not persons:
        story.append(Paragraph("No persons detected in frame at time of report.", styles["body"]))
        story.append(Spacer(1, 8))
        return

    header = [Paragraph(h, styles["table_header"]) for h in
              ["Person", "Status", "Helmet", "Vest", "Goggles", "Gloves", "Boots", "Violations"]]
    rows = [header]
    for p in persons:
        ppe = p.get("ppe", {})
        def tick(key): return "YES" if ppe.get(key) else "NO"
        violations = "; ".join(p.get("violations", [])) or "None"
        rows.append([
            p.get("person_id", "-"),
            p.get("status", "-"),
            tick("helmet"), tick("vest"), tick("goggles"), tick("gloves"), tick("boots"),
            violations,
        ])

    col_w = [2.2*cm, 1.8*cm, 1.4*cm, 1.4*cm, 1.5*cm, 1.5*cm, 1.4*cm, 5.8*cm]
    tbl = Table(rows, colWidths=col_w, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8fafc"), WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (2, 1), (6, -1), "CENTER"),
    ]))
    for i, p in enumerate(persons, start=1):
        ppe = p.get("ppe", {})
        for j, key in enumerate(["helmet", "vest", "goggles", "gloves", "boots"], start=2):
            color = GREEN if ppe.get(key) else RED
            tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (j, i), (j, i), color),
                ("FONTNAME",  (j, i), (j, i), "Helvetica-Bold"),
            ]))
        if p.get("status") == "VIOLATION":
            tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (1, i), (1, i), RED),
                ("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"),
            ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _screenshot_section(story, styles, reports_dir: str = "reports"):
    """
    Embed ALL violation_*.jpg screenshots found in reports_dir, newest first.
    Up to 20 images to keep PDF size reasonable.
    """
    story.append(Paragraph("3. Violation Screenshots", styles["section"]))

    shots = sorted(
        glob.glob(os.path.join(reports_dir, "violation_*.jpg")),
        reverse=True
    )[:20]

    if not shots:
        story.append(Paragraph(
            "No violation screenshots captured this session. "
            "Screenshots are saved automatically whenever a PPE violation is detected.",
            styles["body"]
        ))
        story.append(Spacer(1, 8))
        return

    story.append(Paragraph(
        f"Showing {len(shots)} most recent violation captures (newest first).",
        styles["body"]
    ))
    story.append(Spacer(1, 6))

    for path in shots:
        try:
            img = RLImage(path, width=15 * cm, height=9 * cm, kind="proportional")
            raw_ts = os.path.basename(path).replace("violation_", "").replace(".jpg", "")
            # Format: 20250101_143000 -> 2025-01-01 14:30:00
            try:
                dt = datetime.strptime(raw_ts, "%Y%m%d_%H%M%S")
                pretty_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pretty_ts = raw_ts

            story.append(KeepTogether([
                img,
                Paragraph(
                    f"Captured: {pretty_ts}  |  File: {os.path.basename(path)}",
                    styles["caption"]
                ),
            ]))
            story.append(Spacer(1, 8))
        except Exception as e:
            story.append(Paragraph(f"Could not load {os.path.basename(path)}: {e}", styles["body"]))


def _recent_decisions(story, styles, db_path: str, limit: int = 10):
    story.append(Paragraph("4. Recent Agent Decisions", styles["section"]))
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        c = conn.cursor()
        c.execute("""SELECT timestamp, risk_level, decision_json, model_used
                     FROM agent_decisions ORDER BY id DESC LIMIT ?""", (limit,))
        rows = c.fetchall()
        conn.close()
    except Exception:
        rows = []

    if not rows:
        story.append(Paragraph("No agent decisions recorded yet.", styles["body"]))
        return

    header = [Paragraph(h, styles["table_header"]) for h in
              ["Time", "Risk", "Model", "Reasoning"]]
    tbl_rows = [header]
    for ts, risk, dj_raw, model in rows:
        dj = json.loads(dj_raw) if dj_raw else {}
        reasoning = dj.get("reasoning", "-")[:90]
        if len(dj.get("reasoning", "")) > 90:
            reasoning += "..."
        tbl_rows.append([
            ts.split(" ")[1] if ts and " " in ts else (ts or "-"),
            risk or "-",
            (model or "-")[:12],
            reasoning,
        ])

    col_w = [1.8*cm, 1.6*cm, 2.0*cm, 10.6*cm]
    tbl = Table(tbl_rows, colWidths=col_w, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("LEADING",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8fafc"), WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    risk_bg = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": AMBER, "LOW": GREEN}
    for i, (_, risk, *_rest) in enumerate(rows, start=1):
        bg = risk_bg.get(risk, MID)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (1, i), (1, i), bg),
            ("TEXTCOLOR",  (1, i), (1, i), WHITE),
            ("FONTNAME",   (1, i), (1, i), "Helvetica-Bold"),
        ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _recent_alerts(story, styles, db_path: str, limit: int = 15):
    story.append(Paragraph("5. Recent Alerts Log", styles["section"]))
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT timestamp, alert_type, details FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
    except Exception:
        rows = []

    if not rows:
        story.append(Paragraph("No alerts recorded yet.", styles["body"]))
        return

    header = [Paragraph(h, styles["table_header"]) for h in ["Time", "Type", "Details"]]
    tbl_rows = [header]
    for ts, atype, details in rows:
        tbl_rows.append([
            ts.split(" ")[1] if ts and " " in ts else (ts or "-"),
            atype or "-",
            (details or "-")[:120],
        ])

    col_w = [2.2*cm, 3.5*cm, 10.3*cm]
    tbl = Table(tbl_rows, colWidths=col_w, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("LEADING",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8fafc"), WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _footer_note(story, styles, generated_at: str):
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID, spaceBefore=10, spaceAfter=6))
    story.append(Paragraph(
        f"WASP - Warden Autonomous Safety Platform  |  UTM FAI Showcase 2026  |  Generated {generated_at}",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=7,
                       textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER)
    ))


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_report(
    output_path: str,
    sensor_data: dict,
    cv_state: dict,
    db_path: str,
    reports_dir: str = "reports",
    ml_result: dict | None = None,
    site: str = "Site A - Zone A",
) -> str:
    """
    Generate a PDF safety report embedding ALL violation screenshots found
    in reports_dir. Returns output_path on success.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(reports_dir, exist_ok=True)

    shot_count = len(glob.glob(os.path.join(reports_dir, "violation_*.jpg")))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="WASP Safety Report",
        author="WASP Autonomous Safety Platform",
    )

    styles = _styles()
    story  = []

    _cover_page(story, styles, generated_at, site, shot_count)
    _summary_table(story, styles, sensor_data, cv_state, ml_result)
    _ppe_table(story, styles, cv_state)
    _screenshot_section(story, styles, reports_dir)
    _recent_decisions(story, styles, db_path)
    _recent_alerts(story, styles, db_path)
    _footer_note(story, styles, generated_at)

    doc.build(story)
    return output_path