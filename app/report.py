"""
app/report.py — NETRA-AI clinical report generator.

Owned by Abhishek (Part A).
Contract:
    render_result(result: dict) -> str   # HTML string

Returns a well-formatted HTML document that can be viewed in the browser
and printed to PDF natively by the user (Ctrl+P). This avoids the need for
heavy GTK3 dependencies on Windows laptops.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

# ── Grade/action label maps ────────────────────────────────────────────────────

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

ACTION_LABELS = {
    "ROUTINE":         "No action needed — rescreen in 12 months",
    "REVIEW":          "Refer to specialist for review",
    "URGENT":          "URGENT: Immediate ophthalmologist referral",
    "URGENT_REFERRAL": "URGENT: Immediate ophthalmologist referral",
}

TREND_LABELS = {
    "WORSENING":   "Worsening ↑",
    "IMPROVING":   "Improving ↓",
    "STABLE":      "Stable →",
    "FIRST_VISIT": "First visit",
}


# ── HTML template ──────────────────────────────────────────────────────────────

_REPORT_CSS = """
  @page { size: A4; margin: 18mm 16mm; }
  body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt;
         color: #111; background: #fff; line-height: 1.5; }
  .header { display: flex; justify-content: space-between;
            align-items: flex-end; border-bottom: 2px solid #1f3a5f;
            padding-bottom: 10px; margin-bottom: 18px; }
  .header h1 { font-size: 18pt; color: #1f3a5f; margin: 0; }
  .header .meta { font-size: 9pt; color: #555; text-align: right; }
  .section { margin-bottom: 16px; }
  .section-title { font-size: 9pt; font-weight: bold; text-transform: uppercase;
                   letter-spacing: 0.08em; color: #555;
                   border-bottom: 1px solid #ddd; padding-bottom: 4px;
                   margin-bottom: 8px; }
  .kv-row { display: flex; gap: 8px; margin-bottom: 4px; font-size: 10.5pt; }
  .kv-label { font-weight: bold; min-width: 140px; color: #333; }
  .decision { padding: 12px 16px; border-radius: 6px; margin-bottom: 18px;
              font-size: 12pt; font-weight: bold; }
  .decision.routine { background: #e6f4ea; color: #2d5a27; border-left: 5px solid #2da44e; }
  .decision.review  { background: #fef9e7; color: #7a5c00; border-left: 5px solid #e3b341; }
  .decision.urgent  { background: #fce8e6; color: #7f2020; border-left: 5px solid #f85149; }
  .grade-block { display: inline-block; padding: 8px 18px; border-radius: 8px;
                 font-size: 22pt; font-weight: bold; margin-right: 14px;
                 vertical-align: middle; }
  .grade-0 { background: #e6f4ea; color: #2da44e; }
  .grade-1 { background: #f0fce0; color: #57ab5a; }
  .grade-2 { background: #fef9e7; color: #b08800; }
  .grade-3 { background: #fce8e6; color: #c0392b; }
  .grade-4 { background: #fce8e6; color: #8b0000; }
  table { width: 100%; border-collapse: collapse; font-size: 10pt; }
  th { background: #f0f4f8; color: #333; font-size: 9pt; text-transform: uppercase;
       letter-spacing: 0.06em; padding: 7px 10px; text-align: left; }
  td { padding: 7px 10px; border-bottom: 1px solid #eee; }
  .footer { margin-top: 20px; padding-top: 10px; border-top: 1px solid #ccc;
            font-size: 8pt; color: #888; text-align: center; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 9pt; font-weight: bold; }
  .badge-green { background: #e6f4ea; color: #2da44e; }
  .badge-amber { background: #fef9e7; color: #b08800; }
  .badge-red   { background: #fce8e6; color: #c0392b; }
"""


def render_result(result: dict) -> str:
    """Render a ScreeningResult dict to a self-contained HTML string."""
    t0 = time.perf_counter_ns()

    grading   = result.get("grading", {}) or {}
    routing   = result.get("routing", {}) or {}
    lesions   = (result.get("lesions", {}) or {}).get("counts", {}) or {}
    xai       = result.get("xai", {}) or {}
    longit    = result.get("longitudinal", {}) or {}
    quality   = result.get("quality", {}) or {}

    grade       = grading.get("icdr_grade", 0) or 0
    grade_label = GRADE_LABELS.get(grade, f"Grade {grade}")
    confidence  = grading.get("confidence", 0.0) or 0.0
    action_raw  = (routing.get("action") or "ROUTINE").upper()
    action_label = ACTION_LABELS.get(action_raw, action_raw)
    action_cls  = "urgent" if "URGENT" in action_raw else action_raw.lower()

    trend_raw   = (longit.get("trend") or "FIRST_VISIT").upper()
    trend_label = TREND_LABELS.get(trend_raw, trend_raw)

    guard       = xai.get("guard_status", "—")
    agreement   = xai.get("cam_lesion_agreement")
    agreement_s = f"{agreement*100:.0f}%" if agreement is not None else "—"

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    captured = result.get("captured_at", "")[:10] or "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NETRA-AI Report — {result.get('screening_id', '')[:8]}</title>
  <style>{_REPORT_CSS}</style>
</head>
<body>

<div class="header">
  <div>
    <h1>NETRA-AI Screening Report</h1>
    <div style="font-size:9pt;color:#555;margin-top:4px">
      Diabetic Retinopathy Screening System
    </div>
  </div>
  <div class="meta">
    Screening ID: {result.get('screening_id', '—')[:12]}<br>
    Generated: {now_str}
  </div>
</div>

<!-- Decision -->
<div class="decision {action_cls}">
  {action_label}
</div>

<!-- Patient info -->
<div class="section">
  <div class="section-title">Patient Information</div>
  <div class="kv-row"><span class="kv-label">Patient ID:</span> {result.get('patient_id', '—')}</div>
  <div class="kv-row"><span class="kv-label">Eye Screened:</span> {result.get('eye', '—')}</div>
  <div class="kv-row"><span class="kv-label">Date of Capture:</span> {captured}</div>
  <div class="kv-row"><span class="kv-label">PHC:</span> {result.get('phc_id', '—')}</div>
  <div class="kv-row"><span class="kv-label">Operator ID:</span> {result.get('operator_id', '—')}</div>
</div>

<!-- Grading -->
<div class="section">
  <div class="section-title">DR Grading</div>
  <p style="margin:8px 0">
    <span class="grade-block grade-{grade}">{grade}</span>
    <strong style="font-size:14pt">{grade_label}</strong>
  </p>
  <div class="kv-row">
    <span class="kv-label">Confidence:</span>
    {confidence*100:.1f}%
    {'<span class="badge badge-amber">Low — review recommended</span>' if confidence < 0.55 else ''}
  </div>
  <div class="kv-row">
    <span class="kv-label">Referable DR:</span>
    {'<span class="badge badge-red">Yes</span>' if grading.get('referable_dr') else '<span class="badge badge-green">No</span>'}
  </div>
  <div class="kv-row"><span class="kv-label">Model:</span>
    {grading.get('model_id', '—')} ({grading.get('model_version', '—')})
  </div>
</div>

<!-- Lesions -->
<div class="section">
  <div class="section-title">Lesions Detected</div>
  <table>
    <thead>
      <tr><th>Lesion Type</th><th>Count</th></tr>
    </thead>
    <tbody>
      <tr><td>Microaneurysms</td><td>{lesions.get('microaneurysms', 0)}</td></tr>
      <tr><td>Haemorrhages</td><td>{lesions.get('haemorrhages', 0)}</td></tr>
      <tr><td>Hard Exudates</td><td>{lesions.get('hard_exudates', 0)}</td></tr>
      <tr><td>Soft Exudates</td><td>{lesions.get('soft_exudates', 0)}</td></tr>
    </tbody>
  </table>
</div>

<!-- XAI + Longitudinal -->
<div class="section">
  <div class="section-title">Evidence & Longitudinal Trend</div>
  <div class="kv-row"><span class="kv-label">XAI Guard Status:</span> {guard}</div>
  <div class="kv-row"><span class="kv-label">CAM-Lesion Agreement:</span> {agreement_s}</div>
  <div class="kv-row"><span class="kv-label">Trend vs Prior:</span> {trend_label}</div>
  <div class="kv-row"><span class="kv-label">Prior Grade:</span>
    {longit.get('prior_grade', '—') if longit.get('prior_grade') is not None else 'No prior screening'}
  </div>
</div>

<!-- Image quality -->
<div class="section">
  <div class="section-title">Image Quality</div>
  <div class="kv-row"><span class="kv-label">Verdict:</span> {quality.get('verdict', '—')}</div>
  <div class="kv-row"><span class="kv-label">Enhancement:</span>
    {', '.join(quality.get('enhancement_applied', [])) or 'None'}
  </div>
</div>

<div class="footer">
  NETRA-AI v1.0 — AI-assisted screening only. Final diagnosis must be made by a qualified ophthalmologist.<br>
  Model: EfficientNet-B0 + U-Net-Lite + Grad-CAM guard. Offline system — no patient data transmitted.
</div>

</body>
</html>"""

    result.setdefault("timings_ms", {})["report"] = (time.perf_counter_ns() - t0) / 1e6
    return html


def _eye_section(result: dict, eye_label: str) -> str:
    """Render one eye's section for the combined report."""
    grading   = result.get("grading", {}) or {}
    routing   = result.get("routing", {}) or {}
    lesions   = (result.get("lesions", {}) or {}).get("counts", {}) or {}
    xai       = result.get("xai", {}) or {}

    grade       = grading.get("icdr_grade", 0) or 0
    grade_label = GRADE_LABELS.get(grade, f"Grade {grade}")
    confidence  = grading.get("confidence", 0.0) or 0.0
    action_raw  = (routing.get("action") or "ROUTINE").upper()
    action_label = ACTION_LABELS.get(action_raw, action_raw)
    action_cls  = "urgent" if "URGENT" in action_raw else action_raw.lower()

    guard       = xai.get("guard_status", "—")
    agreement   = xai.get("cam_lesion_agreement")
    agreement_s = f"{agreement*100:.0f}%" if agreement is not None else "—"

    return f"""
<div style="flex:1;min-width:260px">
  <h3 style="font-size:12pt;margin:0 0 8px;color:#1f3a5f">{eye_label} ({result.get('eye', '—')})</h3>
  <div class="decision {action_cls}" style="font-size:10pt;padding:8px 12px;margin-bottom:10px">
    {action_label}
  </div>
  <div style="margin-bottom:8px">
    <span class="grade-block grade-{grade}" style="font-size:16pt;padding:5px 12px">{grade}</span>
    <strong>{grade_label}</strong> — Confidence: {confidence*100:.0f}%
    {'<span class="badge badge-red">Referable</span>' if grading.get('referable_dr') else ''}
  </div>
  <table style="width:100%;margin-bottom:8px">
    <tr><td style="padding:3px 6px;font-size:9pt">Microaneurysms</td><td style="padding:3px 6px;font-size:9pt;font-weight:bold">{lesions.get('microaneurysms', 0)}</td></tr>
    <tr><td style="padding:3px 6px;font-size:9pt">Haemorrhages</td><td style="padding:3px 6px;font-size:9pt;font-weight:bold">{lesions.get('haemorrhages', 0)}</td></tr>
    <tr><td style="padding:3px 6px;font-size:9pt">Hard Exudates</td><td style="padding:3px 6px;font-size:9pt;font-weight:bold">{lesions.get('hard_exudates', 0)}</td></tr>
    <tr><td style="padding:3px 6px;font-size:9pt">Soft Exudates</td><td style="padding:3px 6px;font-size:9pt;font-weight:bold">{lesions.get('soft_exudates', 0)}</td></tr>
  </table>
  <div style="font-size:8.5pt;color:#555">
    XAI Guard: {guard} · CAM Agreement: {agreement_s}
  </div>
</div>"""


def render_combined_result(result_od: dict, result_os: dict) -> str:
    """Render a combined both-eyes report as a single HTML document."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    patient_id = result_od.get("patient_id", "—")
    captured = result_od.get("captured_at", "")[:10] or "—"

    # Determine worst-case action
    od_action = (result_od.get("routing", {}) or {}).get("action", "ROUTINE").upper()
    os_action = (result_os.get("routing", {}) or {}).get("action", "ROUTINE").upper()
    if "URGENT" in od_action or "URGENT" in os_action:
        worst_action = "URGENT"
    elif od_action == "REVIEW" or os_action == "REVIEW":
        worst_action = "REVIEW"
    else:
        worst_action = "ROUTINE"

    worst_label = ACTION_LABELS.get(worst_action, worst_action)
    worst_cls = "urgent" if "URGENT" in worst_action else worst_action.lower()

    od_section = _eye_section(result_od, "Right Eye")
    os_section = _eye_section(result_os, "Left Eye")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NETRA-AI Combined Report — {patient_id}</title>
  <style>{_REPORT_CSS}</style>
</head>
<body>

<div class="header">
  <div>
    <h1>NETRA-AI Screening Report</h1>
    <div style="font-size:9pt;color:#555;margin-top:4px">
      Combined Both-Eye Assessment
    </div>
  </div>
  <div class="meta">
    Patient: {patient_id}<br>
    Date: {captured}<br>
    Generated: {now_str}
  </div>
</div>

<!-- Overall decision -->
<div class="decision {worst_cls}">
  {worst_label}
</div>

<!-- Patient info -->
<div class="section">
  <div class="section-title">Patient Information</div>
  <div class="kv-row"><span class="kv-label">Patient ID:</span> {patient_id}</div>
  <div class="kv-row"><span class="kv-label">Date of Screening:</span> {captured}</div>
  <div class="kv-row"><span class="kv-label">PHC:</span> {result_od.get('phc_id', '—')}</div>
  <div class="kv-row"><span class="kv-label">Operator ID:</span> {result_od.get('operator_id', '—')}</div>
</div>

<!-- Both eyes side-by-side -->
<div class="section">
  <div class="section-title">Eye-by-Eye Results</div>
  <div style="display:flex;gap:20px;flex-wrap:wrap">
    {od_section}
    {os_section}
  </div>
</div>

<div class="footer">
  NETRA-AI v1.0 — AI-assisted screening only. Final diagnosis must be made by a qualified ophthalmologist.<br>
  Model: EfficientNet-B0 + U-Net-Lite + Grad-CAM guard. Offline system — no patient data transmitted.
</div>

</body>
</html>"""
