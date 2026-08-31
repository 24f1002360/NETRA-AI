import os

from flask import (
    Blueprint, redirect, url_for, render_template,
    request, session, jsonify, send_file, abort
)
from app.stubs import make_stub_result, make_stub_retake_result, make_stub_history
from app.report import render_result

bp = Blueprint("main", __name__)

# In-memory result store (replaced by Divyanshu's DB on integration)
_results: dict = {}


# ── Data adapters ──────────────────────────────────────────────────────────────

def _for_result_view(screening: dict) -> dict:
    """Adapt the shared ScreeningResult contract dict for the result template."""
    grading  = screening.get("grading", {}) or {}
    routing  = screening.get("routing", {}) or {}
    quality  = dict(screening.get("quality", {}) or {})
    scores   = dict(quality.get("scores", {}) or {})
    scores.setdefault("artefact", 0.0)
    quality["scores"] = scores

    action = (routing.get("action") or "ROUTINE").upper()
    if action == "URGENT_REFERRAL":
        action = "URGENT"

    return {
        **screening,
        "timestamp":    screening.get("captured_at", ""),
        "grade":        grading.get("icdr_grade", 0) or 0,
        "confidence":   grading.get("confidence", 0.0) or 0.0,
        "lesions":      (screening.get("lesions", {}) or {}).get("counts", {}),
        "quality":      quality,
        "xai_agreement": (screening.get("xai", {}) or {}).get("guard_status") == "OK",
        "action_type":  action,
        "action_reason": routing.get("reason", ""),
    }


def _for_history_view(screenings: list) -> list:
    rows = []
    for s in screenings:
        v = _for_result_view(s)
        rows.append({
            "screening_id": v["screening_id"],
            "date":         v["timestamp"],
            "eye":          v["eye"],
            "grade":        v["grade"],
            "confidence":   v["confidence"],
            "action":       v["action_type"],
        })
    return rows


def _trend_from_history(screenings: list) -> str:
    """Compute a trend string from a list of ScreeningResult dicts."""
    if not screenings:
        return "First Visit"
    # Check longitudinal block of the latest screening
    latest = screenings[-1] if screenings else {}
    trend = (latest.get("longitudinal") or {}).get("trend", "FIRST_VISIT")
    mapping = {
        "WORSENING":   "Worsening",
        "IMPROVING":   "Improving",
        "STABLE":      "Stable",
        "FIRST_VISIT": "First Visit",
    }
    return mapping.get(trend.upper(), "First Visit")


# ── Routes ─────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    return redirect(url_for("main.capture"))


@bp.route("/capture")
def capture():
    return render_template("capture.html")


@bp.route("/capture/upload", methods=["POST"])
def capture_upload():
    patient_id = (request.form.get("patient_id") or "").strip() or "UNKNOWN"

    import random

    # Generate result for Right Eye (OD)
    if random.random() < 0.2:
        res_od = make_stub_retake_result(patient_id, "OD")
    else:
        res_od = make_stub_result(patient_id, "OD")

    # Generate result for Left Eye (OS)
    if random.random() < 0.2:
        res_os = make_stub_retake_result(patient_id, "OS")
    else:
        res_os = make_stub_result(patient_id, "OS")

    # Store both individually
    _results[res_od["screening_id"]] = res_od
    _results[res_os["screening_id"]] = res_os

    # Create a combined screening session
    import uuid
    session_id = str(uuid.uuid4())
    _results["session_" + session_id] = {
        "session_id": session_id,
        "patient_id": patient_id,
        "od_screening_id": res_od["screening_id"],
        "os_screening_id": res_os["screening_id"],
    }

    # If either eye needs retake, go to quality for the bad one first
    od_retake = res_od["quality"]["verdict"] == "RETAKE"
    os_retake = res_os["quality"]["verdict"] == "RETAKE"

    if od_retake or os_retake:
        # Go to combined quality screen
        return redirect(url_for("main.quality_combined", session_id=session_id))

    # Both passed → go directly to combined result
    return redirect(url_for("main.result_combined", session_id=session_id))


@bp.route("/quality/<screening_id>")
def quality(screening_id):
    res = _results.get(screening_id)
    if not res:
        return render_template("_error.html",
                               title="Screening not found",
                               detail="This screening result has expired or does not exist."), 404
    return render_template("quality.html", result=res)


@bp.route("/result/<screening_id>")
def result(screening_id):
    res = _results.get(screening_id)
    if not res:
        return render_template("_error.html",
                               title="Result not found",
                               detail="This screening result has expired or does not exist."), 404
    return render_template("result.html", result=_for_result_view(res))


# ── Combined (both-eye) routes ─────────────────────────────────────────────────

@bp.route("/quality/session/<session_id>")
def quality_combined(session_id):
    sess = _results.get("session_" + session_id)
    if not sess:
        return render_template("_error.html",
                               title="Session not found",
                               detail="This screening session has expired."), 404
    res_od = _results.get(sess["od_screening_id"], {})
    res_os = _results.get(sess["os_screening_id"], {})
    return render_template("quality.html",
                           result=res_od,
                           result_os=res_os,
                           session_id=session_id,
                           combined=True)


@bp.route("/result/session/<session_id>")
def result_combined(session_id):
    sess = _results.get("session_" + session_id)
    if not sess:
        return render_template("_error.html",
                               title="Session not found",
                               detail="This screening session has expired."), 404
    res_od = _results.get(sess["od_screening_id"], {})
    res_os = _results.get(sess["os_screening_id"], {})
    return render_template("result_combined.html",
                           od=_for_result_view(res_od),
                           os=_for_result_view(res_os),
                           patient_id=sess["patient_id"],
                           session_id=session_id)


@bp.route("/report/session/<session_id>/view")
def report_combined_view(session_id):
    sess = _results.get("session_" + session_id)
    if not sess:
        abort(404)
    res_od = _results.get(sess["od_screening_id"], {})
    res_os = _results.get(sess["os_screening_id"], {})
    from app.report import render_combined_result
    html_content = render_combined_result(res_od, res_os)
    return html_content


@bp.route("/report/<screening_id>/view")
def report_view(screening_id):
    res = _results.get(screening_id)
    if not res:
        abort(404)
    html_content = render_result(res)
    return html_content


@bp.route("/history/")
@bp.route("/history/<patient_id>")
def history(patient_id=""):
    if not patient_id:
        patient_id = request.args.get("patient_id", "")

    if patient_id:
        history_data = make_stub_history(patient_id)
    else:
        history_data = []

    return render_template(
        "history.html",
        screenings=_for_history_view(history_data),
        trend=_trend_from_history(history_data),
        patient_id=patient_id,
    )


@bp.route("/artifacts/<path:path>")
def serve_artifact(path):
    """Serve XAI/segmentation images stored in the artifacts/ directory."""
    root = os.path.abspath("artifacts")
    target = os.path.abspath(os.path.join(root, path))
    if not target.startswith(root):
        abort(403)
    if not os.path.exists(target):
        abort(404)
    return send_file(target)


@bp.route("/api/language", methods=["POST"])
def set_language():
    data = request.get_json(silent=True) or {}
    lang = data.get("lang")
    if lang:
        session["lang"] = lang
        return jsonify({"status": "ok"})
    return jsonify({"error": "invalid"}), 400


@bp.route("/api/screening/<screening_id>")
def get_screening(screening_id):
    res = _results.get(screening_id)
    if not res:
        return jsonify({"error": "not found"}), 404
    return jsonify(res)
