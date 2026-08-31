import os

from flask import Blueprint, redirect, url_for, render_template, request, session, jsonify, send_file
from app.stubs import make_stub_result, make_stub_retake_result, make_stub_history
from app.report import generate_pdf

bp = Blueprint("main", __name__)
_results = {}


def _for_result_view(screening):
    """Adapt the shared ScreeningResult contract for the initial templates."""
    grading = screening.get("grading", {})
    routing = screening.get("routing", {})
    quality = dict(screening.get("quality", {}))
    scores = dict(quality.get("scores", {}))
    scores.setdefault("artefact", 0.0)
    quality["scores"] = scores
    action = routing.get("action", "ROUTINE").upper()
    if action == "URGENT_REFERRAL":
        action = "URGENT"
    return {
        **screening,
        "timestamp": screening.get("captured_at", ""),
        "grade": grading.get("icdr_grade", 0),
        "confidence": grading.get("confidence", 0.0),
        "lesions": screening.get("lesions", {}).get("counts", {}),
        "quality": quality,
        "xai_agreement": screening.get("xai", {}).get("guard_status") == "OK",
        "action_type": action,
        "action_reason": routing.get("reason", ""),
    }


def _for_history_view(screenings):
    rows = []
    for screening in screenings:
        view = _for_result_view(screening)
        rows.append({
            "screening_id": view["screening_id"],
            "date": view["timestamp"],
            "eye": view["eye"],
            "grade": view["grade"],
            "confidence": view["confidence"],
            "action": view["action_type"],
        })
    return rows

@bp.route("/")
def index():
    return redirect(url_for("main.capture"))

@bp.route("/capture")
def capture():
    return render_template("capture.html")

@bp.route("/capture/upload", methods=["POST"])
def capture_upload():
    patient_id = request.form.get("patient_id", "UNKNOWN")
    eye = request.form.get("eye", "OD")
    res = make_stub_result(patient_id, eye)
    _results[res["screening_id"]] = res
    return redirect(url_for("main.quality", screening_id=res["screening_id"]))

@bp.route("/quality/<screening_id>")
def quality(screening_id):
    res = _results.get(screening_id)
    if not res:
        return "Not found", 404
    return render_template("quality.html", result=res)

@bp.route("/result/<screening_id>")
def result(screening_id):
    res = _results.get(screening_id)
    if not res:
        return "Not found", 404
    return render_template("result.html", result=_for_result_view(res))

@bp.route("/history/<patient_id>")
def history(patient_id):
    history_data = make_stub_history(patient_id)
    return render_template(
        "history.html",
        screenings=_for_history_view(history_data),
        trend="First Visit",
        patient_id=patient_id,
    )

@bp.route("/report/<screening_id>/pdf")
def report_pdf(screening_id):
    res = _results.get(screening_id)
    if not res:
        return "Not found", 404
    pdf_path = generate_pdf(res)
    return send_file(os.path.abspath(pdf_path), as_attachment=True)

@bp.route("/api/language", methods=["POST"])
def set_language():
    data = request.get_json()
    language = data.get("lang") if data else None
    if language:
        session["lang"] = language
        return jsonify({"status": "ok"})
    return jsonify({"error": "invalid"}), 400

@bp.route("/api/screening/<screening_id>")
def get_screening(screening_id):
    res = _results.get(screening_id)
    if not res:
        return jsonify({"error": "not found"}), 404
    return jsonify(res)
