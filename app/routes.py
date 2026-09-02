import uuid
from pathlib import Path

from flask import (
    Blueprint, redirect, url_for, render_template,
    request, session, jsonify, send_file, abort
)
from app.report import render_result
from core.inference import run_screening
from db.dao import get_history, get_screening as get_persisted_screening

bp = Blueprint("main", __name__)
screening_bp = Blueprint("screening", __name__)

# Short-lived cache for an active two-eye session. Individual records are
# persisted by ``run_screening`` and can be restored from SQLite on demand.
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

    image = screening.get("image", {}) or {}
    xai = screening.get("xai", {}) or {}

    # ``None`` is meaningful here: it is how the inference layer reports that
    # the grading model could not be loaded.  Never coerce it to grade 0,
    # because grade 0 means a real, healthy "No DR" model prediction.
    analysis_available = grading.get("icdr_grade") is not None

    return {
        **screening,
        "timestamp":    screening.get("captured_at", ""),
        "grade":        grading.get("icdr_grade") if analysis_available else None,
        "confidence":   grading.get("confidence", 0.0) or 0.0,
        "analysis_available": analysis_available,
        "lesions":      (screening.get("lesions", {}) or {}).get("counts", {}),
        "quality":      quality,
        "xai_agreement": (screening.get("xai", {}) or {}).get("guard_status") == "OK",
        "action_type":  action,
        "action_reason": routing.get("reason", ""),
        "media": {
            "raw": _media_path(image.get("raw_path")),
            "processed": _media_path(image.get("processed_path")),
            "lesion": _media_path((screening.get("lesions", {}) or {}).get("mask_path")),
            "overlay": _media_path(xai.get("overlay_path")),
            "gradcam": _media_path(xai.get("gradcam_path")),
        },
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
            "analysis_available": v["analysis_available"],
        })
    return rows


def _get_result(screening_id: str) -> dict | None:
    """Read a current result from cache first, then its durable offline record."""
    return _results.get(screening_id) or get_persisted_screening(screening_id)


def _media_path(value):
    """Return a path safely relative to locally served screening media."""
    if not value:
        return None
    root = (Path.cwd() / "data" / "captures").resolve()
    try:
        return str(Path(value).resolve().relative_to(root)).replace("\\", "/")
    except (OSError, ValueError):
        return None


def _trend_from_history(screenings: list) -> str:
    """Compute a trend string from a list of ScreeningResult dicts."""
    if not screenings:
        return "First Visit"
    # Check longitudinal block of the latest screening
    # DAO history is newest-first, so the first record is the current result.
    latest = screenings[0]
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
    uploads = {
        "OD": request.files.get("image_od") or request.files.get("image_od_cam"),
        "OS": request.files.get("image_os") or request.files.get("image_os_cam"),
    }

    if any(not upload or not upload.filename for upload in uploads.values()):
        return render_template(
            "_error.html",
            title="Both eye images are required",
            detail="Capture or upload one retinal image for each eye before starting analysis.",
        ), 400

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    invalid_files = [
        upload.filename
        for upload in uploads.values()
        if Path(upload.filename).suffix.lower() not in allowed_extensions
    ]
    if invalid_files:
        return render_template(
            "_error.html",
            title="Use JPG or PNG images",
            detail="Choose a retinal photo in JPG or PNG format for each eye.",
        ), 400

    capture_dir = Path("data") / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = {}
        for eye, upload in uploads.items():
            extension = Path(upload.filename).suffix.lower() or ".jpg"
            image_path = capture_dir / f"{uuid.uuid4().hex}{extension}"
            upload.save(image_path)
            results[eye] = run_screening(
                image_path=image_path,
                patient_id=patient_id,
                eye=eye,
                operator_id=request.form.get("operator_id", ""),
                phc_id=request.form.get("phc_id", ""),
            )
    except Exception:
        return render_template(
            "_error.html",
            title="Screening could not start",
            detail=(
                "The local AI models are unavailable or could not read this image. "
                "Check the model setup, then try again."
            ),
        ), 503

    for result in results.values():
        _results[result["screening_id"]] = result

    session_id = str(uuid.uuid4())
    _results[f"session_{session_id}"] = {
        "session_id": session_id,
        "patient_id": patient_id,
        "od_screening_id": results["OD"]["screening_id"],
        "os_screening_id": results["OS"]["screening_id"],
    }
    if any(result["quality"]["verdict"] == "RETAKE" for result in results.values()):
        return redirect(url_for("main.quality_combined", session_id=session_id))
    return redirect(url_for("main.result_combined", session_id=session_id))


@bp.route("/quality/<screening_id>")
def quality(screening_id):
    res = _get_result(screening_id)
    if not res:
        return render_template("_error.html",
                               title="Screening not found",
                               detail="This screening result has expired or does not exist."), 404
    return render_template("quality.html", result=res)


@bp.route("/result/<screening_id>")
def result(screening_id):
    res = _get_result(screening_id)
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
    res = _get_result(screening_id)
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
        history_data = get_history(patient_id)
    else:
        history_data = []

    return render_template(
        "history.html",
        screenings=_for_history_view(history_data),
        trend=_trend_from_history(history_data),
        patient_id=patient_id,
    )


@bp.route("/media/<path:path>")
def serve_capture_media(path):
    """Serve only generated local screening media, never arbitrary files."""
    root = (Path.cwd() / "data" / "captures").resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        abort(403)
    if not target.is_file():
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
    res = _get_result(screening_id)
    if not res:
        return jsonify({"error": "not found"}), 404
    return jsonify(res)


@screening_bp.post("/screenings")
def create_screening():
    """JSON API for clients that already have a local image path."""
    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path")
    patient_id = data.get("patient_id")
    if not image_path or not patient_id:
        return jsonify({"error": "image_path and patient_id are required"}), 400
    if not Path(image_path).exists():
        return jsonify({"error": "image not found"}), 404

    result = run_screening(
        image_path=image_path,
        patient_id=patient_id,
        eye=data.get("eye", "OD"),
        operator_id=data.get("operator_id", ""),
        phc_id=data.get("phc_id", ""),
    )
    return jsonify(result), 200


@screening_bp.get("/patients/<patient_id>/history")
def patient_history(patient_id):
    try:
        limit = int(request.args.get("limit", 10))
    except ValueError:
        limit = 10
    return jsonify(get_history(patient_id, limit=limit))
