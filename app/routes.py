from flask import Blueprint, redirect, url_for, render_template, request, session, jsonify, send_file
from app.stubs import make_stub_result, make_stub_retake_result, make_stub_history
from app.report import generate_pdf

bp = Blueprint("main", __name__)
_results = {}

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
    return render_template("result.html", result=res)

@bp.route("/history/<patient_id>")
def history(patient_id):
    history_data = make_stub_history(patient_id)
    return render_template("history.html", history=history_data, patient_id=patient_id)

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
    if data and "lang" in data:
        session["lang"] = data["lang"]
        return jsonify({"status": "ok"})
    return jsonify({"error": "invalid"}), 400

@bp.route("/api/screening/<screening_id>")
def get_screening(screening_id):
    res = _results.get(screening_id)
    if not res:
        return jsonify({"error": "not found"}), 404
    return jsonify(res)
