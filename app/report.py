import time
import os

def render_result(result: dict) -> str:
    start = time.perf_counter_ns()
    html_string = f"<h1>Screening Result {result['screening_id']}</h1>"
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return html_string

def generate_pdf(result: dict) -> str:
    start = time.perf_counter_ns()
    report_dir = os.path.join('data', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"{result['screening_id']}_report.html")
    with open(path, 'w') as f:
        f.write(f"<html><body><h1>Report {result['screening_id']}</h1></body></html>")
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return path
