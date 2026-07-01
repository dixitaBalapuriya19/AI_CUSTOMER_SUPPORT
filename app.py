"""Flask entry point for the AI Customer Support Triage System."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import (
    Flask,
    current_app,
    flash,
    redirect,
    render_template,
    render_template_string,
    request,
    session,
    send_file,
    url_for,
)
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import triage
import validator


logger = logging.getLogger(__name__)


@dataclass
class ProcessedRow:
    """A single row from the uploaded dataset after triage processing."""

    original_message: str
    category: str = ""
    priority: str = ""
    summary: str = ""
    suggested_action: str = ""
    needs_human: bool = False
    confidence: float = 0.0
    processing_status: str = "Pending"
    raw_json_response: Dict[str, Any] = field(default_factory=dict)
    prompt_injection_detected: bool = False
    expected_category: str = ""
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Original Message": self.original_message,
            "Category": self.category,
            "Priority": self.priority,
            "Summary": self.summary,
            "Suggested Action": self.suggested_action,
            "Needs Human": self.needs_human,
            "Confidence": self.confidence,
            "Processing Status": self.processing_status,
        }

    def to_dashboard_dict(self) -> Dict[str, Any]:
        return {
            **self.to_dict(),
            "Escalation Reason": _determine_escalation_reason(
                self.category,
                self.confidence,
                self.original_message,
                self.prompt_injection_detected,
            ),
            "Raw JSON Response": self.raw_json_response,
            "Prompt Injection Detected": self.prompt_injection_detected,
            "Review Label": "Needs Human Review" if self.needs_human else "Auto Resolved",
            "Row Class": "review-row-human" if self.needs_human else "review-row-auto",
        }


ALLOWED_EXTENSIONS = {"csv"}
RESULT_SESSION_KEY = "triage_results_path"
UPLOAD_SESSION_KEY = "triage_upload_path"
EVAL_SESSION_KEY = "triage_evaluation_report_path"

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Customer Support Triage</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --surface: rgba(255, 255, 255, 0.92);
      --surface-strong: #ffffff;
      --text: #102033;
      --muted: #5b6b7d;
      --primary: #1f6feb;
      --primary-dark: #1858bc;
      --border: rgba(16, 32, 51, 0.1);
      --danger: #b42318;
      --success: #067647;
      --warning: #b54708;
      --shadow: 0 20px 60px rgba(16, 32, 51, 0.12);
      --radius: 20px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(31, 111, 235, 0.15), transparent 36%),
        radial-gradient(circle at top right, rgba(6, 118, 71, 0.12), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #eef3f9 100%);
      color: var(--text);
      min-height: 100vh;
    }

    .container {
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }

    .hero {
      background: linear-gradient(135deg, rgba(31, 111, 235, 0.95), rgba(24, 88, 188, 0.92));
      color: white;
      border-radius: 28px;
      padding: 32px;
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: auto -80px -140px auto;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.08);
      filter: blur(2px);
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.9;
      margin-bottom: 12px;
    }

    h1 {
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 1.05;
      margin: 0 0 14px;
      max-width: 12ch;
    }

    .hero p {
      margin: 0;
      max-width: 70ch;
      line-height: 1.65;
      color: rgba(255, 255, 255, 0.9);
    }

    .panel {
      margin-top: 24px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px;
      backdrop-filter: blur(10px);
    }

    .upload-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }

    .file-picker {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }

    .btn,
    .file-label {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      min-height: 48px;
      padding: 0 18px;
      border-radius: 14px;
      border: 1px solid transparent;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
      user-select: none;
    }

    .btn:hover,
    .file-label:hover { transform: translateY(-1px); }

    .btn-primary {
      background: var(--primary);
      color: white;
      box-shadow: 0 14px 28px rgba(31, 111, 235, 0.25);
    }

    .btn-primary:hover { background: var(--primary-dark); }

    .btn-secondary,
    .file-label {
      background: white;
      color: var(--text);
      border-color: var(--border);
    }

    .file-label input {
      display: none;
    }

    .file-name {
      color: var(--muted);
      font-size: 0.95rem;
    }

    .note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.94rem;
      line-height: 1.5;
    }

    .flash {
      margin-top: 18px;
      border-radius: 14px;
      padding: 14px 16px;
      font-weight: 600;
      border: 1px solid transparent;
    }

    .flash.error {
      background: rgba(180, 35, 24, 0.08);
      color: var(--danger);
      border-color: rgba(180, 35, 24, 0.16);
    }

    .flash.success {
      background: rgba(6, 118, 71, 0.08);
      color: var(--success);
      border-color: rgba(6, 118, 71, 0.16);
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 16px;
      margin-top: 24px;
    }

    .stat-card {
      background: var(--surface-strong);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 28px rgba(16, 32, 51, 0.06);
    }

    .stat-card .label {
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 10px;
    }

    .stat-card .value {
      font-size: 1.8rem;
      font-weight: 800;
      letter-spacing: -0.03em;
    }

    .table-wrap {
      margin-top: 24px;
      overflow: auto;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: white;
      box-shadow: 0 12px 28px rgba(16, 32, 51, 0.06);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1080px;
    }

    thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      text-align: left;
      background: #f8fafc;
      color: #324559;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }

    tbody td {
      padding: 16px;
      border-bottom: 1px solid rgba(16, 32, 51, 0.08);
      vertical-align: top;
      line-height: 1.5;
    }

    tbody tr:hover {
      background: rgba(31, 111, 235, 0.03);
    }

    .status-success { color: var(--success); font-weight: 700; }
    .status-failed { color: var(--danger); font-weight: 700; }
    .muted { color: var(--muted); }
    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }

    .empty {
      padding: 48px 24px;
      text-align: center;
      color: var(--muted);
    }

    @media (max-width: 960px) {
      .upload-grid,
      .stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <div class="eyebrow">Customer Support Operations</div>
      <h1>AI Customer Support Triage</h1>
      <p>
        Upload a CSV file, analyze each customer message, and review validated triage results with export support for downstream workflows.
      </p>
    </section>

    <section class="panel">
      <form method="post" action="{{ url_for('analyze') }}" enctype="multipart/form-data">
        <div class="upload-grid">
          <div class="file-picker">
            <label class="file-label" for="file-input">
              Upload CSV
              <input id="file-input" type="file" name="file" accept=".csv,text/csv">
            </label>
            <span class="file-name" id="file-name">No file selected</span>
          </div>
          <button class="btn btn-primary" type="submit">Analyze Dataset</button>
        </div>
        <div class="note">
          Accepted format: CSV with a required <strong>message</strong> column. Uploads are processed temporarily and discarded after analysis.
        </div>
      </form>

      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="flash {{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
    </section>

    {% if stats %}
    <section class="stats">
      <div class="stat-card"><div class="label">Total Messages</div><div class="value">{{ stats.total }}</div></div>
      <div class="stat-card"><div class="label">Successfully Processed</div><div class="value">{{ stats.success }}</div></div>
      <div class="stat-card"><div class="label">Failed</div><div class="value">{{ stats.failed }}</div></div>
      <div class="stat-card"><div class="label">Average Confidence</div><div class="value">{{ stats.avg_confidence }}</div></div>
      <div class="stat-card"><div class="label">Human Review</div><div class="value">{{ stats.needs_human }}</div></div>
    </section>
    {% endif %}

    {% if results %}
    <section class="panel">
      <div class="actions">
        <a class="btn btn-secondary" href="{{ url_for('download_results') }}">Download output.json</a>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Original Message</th>
              <th>Category</th>
              <th>Priority</th>
              <th>Summary</th>
              <th>Suggested Action</th>
              <th>Needs Human</th>
              <th>Confidence</th>
              <th>Processing Status</th>
            </tr>
          </thead>
          <tbody>
            {% for row in results %}
            <tr>
              <td>{{ row['Original Message'] }}</td>
              <td>{{ row['Category'] }}</td>
              <td>{{ row['Priority'] }}</td>
              <td>{{ row['Summary'] }}</td>
              <td>{{ row['Suggested Action'] }}</td>
              <td>{{ row['Needs Human'] }}</td>
              <td>{{ row['Confidence'] }}</td>
              <td class="{{ 'status-success' if row['Processing Status'] == 'Success' else 'status-failed' }}">{{ row['Processing Status'] }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </section>
    {% elif results is not none %}
    <section class="panel empty">
      No messages were processed.
    </section>
    {% endif %}
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    if (fileInput && fileName) {
      fileInput.addEventListener('change', () => {
        fileName.textContent = fileInput.files.length ? fileInput.files[0].name : 'No file selected';
      });
    }
  </script>
</body>
</html>
"""


def create_app() -> Flask:
  """Application factory for the triage dashboard."""

  app = Flask(__name__)
  app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
  app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
  app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", tempfile.gettempdir())
  app.config["SESSION_COOKIE_HTTPONLY"] = True
  app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

  _configure_logging(app)

  @app.get("/")
  def index() -> str:
    return render_page()

  @app.get("/favicon.ico")
  def favicon() -> Any:
    return "", 204

  @app.post("/analyze")
  def analyze() -> Any:
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
      flash("Please choose a CSV file before analyzing.", "error")
      return redirect(url_for("index"))

    if not _allowed_file(uploaded_file.filename):
      flash("Only .csv files are supported.", "error")
      return redirect(url_for("index"))

    previous_export = session.pop(RESULT_SESSION_KEY, None)
    _safe_remove(previous_export)
    previous_eval_report = session.pop(EVAL_SESSION_KEY, None)
    _safe_remove(previous_eval_report)
    session.pop(UPLOAD_SESSION_KEY, None)

    upload_path = _save_upload_temporarily(uploaded_file)
    session[UPLOAD_SESSION_KEY] = upload_path

    try:
      dataframe = _load_dataset(upload_path)
      if dataframe.empty:
        flash("The uploaded CSV is empty.", "error")
        return render_page(results=[], stats=_empty_stats(), evaluation=None)

      if "message" not in dataframe.columns:
        flash('The uploaded CSV must contain a "message" column.', "error")
        return render_page(results=[], stats=_empty_stats(), evaluation=None)

      if "expected_category" not in dataframe.columns:
        logger.info("expected_category column not found; accuracy metrics will be hidden")

      processed_rows, evaluation_report = _process_dataset(dataframe)
      export_path = _write_export_file(processed_rows)
      eval_export_path = _write_evaluation_report_file(evaluation_report)
      session[RESULT_SESSION_KEY] = export_path
      session[EVAL_SESSION_KEY] = eval_export_path

      stats = _build_stats(processed_rows)
      flash("Dataset analyzed successfully.", "success")
      return render_page(
        results=[row.to_dashboard_dict() for row in processed_rows],
        stats=stats,
        evaluation=evaluation_report,
      )
    except ValueError as exc:
      logger.warning(
        "triage csv validation failed",
        extra={"event": "triage_csv_validation_failed", "error": str(exc)},
      )
      flash(str(exc), "error")
      return render_page(results=[], stats=_empty_stats(), evaluation=None)
    except pd.errors.EmptyDataError:
      flash("The uploaded CSV is empty.", "error")
      return render_page(results=[], stats=_empty_stats(), evaluation=None)
    except pd.errors.ParserError:
      flash("The uploaded file could not be read as a valid CSV.", "error")
      return render_page(results=[], stats=_empty_stats(), evaluation=None)
    except Exception as exc:
      logger.exception(
        "triage processing failed",
        extra={"event": "triage_processing_failed", "error": str(exc)},
      )
      flash("Something went wrong while processing the file. Please try again.", "error")
      return render_page(results=[], stats=_empty_stats(), evaluation=None)
    finally:
      _safe_remove(upload_path)
      session.pop(UPLOAD_SESSION_KEY, None)

  @app.get("/download/output.json")
  def download_results() -> Any:
    export_path = session.get(RESULT_SESSION_KEY)
    if not export_path or not Path(export_path).exists():
      flash("No analyzed results are available to download.", "error")
      return redirect(url_for("index"))

    return send_file(
      export_path,
      as_attachment=True,
      download_name="output.json",
      mimetype="application/json",
    )

  @app.get("/download/evaluation_report.json")
  def download_evaluation_report() -> Any:
    report_path = session.get(EVAL_SESSION_KEY)
    if not report_path or not Path(report_path).exists():
      flash("No evaluation report is available to download.", "error")
      return redirect(url_for("index"))

    return send_file(
      report_path,
      as_attachment=True,
      download_name="evaluation_report.json",
      mimetype="application/json",
    )

  @app.errorhandler(413)
  def request_entity_too_large(_: Exception) -> Any:
    flash("The uploaded file is too large.", "error")
    return redirect(url_for("index"))

  @app.errorhandler(Exception)
  def handle_unexpected_error(error: Exception) -> Any:
    if isinstance(error, HTTPException):
      return error

    logger.exception(
      "unexpected application error",
      extra={"event": "unexpected_application_error", "error": str(error)},
    )
    flash("An unexpected error occurred. Please try again.", "error")
    return redirect(url_for("index")), 500

  return app


def render_page(
  *,
  results: Optional[List[Dict[str, Any]]] = None,
  stats: Optional[Dict[str, Any]] = None,
  evaluation: Optional[Dict[str, Any]] = None,
) -> str:
  """Render the dashboard HTML with optional results, stats, and evaluation."""

  return render_template(
    "index.html",
    results=results,
    stats=stats,
    evaluation=evaluation,
    output_exists=bool(results),
    evaluation_exists=bool(evaluation),
  )


def _configure_logging(app: Flask) -> None:
  if not logging.getLogger().handlers:
    logging.basicConfig(
      level=os.getenv("LOG_LEVEL", "INFO").upper(),
      format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
  app.logger.handlers = logging.getLogger().handlers
  app.logger.setLevel(logging.getLogger().level)


def _allowed_file(filename: str) -> bool:
  return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload_temporarily(uploaded_file: Any) -> str:
  filename = secure_filename(uploaded_file.filename or "dataset.csv")
  upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
  upload_dir.mkdir(parents=True, exist_ok=True)
  temp_file = tempfile.NamedTemporaryFile(
    mode="wb",
    suffix=f"_{filename}",
    delete=False,
    dir=upload_dir,
  )
  try:
    uploaded_file.save(temp_file.name)
    return temp_file.name
  finally:
    temp_file.close()


def _load_dataset(path: str) -> pd.DataFrame:
  dataframe = pd.read_csv(path)
  if dataframe is None:
    raise ValueError("The uploaded CSV could not be loaded.")
  return dataframe


def _process_dataset(dataframe: pd.DataFrame) -> tuple[List[ProcessedRow], Dict[str, Any]]:
  messages = [_coerce_message(raw_message) for raw_message in dataframe["message"].tolist()]
  expected_categories = (
    [_coerce_optional_text(value) for value in dataframe["expected_category"].tolist()]
    if "expected_category" in dataframe.columns
    else [""] * len(messages)
  )

  rows: List[ProcessedRow] = []
  batch_count = 0
  total_api_calls_made = 0
  retry_count = 0
  failed_requests = 0
  skipped_empty_messages = 0
  batch_durations: List[float] = []
  overall_start = time.perf_counter()

  indexed_records = list(enumerate(zip(messages, expected_categories), start=1))
  for batch_number, batch_records in enumerate(_chunk_records(indexed_records, batch_size=10), start=1):
    batch_count += 1
    batch_start = time.perf_counter()
    batch_row_indices = [record[0] for record in batch_records]
    batch_messages = [record[1][0] for record in batch_records]
    batch_expected_categories = [record[1][1] for record in batch_records]

    logger.info(
      f"Batch {batch_number} started",
      extra={
        "event": "batch_started",
        "batch_number": batch_number,
        "batch_size": len(batch_messages),
        "row_indices": batch_row_indices,
        "first_messages": batch_messages[:2],
      },
    )

    try:
      batch_row_map: Dict[int, ProcessedRow] = {}
      valid_records: List[tuple[int, str, str]] = []
      for row_index, (message, expected_category) in batch_records:
        if _is_empty_customer_message(message):
          skipped_empty_messages += 1
          logger.info(
            f"Skipped empty message at row {row_index}.",
            extra={
              "event": "empty_message_skipped",
              "row_index": row_index,
              "batch_number": batch_number,
            },
          )
          batch_row_map[row_index] = _build_processed_row(
            original_message="",
            expected_category=expected_category,
            response=_empty_message_response(),
          )
          continue

        valid_records.append((row_index, message, expected_category))

      batch_results: List[Dict[str, Any]] = []
      if valid_records:
        valid_row_indices = [record[0] for record in valid_records]
        valid_messages = [record[1] for record in valid_records]
        logger.info(
          "API request started",
          extra={
            "event": "api_request_started",
            "batch_number": batch_number,
            "batch_size": len(valid_messages),
            "row_indices": valid_row_indices,
            "first_messages": valid_messages[:2],
          },
        )

        triage.set_batch_context(batch_number)
        batch_results = triage.analyze_messages(valid_messages)
        batch_metrics = triage.get_batch_metrics()
        total_api_calls_made += batch_metrics.get("api_calls_made", 0)
        retry_count += batch_metrics.get("retry_count", 0)

        logger.info(
          "API request completed",
          extra={
            "event": "api_request_completed",
            "batch_number": batch_number,
            "batch_size": len(valid_messages),
            "row_indices": valid_row_indices,
            "api_calls_made": batch_metrics.get("api_calls_made", 0),
            "retry_count": batch_metrics.get("retry_count", 0),
            "quota_exceeded": batch_metrics.get("quota_exceeded", 0),
          },
        )

        valid_result_iter = iter(batch_results)
        for row_index, message, expected_category in valid_records:
          batch_row_map[row_index] = _build_processed_row(
            original_message=message,
            expected_category=expected_category,
            response=next(valid_result_iter, {}),
          )

      batch_duration = time.perf_counter() - batch_start
      per_row_duration = batch_duration / max(len(batch_records), 1)
      for row in batch_row_map.values():
        row.processing_time_seconds = per_row_duration
      batch_durations.append(batch_duration)

      if _is_quota_fallback_batch(batch_results):
        failed_requests += 1

      rows.extend(batch_row_map[row_index] for row_index in batch_row_indices)

      logger.info(
        f"Batch {batch_number} completed",
        extra={
          "event": "batch_completed",
          "batch_number": batch_number,
          "message_count": len(batch_messages),
          "skipped_empty_messages": sum(1 for message in batch_messages if _is_empty_customer_message(message)),
        },
      )
    except Exception as exc:
      batch_duration = time.perf_counter() - batch_start
      batch_durations.append(batch_duration)
      failed_requests += 1

      logger.exception(
        f"Batch {batch_number} failed: {exc}",
        extra={
          "event": "batch_failed",
          "batch_number": batch_number,
          "batch_size": len(batch_messages),
          "row_indices": batch_row_indices,
          "first_messages": batch_messages[:2],
          "error": str(exc),
        },
      )

      batch_row_map = {
        row_index: _build_processed_row(
          original_message=message,
          expected_category=expected_category,
          response=_build_batch_failure_response(str(exc)),
          status=f"Failed: {exc}",
        )
        for row_index, (message, expected_category) in batch_records
        if not _is_empty_customer_message(message)
      }

      for row_index, (message, expected_category) in batch_records:
        if _is_empty_customer_message(message):
          batch_row_map[row_index] = _build_processed_row(
            original_message="",
            expected_category=expected_category,
            response=_empty_message_response(),
          )

      per_row_duration = batch_duration / max(len(batch_records), 1)
      for row in batch_row_map.values():
        row.processing_time_seconds = per_row_duration
      rows.extend(batch_row_map[row_index] for row_index in batch_row_indices)
    finally:
      triage.set_batch_context(None)

  total_runtime = time.perf_counter() - overall_start
  logger.info(
    "Finished processing dataset",
    extra={
      "event": "dataset_processing_finished",
      "message_count": len(messages),
      "result_count": len(rows),
      "api_calls_made": total_api_calls_made,
      "retry_count": retry_count,
      "skipped_empty_messages": skipped_empty_messages,
    },
  )

  evaluation_report = _build_evaluation_report(
    rows=rows,
    message_count=len(messages),
    batch_count=batch_count,
    retry_count=retry_count,
    failed_requests=failed_requests,
    batch_durations=batch_durations,
    total_runtime=total_runtime,
  )
  return rows, evaluation_report


def _chunk_records(records: List[tuple[int, tuple[str, str]]], batch_size: int) -> List[List[tuple[int, tuple[str, str]]]]:
  return [records[index:index + batch_size] for index in range(0, len(records), batch_size)]


def _build_processed_row(
  *,
  original_message: str,
  expected_category: str,
  response: Dict[str, Any],
  status: str = "Success",
) -> ProcessedRow:
  try:
    validated = validator.validate_response(response)
  except Exception:
    validated = validator.validate_response({})

  response_status = response.get("processing_status", status) if isinstance(response, dict) else status
  return ProcessedRow(
    original_message=original_message,
    category=validated["category"],
    priority=validated["priority"],
    summary=validated["summary"],
    suggested_action=validated["suggested_action"],
    needs_human=validated["needs_human"],
    confidence=_safe_float(validated["confidence"]),
    raw_json_response=response if isinstance(response, dict) else validated,
    prompt_injection_detected=_detect_prompt_injection(original_message),
    expected_category=expected_category,
    processing_time_seconds=0.0,
    processing_status=str(response_status),
  )


def _build_batch_failure_response(error_message: str) -> Dict[str, Any]:
  return {
    "category": "Other",
    "priority": "P3",
    "summary": "Batch processing failed.",
    "suggested_action": "Route to human support.",
    "needs_human": True,
    "confidence": 0.0,
    "processing_status": f"Failed: {error_message}",
  }


def _empty_message_response() -> Dict[str, Any]:
  return {
    "category": "Other",
    "priority": "P3",
    "summary": "Empty customer message.",
    "suggested_action": "Request additional information from the customer.",
    "needs_human": True,
    "confidence": 0.0,
    "processing_status": "Skipped - Empty Message",
  }


def _write_export_file(rows: List[ProcessedRow]) -> str:
  payload = [row.to_dict() for row in rows]
  export_file = tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".json",
    delete=False,
    dir=current_app.config["UPLOAD_FOLDER"],
    encoding="utf-8",
  )
  try:
    json.dump(payload, export_file, ensure_ascii=False, indent=2)
    export_file.flush()
    return export_file.name
  finally:
    export_file.close()


def _write_evaluation_report_file(report: Dict[str, Any]) -> str:
  export_file = tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".json",
    delete=False,
    dir=current_app.config["UPLOAD_FOLDER"],
    encoding="utf-8",
  )
  try:
    json.dump(report, export_file, ensure_ascii=False, indent=2)
    export_file.flush()
    return export_file.name
  finally:
    export_file.close()


def _build_stats(rows: List[ProcessedRow]) -> Dict[str, Any]:
  total = len(rows)
  success = sum(1 for row in rows if row.processing_status == "Success")
  failed = total - success
  confidences = [row.confidence for row in rows if row.processing_status == "Success"]
  needs_human = sum(1 for row in rows if row.processing_status == "Success" and row.needs_human)
  auto_resolved = sum(1 for row in rows if not row.needs_human and row.processing_status == "Success")
  average_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
  review_percentage = round((needs_human / total) * 100, 1) if total else 0.0

  return {
    "total": total,
    "success": success,
    "failed": failed,
    "avg_confidence": f"{average_confidence:.2f}",
    "needs_human": needs_human,
    "auto_resolved": auto_resolved,
    "review_percentage": f"{review_percentage:.1f}%",
  }


def _empty_stats() -> Dict[str, Any]:
  return {
    "total": 0,
    "success": 0,
    "failed": 0,
    "avg_confidence": "0.00",
    "needs_human": 0,
    "auto_resolved": 0,
    "review_percentage": "0.0%",
  }


def _build_evaluation_report(
  *,
  rows: List[ProcessedRow],
  message_count: int,
  batch_count: int,
  retry_count: int,
  failed_requests: int,
  batch_durations: List[float],
  total_runtime: float,
) -> Dict[str, Any]:
  labeled_rows = [row for row in rows if row.expected_category]
  accuracy_available = bool(labeled_rows)
  normalized_labeled_rows = [
    (row, _normalize_evaluation_category(row.expected_category), _normalize_evaluation_category(row.category))
    for row in labeled_rows
  ]
  correct_predictions = sum(1 for _, expected_category, predicted_category in normalized_labeled_rows if expected_category == predicted_category)
  incorrect_predictions = len(labeled_rows) - correct_predictions
  overall_accuracy = round((correct_predictions / len(labeled_rows)) * 100, 1) if labeled_rows else None

  average_confidence = _safe_mean([row.confidence for row in rows])
  human_review_count = sum(1 for row in rows if row.needs_human)
  human_review_percentage = round((human_review_count / message_count) * 100, 1) if message_count else 0.0
  average_processing_time = _safe_mean([row.processing_time_seconds for row in rows])
  fastest_prediction = min((row.processing_time_seconds for row in rows), default=0.0)
  slowest_prediction = max((row.processing_time_seconds for row in rows), default=0.0)
  average_batch_size = round(message_count / batch_count, 2) if batch_count else 0.0

  predicted_distribution = _count_values([_normalize_evaluation_category(row.category) for row in rows])
  expected_distribution = _count_values([_normalize_evaluation_category(row.expected_category) for row in labeled_rows]) if accuracy_available else {}

  category_performance = []
  if accuracy_available:
    categories = sorted(
      set(expected_distribution.keys()) | set(predicted_distribution.keys())
    )
    for category in categories:
      expected_count = sum(
        1 for row in labeled_rows
        if _normalize_evaluation_category(row.expected_category) == category
      )
      predicted_count = sum(
        1 for row in rows
        if _normalize_evaluation_category(row.category) == category
      )
      correct_count = sum(
        1 for row in labeled_rows
        if _normalize_evaluation_category(row.expected_category) == category
        and _normalize_evaluation_category(row.category) == category
      )
      accuracy = round((correct_count / expected_count) * 100, 1) if expected_count else 0.0
      category_performance.append(
        {
          "category": category,
          "expected": expected_count,
          "predicted": predicted_count,
          "correct": correct_count,
          "accuracy": f"{accuracy:.1f}%",
        }
      )

  known_failures = _build_known_failures(rows)

  provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
  model_name = os.getenv("GEMINI_MODEL", "Gemini 2.5 Flash") if provider == "gemini" else os.getenv("OPENAI_MODEL", "OpenAI Model")

  report = {
    "accuracy": {
      "available": accuracy_available,
      "dataset_size": message_count,
      "correct_predictions": correct_predictions if accuracy_available else None,
      "incorrect_predictions": incorrect_predictions if accuracy_available else None,
      "overall_accuracy": f"{overall_accuracy:.1f}%" if overall_accuracy is not None else None,
      "category_performance": category_performance,
    },
    "metrics": {
      "dataset_size": message_count,
      "correct_predictions": correct_predictions if accuracy_available else None,
      "incorrect_predictions": incorrect_predictions if accuracy_available else None,
      "overall_accuracy": f"{overall_accuracy:.1f}%" if overall_accuracy is not None else None,
      "average_confidence": f"{average_confidence:.2f}",
      "human_review_percentage": f"{human_review_percentage:.1f}%",
      "average_processing_time": _format_seconds(average_processing_time),
      "api_calls_made": batch_count + retry_count,
      "retry_count": retry_count,
      "failed_requests": failed_requests,
      "average_batch_size": f"{average_batch_size:.2f}",
      "estimated_cost": "$0 (Free Tier)" if provider == "gemini" else "N/A",
    },
    "category_distribution": {
      "expected": expected_distribution,
      "predicted": predicted_distribution,
    },
    "latency": {
      "average_processing_time": _format_seconds(average_processing_time),
      "fastest_prediction": _format_seconds(fastest_prediction),
      "slowest_prediction": _format_seconds(slowest_prediction),
      "total_runtime": _format_seconds(total_runtime),
    },
    "retry_count": retry_count,
    "processing_summary": {
      "dataset_size": message_count,
      "batch_count": batch_count,
      "average_batch_size": f"{average_batch_size:.2f}",
      "human_review_count": human_review_count,
      "auto_resolved_count": sum(1 for row in rows if not row.needs_human),
      "failed_requests": failed_requests,
      "successful_predictions": len(rows) - failed_requests,
      "runtime_seconds": round(total_runtime, 2),
    },
    "known_failures": known_failures,
    "model_information": {
      "model": model_name,
      "batch_size": 10,
      "retry_strategy": "One retry on HTTP 429 with exponential backoff",
      "confidence_threshold": 0.70,
      "validator_enabled": True,
      "prompt_injection_protection": True,
    },
  }

  return report


def _normalize_evaluation_category(category: str) -> str:
  normalized = (category or "").strip().lower()
  synonyms = {
    "technical": "Technical Support",
    "technical support": "Technical Support",
    "tech support": "Technical Support",
    "shipping": "Order",
    "order": "Order",
    "payment": "Billing",
    "billing": "Billing",
    "charge": "Billing",
    "refund": "Billing",
    "complaint": "Complaint",
    "fraud": "Fraud",
    "security": "Security",
    "account": "Account",
    "feature": "Feature Request",
    "feature request": "Feature Request",
    "general inquiry": "General Inquiry",
    "general question": "General Inquiry",
    "question": "General Inquiry",
    "spam": "Spam",
    "other": "Other",
    "unknown": "Unknown",
  }
  return synonyms.get(normalized, category.strip() if isinstance(category, str) and category.strip() else "Unknown")


def _build_known_failures(rows: List[ProcessedRow]) -> List[Dict[str, Any]]:
  failure_counts: Dict[str, int] = {}
  for row in rows:
    labels = []
    if _is_multi_intent(row.original_message):
      labels.append("Multi-intent messages")
    if _is_vague_request(row.original_message, row.category):
      labels.append("Very vague requests")
    if _is_sarcastic(row.original_message):
      labels.append("Heavy sarcasm")
    if _is_security_prompt(row.original_message):
      labels.append("Security prompts")
    if row.prompt_injection_detected:
      labels.append("Prompt injections")
    if len(row.original_message) > 250:
      labels.append("Long customer messages")

    for label in labels:
      failure_counts[label] = failure_counts.get(label, 0) + 1

  return [
    {"label": label, "count": count}
    for label, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
  ]


def _count_values(values: List[str]) -> Dict[str, int]:
  counts: Dict[str, int] = {}
  for value in values:
    if not value:
      continue
    counts[value] = counts.get(value, 0) + 1
  return counts


def _safe_mean(values: List[float]) -> float:
  filtered = [value for value in values if isinstance(value, (int, float))]
  if not filtered:
    return 0.0
  return round(sum(filtered) / len(filtered), 2)


def _format_seconds(seconds: float) -> str:
  return f"{seconds:.2f}s"


def _coerce_message(value: Any) -> str:
  if value is None or pd.isna(value):
    return ""
  if isinstance(value, str):
    return value.strip()
  return str(value).strip()


def _is_empty_customer_message(value: Any) -> bool:
  if value is None:
    return True
  if isinstance(value, str):
    return not value.strip()
  try:
    return bool(pd.isna(value))
  except Exception:
    return not str(value).strip()


def _coerce_optional_text(value: Any) -> str:
  if value is None or pd.isna(value):
    return ""
  return str(value).strip()


def _safe_float(value: Any) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return 0.0


def _detect_prompt_injection(message: str) -> bool:
  lowered = message.lower()
  patterns = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "act as system",
    "bypass the usual process",
    "reveal the internal policy",
    "admin password",
  ]
  return any(pattern in lowered for pattern in patterns)


def _is_quota_fallback_batch(batch_results: List[Dict[str, Any]]) -> bool:
  if not batch_results:
    return False

  fallback_signature = {
    "category": "Unknown",
    "priority": "P3",
    "summary": "LLM unavailable due to API quota.",
    "suggested_action": "Retry later or route to human support.",
    "needs_human": True,
    "confidence": 0.0,
  }

  for result in batch_results:
    if not isinstance(result, dict):
      return False
    for key, expected_value in fallback_signature.items():
      if result.get(key) != expected_value:
        return False
  return True


def _build_escalation_reason(
  category: str,
  confidence: float,
  original_message: str,
  prompt_injection_detected: bool,
) -> str:
  if prompt_injection_detected:
    return "Security Risk"
  if confidence < 0.70:
    return "Low Confidence"
  if category in {"Billing", "Complaint", "Fraud", "Security"}:
    return "Business Rule"
  return "Complex Case"


def _determine_escalation_reason(
  category: str,
  confidence: float,
  original_message: str,
  prompt_injection_detected: bool,
) -> str:
  return _build_escalation_reason(category, confidence, original_message, prompt_injection_detected)


def _is_multi_intent(message: str) -> bool:
  lowered = message.lower()
  intent_hits = sum(
    1
    for keyword in ["billing", "refund", "charge", "login", "password", "order", "shipping", "account", "crash", "bug", "cancel", "subscription"]
    if keyword in lowered
  )
  return intent_hits >= 2


def _is_vague_request(message: str, category: str) -> bool:
  lowered = message.lower().strip()
  if len(lowered) < 40:
    return True
  vague_phrases = ["help", "issue", "problem", "something wrong", "not working", "what is happening"]
  return category == "Other" or any(phrase in lowered for phrase in vague_phrases)


def _is_sarcastic(message: str) -> bool:
  lowered = message.lower()
  return any(
    phrase in lowered
    for phrase in ["love that for me", "great, another", "very impressive", "thanks a lot", "sure, because"]
  )


def _is_security_prompt(message: str) -> bool:
  lowered = message.lower()
  return any(
    phrase in lowered
    for phrase in ["security", "phishing", "hack", "breach", "unauthorized", "password reset", "verify your identity"]
  )


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "temporary file cleanup failed",
            extra={"event": "temporary_file_cleanup_failed", "path": path},
        )


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )