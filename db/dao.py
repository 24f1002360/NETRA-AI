from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "netra.db"


# ============================================================
# DATABASE
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    phc_id TEXT,
    name_hash TEXT,
    age INTEGER,
    sex TEXT,
    diabetes_years INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screenings (
    screening_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    eye TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    operator_id TEXT,
    quality_verdict TEXT,
    icdr_grade INTEGER,
    confidence REAL,
    referable INTEGER,
    guard_status TEXT,
    routing_action TEXT,
    sync_status TEXT,
    result_json TEXT NOT NULL,
    raw_image_path TEXT,
    overlay_path TEXT,

    FOREIGN KEY(patient_id)
        REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id TEXT NOT NULL UNIQUE,
    payload_bytes INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING'
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient_hash TEXT,
    sent_at TEXT,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_screen_patient
    ON screenings(patient_id, captured_at);

CREATE INDEX IF NOT EXISTS idx_sync_status
    ON sync_queue(status);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _db_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path or DEFAULT_DB_PATH)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def get_connection(
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    """
    Open a SQLite connection and ensure the database schema exists.
    """

    path = _db_path(db_path)

    conn = sqlite3.connect(
        str(path),
        timeout=10,
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(SCHEMA_SQL)

    return conn


# ============================================================
# PATIENT
# ============================================================

def _ensure_patient(
    conn: sqlite3.Connection,
    result: dict[str, Any],
) -> None:
    patient_id = result["patient_id"]
    phc_id = result.get("phc_id")

    conn.execute(
        """
        INSERT INTO patients (
            patient_id,
            phc_id,
            created_at
        )
        VALUES (?, ?, ?)
        ON CONFLICT(patient_id)
        DO UPDATE SET
            phc_id = COALESCE(excluded.phc_id, patients.phc_id)
        """,
        (
            patient_id,
            phc_id,
            _now_iso(),
        ),
    )


# ============================================================
# SAVE SCREENING
# ============================================================

def save_screening(
    result: dict[str, Any],
    db_path: str | Path | None = None,
) -> str:
    """
    Persist the complete ScreeningResult.

    The complete object is stored in result_json while commonly
    queried fields are denormalised into screening columns.
    """

    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary")

    screening_id = result["screening_id"]

    quality = result.get("quality") or {}
    grading = result.get("grading") or {}
    xai = result.get("xai") or {}
    routing = result.get("routing") or {}
    image = result.get("image") or {}

    result_json = json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    payload_bytes = len(
        result_json.encode("utf-8")
    )

    sync_status = routing.get(
        "sync_status",
        "PENDING",
    )

    conn = get_connection(db_path)

    try:
        _ensure_patient(
            conn,
            result,
        )

        conn.execute(
            """
            INSERT INTO screenings (
                screening_id,
                patient_id,
                eye,
                captured_at,
                operator_id,
                quality_verdict,
                icdr_grade,
                confidence,
                referable,
                guard_status,
                routing_action,
                sync_status,
                result_json,
                raw_image_path,
                overlay_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(screening_id)
            DO UPDATE SET
                quality_verdict = excluded.quality_verdict,
                icdr_grade = excluded.icdr_grade,
                confidence = excluded.confidence,
                referable = excluded.referable,
                guard_status = excluded.guard_status,
                routing_action = excluded.routing_action,
                sync_status = excluded.sync_status,
                result_json = excluded.result_json,
                raw_image_path = excluded.raw_image_path,
                overlay_path = excluded.overlay_path
            """,
            (
                screening_id,
                result["patient_id"],
                result["eye"],
                result["captured_at"],
                result.get("operator_id"),
                quality.get("verdict"),
                grading.get("icdr_grade"),
                grading.get("confidence"),
                int(bool(grading.get("referable_dr", False))),
                xai.get("guard_status"),
                routing.get("action"),
                sync_status,
                result_json,
                image.get("raw_path"),
                xai.get("overlay_path"),
            ),
        )

        conn.execute(
            """
            INSERT INTO sync_queue (
                screening_id,
                payload_bytes,
                attempts,
                status
            )
            VALUES (?, ?, 0, ?)
            ON CONFLICT(screening_id)
            DO UPDATE SET
                payload_bytes = excluded.payload_bytes,
                status = excluded.status
            """,
            (
                screening_id,
                payload_bytes,
                sync_status,
            ),
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    return screening_id


# ============================================================
# HISTORY
# ============================================================

def get_history(
    patient_id: str,
    limit: int = 10,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return the most recent ScreeningResult objects for a patient.
    """

    if limit <= 0:
        return []

    conn = get_connection(db_path)

    try:
        rows = conn.execute(
            """
            SELECT result_json
            FROM screenings
            WHERE patient_id = ?
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            (
                patient_id,
                int(limit),
            ),
        ).fetchall()

    finally:
        conn.close()

    results = []

    for row in rows:
        try:
            results.append(
                json.loads(row["result_json"])
            )
        except json.JSONDecodeError:
            # Corrupt JSON should not silently become
            # an invented ScreeningResult.
            continue

    return results


# ============================================================
# LONGITUDINAL
# ============================================================

def compare_with_prior(
    patient_id: str,
    result: dict[str, Any],
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare the current screening with the most recent prior
    screening for the same patient and eye.
    """

    eye = result["eye"]
    current_id = result.get("screening_id")

    conn = get_connection(db_path)

    try:
        row = conn.execute(
            """
            SELECT screening_id, result_json
            FROM screenings
            WHERE patient_id = ?
              AND eye = ?
              AND screening_id != ?
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            (
                patient_id,
                eye,
                current_id or "",
            ),
        ).fetchone()

    finally:
        conn.close()

    if row is None:
        return {
            "prior_screening_id": None,
            "prior_grade": None,
            "delta": None,
            "trend": "FIRST_VISIT",
        }

    try:
        prior = json.loads(
            row["result_json"]
        )
    except json.JSONDecodeError:
        return {
            "prior_screening_id": row["screening_id"],
            "prior_grade": None,
            "delta": None,
            "trend": "FIRST_VISIT",
        }

    prior_grading = (
        prior.get("grading") or {}
    )

    prior_grade = prior_grading.get(
        "icdr_grade"
    )

    current_grading = (
        result.get("grading") or {}
    )

    current_grade = current_grading.get(
        "icdr_grade"
    )

    if prior_grade is None or current_grade is None:
        return {
            "prior_screening_id": row["screening_id"],
            "prior_grade": prior_grade,
            "delta": None,
            "trend": "STABLE",
        }

    delta_value = int(current_grade) - int(prior_grade)

    if delta_value > 0:
        trend = "WORSENING"
    elif delta_value < 0:
        trend = "IMPROVING"
    else:
        trend = "STABLE"

    return {
        "prior_screening_id": row["screening_id"],
        "prior_grade": int(prior_grade),
        "delta": f"{delta_value:+d}",
        "trend": trend,
    }


# ============================================================
# ROUTING
# ============================================================

def compute_routing(
    result: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute the Divyanshu routing block.

    Priority:
        1. urgent referral
        2. review
        3. routine
    """

    thresholds = thresholds or {}

    routing_cfg = (
        thresholds.get("routing")
        or {}
    )

    urgent_grade = int(
        routing_cfg.get(
            "urgent_grade",
            4,
        )
    )

    referable_grade = int(
        routing_cfg.get(
            "referable_grade",
            2,
        )
    )

    low_confidence = float(
        routing_cfg.get(
            "low_confidence",
            0.55,
        )
    )

    grading = result.get("grading") or {}
    xai = result.get("xai") or {}
    conditions = (
        result.get("other_conditions")
        or {}
    )

    grade = grading.get(
        "icdr_grade"
    )

    confidence = float(
        grading.get(
            "confidence",
            0.0,
        ) or 0.0
    )

    guard_status = xai.get(
        "guard_status",
        "OK",
    )

    glaucoma = (
        conditions
        .get("glaucoma_suspect")
        or {}
    )

    glaucoma_flag = bool(
        glaucoma.get("flag", False)
    )

    # --------------------------------------------------------
    # URGENT REFERRAL
    # --------------------------------------------------------

    if grade is not None:

        if int(grade) >= urgent_grade:
            return {
                "action": "URGENT_REFERRAL",
                "reason": "URGENT_GRADE",
                "alert_sent": False,
                "sync_status": "PENDING",
            }

        if (
            int(grade) == 3
            and confidence >= 0.70
        ):
            return {
                "action": "URGENT_REFERRAL",
                "reason": "GRADE_3_HIGH_CONFIDENCE",
                "alert_sent": False,
                "sync_status": "PENDING",
            }

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    if (
        grade is not None
        and int(grade) >= referable_grade
    ):
        return {
            "action": "REVIEW",
            "reason": "REFERABLE_DR",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

    if confidence < low_confidence:
        return {
            "action": "REVIEW",
            "reason": "LOW_CONFIDENCE",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

    if guard_status != "OK":
        return {
            "action": "REVIEW",
            "reason": "XAI_GUARD",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

    if glaucoma_flag:
        return {
            "action": "REVIEW",
            "reason": "GLAUCOMA_SUSPECT",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

    # --------------------------------------------------------
    # ROUTINE
    # --------------------------------------------------------

    return {
        "action": "ROUTINE",
        "reason": "NO_REFERRAL_CRITERIA",
        "alert_sent": False,
        "sync_status": "PENDING",
    }


# ============================================================
# SYNC QUEUE
# ============================================================

def enqueue_sync(
    screening_id: str,
    db_path: str | Path | None = None,
) -> None:
    """
    Ensure a screening exists in the offline sync queue.
    """

    conn = get_connection(db_path)

    try:
        row = conn.execute(
            """
            SELECT result_json
            FROM screenings
            WHERE screening_id = ?
            """,
            (screening_id,),
        ).fetchone()

        if row is None:
            raise ValueError(
                f"Unknown screening_id: {screening_id}"
            )

        payload_bytes = len(
            row["result_json"].encode(
                "utf-8"
            )
        )

        conn.execute(
            """
            INSERT INTO sync_queue (
                screening_id,
                payload_bytes,
                attempts,
                status
            )
            VALUES (?, ?, 0, 'PENDING')
            ON CONFLICT(screening_id)
            DO UPDATE SET
                payload_bytes = excluded.payload_bytes,
                status = 'PENDING'
            """,
            (
                screening_id,
                payload_bytes,
            ),
        )

        conn.execute(
            """
            UPDATE screenings
            SET sync_status = 'PENDING'
            WHERE screening_id = ?
            """,
            (screening_id,),
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# SYNC QUEUE HELPERS
# ============================================================

def get_pending_sync(
    limit: int = 10,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return pending sync records for a future background worker.
    """

    if limit <= 0:
        return []

    conn = get_connection(db_path)

    try:
        rows = conn.execute(
            """
            SELECT
                id,
                screening_id,
                payload_bytes,
                attempts,
                last_attempt,
                status
            FROM sync_queue
            WHERE status IN ('PENDING', 'FAILED')
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    finally:
        conn.close()

    return [
        dict(row)
        for row in rows
    ]


def mark_sync_attempt(
    screening_id: str,
    status: str,
    db_path: str | Path | None = None,
) -> None:
    """
    Record one sync attempt.

    status must be PENDING, SYNCED or FAILED.
    """

    allowed = {
        "PENDING",
        "SYNCED",
        "FAILED",
    }

    if status not in allowed:
        raise ValueError(
            f"Invalid sync status: {status}"
        )

    conn = get_connection(db_path)

    try:
        conn.execute(
            """
            UPDATE sync_queue
            SET
                attempts = attempts + 1,
                last_attempt = ?,
                status = ?
            WHERE screening_id = ?
            """,
            (
                _now_iso(),
                status,
                screening_id,
            ),
        )

        conn.execute(
            """
            UPDATE screenings
            SET sync_status = ?
            WHERE screening_id = ?
            """,
            (
                status,
                screening_id,
            ),
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()