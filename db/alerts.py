from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db.dao import get_connection

logger = logging.getLogger(__name__)


class MockSender:
    """
    Offline-safe alert backend for demonstrations.

    No network dependency.
    """

    def __init__(
        self,
        log_path: str | Path = "data/alerts.log",
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def send(
        self,
        screening_id: str,
        message: str,
        recipient: str = "",
    ) -> bool:

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat()
        )

        line = (
            f"{timestamp} | "
            f"screening={screening_id} | "
            f"recipient={recipient} | "
            f"{message}\n"
        )

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(line)

        print(
            "[MOCK ALERT]",
            line.strip(),
        )

        return True


def send_alert(
    result: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    recipient: str = "",
) -> bool:
    """
    Send an alert only when alerts are enabled and
    routing requires urgent referral.
    """

    cfg = cfg or {}

    alerts_cfg = (
        cfg.get("alerts") or {}
    )

    if not alerts_cfg.get(
        "enabled",
        False,
    ):
        return False

    routing = (
        result.get("routing") or {}
    )

    if routing.get("action") != (
        "URGENT_REFERRAL"
    ):
        return False

    sender = MockSender(
        alerts_cfg.get(
            "mock_log",
            "data/alerts.log",
        )
    )

    sent = sender.send(
        result["screening_id"],
        "URGENT_REFERRAL",
        recipient,
    )

    if sent:
        conn = get_connection()

        try:
            conn.execute(
                """
                INSERT INTO alerts (
                    screening_id,
                    channel,
                    recipient_hash,
                    sent_at,
                    status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result["screening_id"],
                    "MOCK",
                    None,
                    datetime.now(
                        timezone.utc
                    )
                    .astimezone()
                    .isoformat(),
                    "SENT",
                ),
            )

            conn.commit()

        finally:
            conn.close()

    return sent