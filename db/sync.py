from __future__ import annotations

import logging
import time
from typing import Any

from db.dao import (
    get_connection,
    get_pending_sync,
    mark_sync_attempt,
)

logger = logging.getLogger(__name__)


class MockSyncTransport:
    """
    Offline/demo transport.

    Replace with a real HTTP backend later without
    changing the queue implementation.
    """

    def upload(
        self,
        screening_id: str,
        payload: str,
    ) -> bool:
        logger.info(
            "Mock sync upload: %s (%d bytes)",
            screening_id,
            len(payload.encode("utf-8")),
        )

        return True


def process_pending(
    limit: int = 10,
    max_attempts: int = 5,
    transport: Any | None = None,
) -> dict[str, int]:

    transport = (
        transport
        or MockSyncTransport()
    )

    rows = get_pending_sync(
        limit=limit
    )

    synced = 0
    failed = 0

    conn = get_connection()

    try:

        for row in rows:

            screening_id = row[
                "screening_id"
            ]

            attempts = int(
                row["attempts"]
            )

            if attempts >= max_attempts:
                continue

            record = conn.execute(
                """
                SELECT result_json
                FROM screenings
                WHERE screening_id = ?
                """,
                (screening_id,),
            ).fetchone()

            if record is None:
                continue

            payload = record[
                "result_json"
            ]

            try:

                ok = transport.upload(
                    screening_id,
                    payload,
                )

                if ok:
                    mark_sync_attempt(
                        screening_id,
                        "SYNCED",
                    )
                    synced += 1

                else:
                    mark_sync_attempt(
                        screening_id,
                        "FAILED",
                    )
                    failed += 1

            except Exception:
                logger.exception(
                    "Sync failed: %s",
                    screening_id,
                )

                mark_sync_attempt(
                    screening_id,
                    "FAILED",
                )

                failed += 1

    finally:
        conn.close()

    return {
        "synced": synced,
        "failed": failed,
    }


def run_worker(
    interval_seconds: int = 30,
    max_attempts: int = 5,
):

    while True:

        process_pending(
            max_attempts=max_attempts
        )

        time.sleep(
            interval_seconds
        )