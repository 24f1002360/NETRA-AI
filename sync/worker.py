from __future__ import annotations

import logging
import time

from db.dao import (
    get_pending_sync,
    mark_sync_attempt,
)


logger = logging.getLogger(__name__)


def calculate_backoff(
    attempts: int,
    base_seconds: float = 2.0,
    max_seconds: float = 300.0,
) -> float:
    """
    Exponential backoff:

        2, 4, 8, 16, 32 ... seconds

    capped at max_seconds.
    """

    attempts = max(
        0,
        int(attempts),
    )

    return min(
        max_seconds,
        base_seconds * (
            2 ** attempts
        ),
    )


def process_pending(
    limit: int = 10,
) -> int:
    """
    Process pending sync records.

    The prototype does not require a live remote server.
    Successful transport can be plugged in later.
    """

    rows = get_pending_sync(
        limit=limit
    )

    processed = 0

    for row in rows:

        screening_id = row[
            "screening_id"
        ]

        attempts = int(
            row.get(
                "attempts",
                0,
            )
        )

        try:

            # ------------------------------------------------
            # PLACEHOLDER TRANSPORT
            # ------------------------------------------------
            #
            # Real network upload belongs here.
            #
            # The important property is that the screening
            # already exists locally in SQLite.
            #

            logger.info(
                "Syncing screening %s (%d bytes)",
                screening_id,
                row["payload_bytes"],
            )

            mark_sync_attempt(
                screening_id,
                "SYNCED",
            )

            processed += 1

        except Exception:

            logger.exception(
                "Sync failed for %s",
                screening_id,
            )

            mark_sync_attempt(
                screening_id,
                "FAILED",
            )

            delay = calculate_backoff(
                attempts
            )

            logger.info(
                "Next retry backoff: %.1f seconds",
                delay,
            )

    return processed


def run_worker(
    interval_seconds: int = 30,
    max_attempts: int = 5,
) -> None:
    """
    Continuous background sync loop.

    SQLite remains the source of truth.
    """

    interval_seconds = max(
        1,
        int(interval_seconds),
    )

    while True:

        try:
            process_pending()

        except Exception:
            logger.exception(
                "Sync worker iteration failed."
            )

        time.sleep(
            interval_seconds
        )