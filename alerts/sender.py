from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


class MockSender:
    """
    Safe demo sender.

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
        recipient: str,
        message: str,
    ) -> bool:

        timestamp = (
            datetime.now(
                timezone.utc
            ).astimezone().isoformat()
        )

        line = (
            f"{timestamp} | "
            f"MOCK_ALERT | "
            f"{recipient} | "
            f"{message}\n"
        )

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(line)

        logger.info(
            "Mock alert sent to %s",
            recipient,
        )

        return True


class TwilioSender:
    """
    Production interface placeholder.

    Network integration is deliberately not required
    for the prototype demo.
    """

    def send(
        self,
        recipient: str,
        message: str,
    ) -> bool:
        raise NotImplementedError(
            "Configure Twilio/Gupshup/MSG91 "
            "before enabling live alerts."
        )


def send_alert(
    recipient: str,
    message: str,
    enabled: bool = False,
) -> bool:
    """
    Global alert kill switch.

    alerts.enabled=false means no network activity.
    """

    if not enabled:
        logger.info(
            "Alerts disabled by configuration."
        )
        return False

    sender = MockSender()

    return sender.send(
        recipient,
        message,
    )