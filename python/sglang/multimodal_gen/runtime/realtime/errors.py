# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class RealtimeProtocolError(ValueError):
    """Caller-visible realtime protocol error."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = {
            key: value for key, value in details.items() if value is not None
        }
