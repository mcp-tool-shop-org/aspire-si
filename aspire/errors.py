"""
ASPIRE's error shape.

Every error ASPIRE raises on purpose carries a stable ``code``, a one-line ``message``, a
``hint`` that says what to do, the ``cause`` when it wraps another error, and whether trying
again could help (``retryable``). The CLI prints these without a traceback and exits with the
error's ``exit_code``: 1 for something the user can fix (a missing key, a bad file), 2 for a
failure while running.
"""

from __future__ import annotations

from typing import Any

USER_ERROR = 1
RUNTIME_ERROR = 2


class AspireError(Exception):
    """An error with a code, a message, and a hint."""

    code = "ASPIRE_ERROR"
    exit_code = RUNTIME_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
        retryable: bool = False,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).code
        self.hint = hint
        self.cause = cause
        self.retryable = retryable
        if exit_code is not None:
            self.exit_code = exit_code

    def __str__(self) -> str:
        return f"{self.message}\n\n{self.hint}" if self.hint else self.message

    def to_dict(self) -> dict[str, Any]:
        shape: dict[str, Any] = {"code": self.code, "message": self.message, "retryable": self.retryable}
        if self.hint:
            shape["hint"] = self.hint
        if self.cause is not None:
            shape["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return shape


class ConfigError(AspireError):
    """A config, prompts or checkpoint file that is missing or cannot be read."""

    code = "ASPIRE_CONFIG"
    exit_code = USER_ERROR


def missing_api_key(variable: str, teacher: str, keys_url: str) -> dict[str, Any]:
    """The arguments for a missing-key error, shared by the API teachers."""
    return {
        "code": "ASPIRE_MISSING_API_KEY",
        "exit_code": USER_ERROR,
        "hint": (
            "To fix this, set your API key:\n"
            f"  Windows:   set {variable}=your-key-here\n"
            f"  Linux/Mac: export {variable}=your-key-here\n\n"
            f"Or pass it directly: {teacher}(api_key='your-key')\n\n"
            f"Get your API key at: {keys_url}"
        ),
    }
