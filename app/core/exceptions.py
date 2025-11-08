"""Custom exception classes for the application."""

from typing import Any


class MXMAPException(Exception):
    """Base exception for MXMAP-X application."""

    def __init__(
        self,
        message: str,
        error_code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MXMAPException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class PredictionError(MXMAPException):
    """Raised when ML prediction fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details,
        )


class NotFoundError(MXMAPException):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            details=details,
        )


class DatabaseError(MXMAPException):
    """Raised when database operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
        )
