from __future__ import annotations


class AppError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMUnavailableError(AppError):
    code = "llm_unavailable"
    http_status = 502


class LLMInvalidOutputError(AppError):
    code = "llm_invalid_output"
    http_status = 502


class TokenBudgetExceededError(AppError):
    code = "token_budget_exceeded"
    http_status = 429


class RequestTimeoutError(AppError):
    code = "request_timeout"
    http_status = 504


class SQLValidationError(AppError):
    code = "sql_rejected"
    http_status = 400
