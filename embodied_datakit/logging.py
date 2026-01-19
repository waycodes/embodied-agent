"""Structured logging utilities for EmbodiedDataKit."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler


@dataclass
class LogContext:
    """Context for structured logging."""

    dataset_id: str | None = None
    episode_id: str | None = None
    stage: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def with_episode(self, episode_id: str) -> "LogContext":
        """Create new context with episode ID."""
        return LogContext(
            dataset_id=self.dataset_id,
            episode_id=episode_id,
            stage=self.stage,
            extra=self.extra.copy(),
        )

    def with_stage(self, stage: str) -> "LogContext":
        """Create new context with stage."""
        return LogContext(
            dataset_id=self.dataset_id,
            episode_id=self.episode_id,
            stage=stage,
            extra=self.extra.copy(),
        )


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if present
        if hasattr(record, "ctx") and record.ctx:
            ctx: LogContext = record.ctx
            if ctx.dataset_id:
                log_entry["dataset_id"] = ctx.dataset_id
            if ctx.episode_id:
                log_entry["episode_id"] = ctx.episode_id
            if ctx.stage:
                log_entry["stage"] = ctx.stage
            if ctx.extra:
                log_entry.update(ctx.extra)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class EDKLogger:
    """Structured logger for EmbodiedDataKit."""

    def __init__(
        self,
        name: str = "edk",
        level: int = logging.INFO,
        format_type: Literal["text", "json"] = "text",
        log_file: Path | str | None = None,
    ) -> None:
        """Initialize logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        self.context = LogContext()
        self._timers: dict[str, float] = {}

        # Console handler
        if format_type == "json":
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
            )
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)

    def set_context(self, ctx: LogContext) -> None:
        """Set logging context."""
        self.context = ctx

    def _log(
        self,
        level: int,
        msg: str,
        ctx: LogContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Log with context."""
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "",
            0,
            msg,
            (),
            None,
        )
        record.ctx = ctx or self.context
        record.extra_fields = kwargs
        self.logger.handle(record)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, **kwargs)

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop timer and return elapsed time."""
        if name not in self._timers:
            return 0.0
        elapsed = time.perf_counter() - self._timers.pop(name)
        return elapsed

    def log_timing(self, name: str, msg: str | None = None) -> None:
        """Stop timer and log elapsed time."""
        elapsed = self.stop_timer(name)
        message = msg or f"{name} completed"
        self.info(message, latency_ms=round(elapsed * 1000, 2))


# Global logger instance
_logger: EDKLogger | None = None


def get_logger() -> EDKLogger:
    """Get or create global logger."""
    global _logger
    if _logger is None:
        _logger = EDKLogger()
    return _logger


def configure_logging(
    level: int = logging.INFO,
    format_type: Literal["text", "json"] = "text",
    log_file: Path | str | None = None,
) -> EDKLogger:
    """Configure global logger."""
    global _logger
    _logger = EDKLogger(level=level, format_type=format_type, log_file=log_file)
    return _logger
