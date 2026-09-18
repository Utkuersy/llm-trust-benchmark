"""The layer for running external tools (pylint, bandit, pip-audit, pytest, docker).

Defensive programming rules:
    * ``shell=True`` is never used (a command-injection surface).
    * Every call has a timeout; the process tree is killed on timeout.
    * stdout/stderr are kept bounded (to avoid unbounded memory growth).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # nosec B404 - commands are built from fixed lists, shell is never used
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_OUTPUT = 200_000


@dataclass
class CommandResult:
    """The result of an external command."""

    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_sec: float = 0.0
    error: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Did the command finish without error and without timing out?"""
        return self.returncode == 0 and not self.timed_out and not self.error


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} characters truncated]"


def run_command(
    command: list[str],
    *,
    cwd: Path | str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> CommandResult:
    """Runs the command and returns the result in a structured form.

    Args:
        command: The argument list (``shell=True`` is never used).
        cwd: The working directory.
        timeout: The upper bound, in seconds.
        env: Additional/override environment variables (merged with the current environment).
        max_output: The character limit for stdout/stderr.
        allowed_returncodes: Codes outside this set are logged as a warning.
    """
    if not command:
        return CommandResult(command=[], returncode=-1, error="empty command")

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.perf_counter()
    try:
        completed = subprocess.run(  # nosec B603
            command,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        logger.warning(
            "command timed out",
            extra={"command": command[:4], "timeout_sec": timeout},
        )
        return CommandResult(
            command=command,
            # SIGKILL is POSIX-specific; it doesn't exist on Windows (AttributeError risk).
            returncode=-getattr(signal, "SIGKILL", signal.SIGTERM),
            stdout=_truncate(exc.stdout or "" if isinstance(exc.stdout, str) else "", max_output),
            stderr=_truncate(exc.stderr or "" if isinstance(exc.stderr, str) else "", max_output),
            timed_out=True,
            duration_sec=duration,
            error=f"timeout after {timeout}s",
        )
    except (OSError, ValueError) as exc:
        duration = time.perf_counter() - start
        logger.error("command could not be run", extra={"command": command[:4], "error": str(exc)})
        return CommandResult(
            command=command, returncode=-1, duration_sec=duration, error=str(exc)
        )

    duration = time.perf_counter() - start
    result = CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=_truncate(completed.stdout or "", max_output),
        stderr=_truncate(completed.stderr or "", max_output),
        duration_sec=duration,
    )
    if completed.returncode not in allowed_returncodes:
        logger.debug(
            "command returned a non-zero code",
            extra={"command": command[:4], "returncode": completed.returncode},
        )
    return result


def python_module_command(module: str, *args: str) -> list[str]:
    """Builds a ``python -m <module> ...`` command using the current interpreter."""
    return [sys.executable, "-m", module, *args]


def tool_available(module: str) -> bool:
    """Checks whether a tool can be invoked as ``python -m <module>``."""
    probe = run_command(python_module_command(module, "--version"), timeout=60)
    return probe.returncode == 0 or "usage" in (probe.stdout + probe.stderr).lower()


def binary_available(name: str) -> bool:
    """Is an executable available on PATH?"""
    return shutil.which(name) is not None
