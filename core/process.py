"""Harici araçları (pylint, bandit, pip-audit, pytest, docker) çalıştırma katmanı.

Savunmacı programlama kuralları:
    * ``shell=True`` asla kullanılmaz (komut enjeksiyonu yüzeyi).
    * Her çağrının bir timeout'u vardır; süre aşımında süreç ağacı öldürülür.
    * stdout/stderr sınırlı boyutta tutulur (bellek şişmesini önlemek için).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # nosec B404 - komutlar sabit listelerden kurulur, shell kullanılmaz
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_OUTPUT = 200_000


@dataclass
class CommandResult:
    """Bir dış komutun sonucu."""

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
        """Komut hatasız ve süre aşımı olmadan bitti mi?"""
        return self.returncode == 0 and not self.timed_out and not self.error


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} karakter kirpildi]"


def run_command(
    command: list[str],
    *,
    cwd: Path | str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> CommandResult:
    """Komutu çalıştırır ve sonucu yapılandırılmış biçimde döndürür.

    Args:
        command: Argüman listesi (``shell=True`` kullanılmaz).
        cwd: Çalışma dizini.
        timeout: Saniye cinsinden üst sınır.
        env: Ek/override ortam değişkenleri (mevcut ortamla birleştirilir).
        max_output: stdout/stderr için karakter sınırı.
        allowed_returncodes: Bunlar dışındaki kodlar uyarı olarak loglanır.
    """
    if not command:
        return CommandResult(command=[], returncode=-1, error="bos komut")

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603 # nosec B603
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
            "komut zaman asimina ugradi",
            extra={"command": command[:4], "timeout_sec": timeout},
        )
        return CommandResult(
            command=command,
            returncode=-signal.SIGKILL,
            stdout=_truncate(exc.stdout or "" if isinstance(exc.stdout, str) else "", max_output),
            stderr=_truncate(exc.stderr or "" if isinstance(exc.stderr, str) else "", max_output),
            timed_out=True,
            duration_sec=duration,
            error=f"timeout after {timeout}s",
        )
    except (OSError, ValueError) as exc:
        duration = time.perf_counter() - start
        logger.error("komut calistirilamadi", extra={"command": command[:4], "error": str(exc)})
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
            "komut sifir olmayan kod dondurdu",
            extra={"command": command[:4], "returncode": completed.returncode},
        )
    return result


def python_module_command(module: str, *args: str) -> list[str]:
    """``python -m <module> ...`` komutunu mevcut yorumlayıcı ile kurar."""
    return [sys.executable, "-m", module, *args]


def tool_available(module: str) -> bool:
    """Bir aracın ``python -m <module>`` olarak çağrılabilir olduğunu kontrol eder."""
    probe = run_command(python_module_command(module, "--version"), timeout=60)
    return probe.returncode == 0 or "usage" in (probe.stdout + probe.stderr).lower()


def binary_available(name: str) -> bool:
    """PATH üzerinde bir çalıştırılabilir var mı?"""
    return shutil.which(name) is not None
