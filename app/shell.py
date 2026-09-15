"""Sandboxed shell exec (pattern ported from nanobot's ExecTool).

Commands run as the container user with the workspace as the default
working directory. Catastrophic commands are denied up front and the app's
own state dirs (/app, /data, /models) are read-only. Output is truncated
head+tail so it stays useful as LLM context.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess

from app import config

MAX_TIMEOUT = config.EXEC_MAX_TIMEOUT
PROTECTED_DIRS = ("/app", "/data", "/models")

_DENY = [
    (
        re.compile(r"\brm\s+(-\w+\s+)*-\w*[rR]\w*(\s+-\w+)*\s+(/|~|\$HOME)(?=[\s/;|&)]|$)"),
        "recursive remove of root or home",
    ),
    (re.compile(r"--no-preserve-root"), "rm --no-preserve-root"),
    (
        re.compile(r"\brm\s+(-\w+\s+)*-\w*[rR]\w*\s+(-\w+\s+)*(~?/\*|\$HOME/\*)"),
        "recursive remove of a top-level glob",
    ),
    (re.compile(r"\bmkfs(\.\w+|\s|\b)"), "filesystem format"),
    (re.compile(r"\bdd\s+if="), "dd block copy"),
    (re.compile(r">\s*/dev/(sd|nvme|mmcblk|hd|xvd)\w*"), "write to a block device"),
    (re.compile(r"\b(shutdown|reboot|poweroff|halt)\b"), "power control"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"), "fork bomb"),
]

_WRITE_VERB = re.compile(
    r"\b(rm|mv|cp|tee|dd|sed|truncate|shred|unlink|touch|mkdir|ln|chmod|chown)\b"
    r"|(?<![&\w./])>(?!&)"
)
_PROTECTED_REF = re.compile(r"(?:^|[\s\"'=:(;|&])(/app|/data|/models)(?:/|\b)")


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


def _denied(command: str) -> str:
    for pattern, why in _DENY:
        if pattern.search(command):
            return f"error: command denied ({why}); use a safer alternative"
    if _WRITE_VERB.search(command) and _PROTECTED_REF.search(command):
        return (
            "error: /app, /data and /models are read-only; "
            "work inside the workspace instead"
        )
    return ""


def _truncate(text: str) -> str:
    cap = max(int(config.EXEC_MAX_OUTPUT), 1000)
    if len(text) <= cap:
        return text
    half = cap // 2
    return (
        text[:half]
        + f"\n\n... ({len(text) - cap:,} chars truncated) ...\n\n"
        + text[-half:]
    )


def run_shell(command: str, working_dir: str = None, timeout: int = None) -> str:
    command = (command or "").strip()
    if not command:
        return "error: missing command"
    denied = _denied(command)
    if denied:
        return denied

    base = os.path.realpath(config.WORK_DIR)
    cwd = base
    if working_dir:
        candidate = os.path.realpath(os.path.join(base, working_dir))
        if candidate != base and not candidate.startswith(base + os.sep):
            return "error: working_dir is outside the workspace"
        cwd = candidate

    try:
        t = min(max(int(timeout or config.EXEC_TIMEOUT), 1), MAX_TIMEOUT)
    except (TypeError, ValueError):
        t = config.EXEC_TIMEOUT

    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=t)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - already killed
            pass
        return f"error: command timed out after {t}s"

    parts = []
    if stdout and stdout.strip("\n"):
        parts.append(stdout.strip("\n"))
    if stderr and stderr.strip("\n"):
        parts.append("STDERR:\n" + stderr.strip("\n"))
    parts.append(f"Exit code: {proc.returncode}")
    result = "\n".join(parts) if parts else f"(no output)\nExit code: {proc.returncode}"
    return _truncate(result)
