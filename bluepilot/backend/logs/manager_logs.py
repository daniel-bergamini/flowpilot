#!/usr/bin/env python3
"""
Manager log utilities.

Provides helpers to read and format manager log lines from swaglog files
and realtime messaging streams.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

from system.swaglog import SWAGLOG_DIR

logger = logging.getLogger(__name__)

TYPE_SUFFIX_SEPARATOR = '$'
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# ANSI color codes for log levels
ANSI_COLORS = {
    'DEBUG': '\x1b[36m',      # Cyan
    'INFO': '\x1b[32m',       # Green
    'WARNING': '\x1b[33m',    # Yellow
    'WARN': '\x1b[33m',       # Yellow
    'ERROR': '\x1b[31m',      # Red
    'CRITICAL': '\x1b[35m',   # Magenta
    'FATAL': '\x1b[35m',      # Magenta
}
ANSI_RESET = '\x1b[0m'
ANSI_DIM = '\x1b[2m'
ANSI_BOLD = '\x1b[1m'


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    if not text:
        return text
    return ANSI_ESCAPE_PATTERN.sub('', text)


def _get_field(data: Optional[Dict], key: str):
    """
    Retrieve a key from a swaglog record, handling type suffixes
    (e.g. msg$s, level$i). Returns None if not present.
    """
    if not isinstance(data, dict):
        return None

    if key in data:
        return data[key]

    prefix = f"{key}{TYPE_SUFFIX_SEPARATOR}"
    for field, value in data.items():
        if field.startswith(prefix):
            return value
    return None


def _format_timestamp(value: Optional[float]) -> str:
    """Format a unix timestamp into a human readable string."""
    try:
        if value:
            dt = datetime.fromtimestamp(float(value))
        else:
            dt = datetime.now()
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    except Exception:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def _format_tmux_line(line: str) -> str:
    """Add a timestamp prefix to a tmux log line."""
    timestamp = _format_timestamp(time.time())
    return f"{ANSI_DIM}{timestamp}{ANSI_RESET} {line}".rstrip()


def _stringify_message(message, preserve_ansi: bool = True) -> str:
    """Return a compact string representation of a log message payload."""
    if message is None:
        return ''

    if isinstance(message, (dict, list)):
        try:
            cleaned = json.dumps(message, separators=(',', ':'))
        except Exception:
            cleaned = str(message)
    else:
        cleaned = str(message)

    # Optionally strip ANSI codes (default: preserve them for web display)
    if not preserve_ansi:
        return _strip_ansi(cleaned)
    return cleaned


def parse_manager_log_line(record: str, colorize: bool = True) -> Optional[str]:
    """
    Parse a single swaglog JSON entry (from messaging or file) and return
    a formatted string if it belongs to the manager daemon.

    Args:
        record: JSON log record string
        colorize: If True, add ANSI color codes based on log level
    """
    try:
        data = json.loads(record)
    except json.JSONDecodeError:
        logger.debug("Failed to parse log record: %s", record[:120])
        return None

    ctx = _get_field(data, 'ctx') or {}
    daemon = _get_field(ctx, 'daemon') or _get_field(ctx, 'daemon_name')
    if daemon != 'manager':
        return None

    created = _get_field(data, 'created')
    level = (_get_field(data, 'level') or 'INFO').upper()
    module = _get_field(data, 'module') or _get_field(data, 'name') or 'manager'
    msg = _stringify_message(_get_field(data, 'msg'))

    timestamp = _format_timestamp(created)

    if colorize:
        # Add ANSI colors based on log level
        level_color = ANSI_COLORS.get(level, '')
        # Format: dim timestamp, colored [LEVEL], normal module: message
        return f"{ANSI_DIM}{timestamp}{ANSI_RESET} {level_color}[{level}]{ANSI_RESET} {module}: {msg}".strip()

    return f"{timestamp} [{level}] {module}: {msg}".strip()


def read_recent_manager_logs(max_lines: int = 1000, max_files: int = 25) -> Tuple[bool, str]:
    """
    Read recent manager log lines from rotating swaglog files.

    Args:
        max_lines: limit number of lines returned
        max_files: number of latest swaglog files to scan

    Returns:
        Tuple[success, output_or_error]
    """
    log_dir = SWAGLOG_DIR
    if not os.path.isdir(log_dir):
        return False, f"Log directory not found: {log_dir}"

    files = [
        os.path.join(log_dir, entry)
        for entry in os.listdir(log_dir)
        if entry.startswith('swaglog.')
    ]

    if not files:
        return False, "No swaglog files found"

    files.sort()
    files = files[-max_files:]

    lines: Deque[str] = deque(maxlen=max_lines)

    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    formatted = parse_manager_log_line(raw_line)
                    if formatted:
                        lines.append(formatted)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.debug("Failed reading %s: %s", path, exc)

    if not lines:
        return False, "No manager log entries found"

    return True, '\n'.join(lines)


def _find_tmux_target() -> Optional[str]:
    """Return a tmux target pane if no current client is available."""
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#S:#I.#P"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    panes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return panes[0] if panes else None


def read_tmux_logs(
    max_lines: int = 2000,
    target: Optional[str] = None,
    with_timestamps: bool = False,
) -> Tuple[bool, str]:
    """Read recent output from a tmux pane.

    Args:
        max_lines: number of lines to capture from scrollback
        target: optional tmux target (session:window.pane)
        with_timestamps: prefix lines with timestamps
    """
    cmd = ["tmux", "capture-pane", "-pS", f"-{max_lines}"]
    if target:
        cmd.extend(["-t", target])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
    except FileNotFoundError:
        return False, "tmux not available"
    except subprocess.TimeoutExpired:
        return False, "tmux capture timed out"

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        if not target and err and "no current client" in err.lower():
            fallback_target = _find_tmux_target()
            if fallback_target:
                return read_tmux_logs(
                    max_lines=max_lines,
                    target=fallback_target,
                    with_timestamps=with_timestamps,
                )
        return False, err or "tmux capture failed"

    output = result.stdout.strip()
    if not output:
        return False, "tmux capture empty"
    if with_timestamps:
        lines = [_format_tmux_line(line) for line in output.splitlines() if line.strip()]
        return True, "\n".join(lines)
    return True, output
