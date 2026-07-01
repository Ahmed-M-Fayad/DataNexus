"""
src/cli/formatting.py
=====================
All terminal display helpers for the DataNexus CLI.

Nothing here touches the database or engine — it's purely
about turning Python dicts/objects into readable terminal output.

No third-party dependencies (no rich, no tabulate) — only stdlib
so this works in any environment the project runs in.
"""

import shutil
import re


# ── Connection strings ─────────────────────────────────────────────────────────

def mask_connection_string(conn: str) -> str:
    """
    Redact the password portion of a SQLAlchemy-style connection string
    before it's ever printed to a terminal (e.g. during a live demo).

    'postgresql://user:secret@host:5432/db' -> 'postgresql://user:***@host:5432/db'
    Non-DB-URL strings (file paths, etc.) are returned unchanged.
    """
    if not conn:
        return conn
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", conn)


# ── Terminal width ─────────────────────────────────────────────────────────────

def _term_width() -> int:
    """Return usable terminal width, clamped to [60, 120]."""
    return max(60, min(120, shutil.get_terminal_size(fallback=(100, 24)).columns))


# ── Status / severity labels ───────────────────────────────────────────────────

_STATUS_ICONS = {
    "pass":    "✓ PASS",
    "fail":    "✗ FAIL",
    "error":   "⚠ ERROR",
    "skip":    "- SKIP",
    "pending": "○ PENDING",
    "running": "↻ RUNNING",
}

_SEVERITY_ICONS = {
    "critical": "[CRITICAL]",
    "high":     "[HIGH]    ",
    "medium":   "[MEDIUM]  ",
    "low":      "[LOW]     ",
}


def status_label(status: str) -> str:
    """Return a short icon+text label for a run/check status."""
    return _STATUS_ICONS.get(str(status).lower(), str(status).upper())


def severity_label(severity: str) -> str:
    """Return a fixed-width severity label."""
    return _SEVERITY_ICONS.get(str(severity).lower(), f"[{severity.upper():<8}]")


# ── Score bar ──────────────────────────────────────────────────────────────────

def score_bar(score: float, width: int = 24) -> str:
    """
    Render a score (0–100) as a compact progress bar.

    Example:
        score_bar(87.5)  →  '[████████████████████░░░░]  87.50'
    """
    if score is None:
        return "[" + "?" * width + "]  N/A"
    filled = int(round((score / 100.0) * width))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]  {score:.2f} / 100"


# ── Simple ASCII table ─────────────────────────────────────────────────────────

def _col_widths(headers: list, rows: list) -> list:
    """Compute per-column widths: max of header length and all cell lengths."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    return widths


def print_table(headers: list, rows: list, title: str = None) -> None:
    """
    Print a plain ASCII table to stdout.

    Args:
        headers: List of column header strings.
        rows:    List of rows; each row is a list of values (will be str()'d).
        title:   Optional title printed above the table.
    """
    if not rows:
        if title:
            print(f"\n  {title}")
        print("  (no rows)\n")
        return

    str_rows = [[str(cell) for cell in row] for row in rows]
    widths   = _col_widths(headers, str_rows)

    sep   = "  " + "  ".join("-" * w for w in widths)
    hdr   = "  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    lines = ["  " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths))
             for row in str_rows]

    if title:
        print(f"\n  {title}")
    print(sep)
    print(hdr)
    print(sep)
    for line in lines:
        print(line)
    print(sep)


# ── Section dividers ───────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    """Print a titled section divider."""
    width = _term_width()
    print()
    print(f"  ── {title} " + "─" * max(0, width - len(title) - 7))


def print_banner(text: str) -> None:
    """Print a simple single-line banner box."""
    width = max(len(text) + 4, 40)
    print()
    print("  ╔" + "═" * width + "╗")
    print("  ║  " + text.ljust(width - 2) + "  ║")
    print("  ╚" + "═" * width + "╝")


# ── Composite cards ────────────────────────────────────────────────────────────

def print_run_card(run_row: dict) -> None:
    """
    Print a compact summary card for one ValidationRun.

    Expected keys in run_row:
        id, status, quality_score, triggered_by,
        started_at, finished_at, config_name (optional)
    """
    print_section(f"Validation Run  #{ run_row.get('id', '?') }")

    status = str(run_row.get("status", "unknown"))
    score  = run_row.get("quality_score")

    print(f"  Status     : {status_label(status)}")
    print(f"  Score      : {score_bar(score)}")

    config_name = run_row.get("config_name")
    config_id   = run_row.get("config_id")
    if config_name:
        print(f"  Config     : {config_name}  (id={config_id})")
    elif config_id:
        print(f"  Config id  : {config_id}")

    triggered = run_row.get("triggered_by", "—")
    print(f"  Triggered  : {triggered}")

    started  = run_row.get("started_at")
    finished = run_row.get("finished_at")
    if started and finished:
        try:
            duration = (finished - started).total_seconds()
            print(f"  Duration   : {duration:.2f}s")
        except Exception:
            pass
    if started:
        print(f"  Started    : {started}")
    if run_row.get("error_message"):
        print(f"  Error      : {run_row['error_message'][:120]}")
    print()
