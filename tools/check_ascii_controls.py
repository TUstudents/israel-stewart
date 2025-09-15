#!/usr/bin/env python3
"""
Pre-commit hook: fail if files contain ASCII control characters
except TAB (0x09) and LF (0x0A).

Flags bytes in ranges: 0x00-0x08, 0x0B, 0x0C, 0x0D, 0x0E-0x1F, 0x7F.
This also catches ANSI ESC (0x1B) sequences and stray CRs.

Usage: python tools/check_ascii_controls.py <files...>
"""
from __future__ import annotations

import pathlib
import sys


DISALLOWED = set(range(0x00, 0x09)) | {0x0B, 0x0C, 0x0D} | set(range(0x0E, 0x20)) | {0x7F}


def find_controls(text: str) -> list[tuple[int, int, int]]:
    """Return list of (line, col, ord) for disallowed control chars."""
    hits: list[tuple[int, int, int]] = []
    for lineno, line in enumerate(text.splitlines(keepends=True), start=1):
        for col, ch in enumerate(line, start=1):
            o = ord(ch)
            if o in DISALLOWED:
                hits.append((lineno, col, o))
    return hits


def check_file(path: pathlib.Path) -> list[str]:
    issues: list[str] = []
    try:
        data = path.read_bytes()
    except Exception as exc:
        return [f"{path}: error reading file: {exc}"]

    # Only attempt to decode as UTF-8; invalid UTF-8 is handled in a separate hook
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return []  # Let the UTF-8 hook report

    for lineno, col, o in find_controls(text):
        issues.append(f"{path}:{lineno}:{col}: ASCII control char 0x{o:02X} detected")

    return issues


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        return 0
    problems: list[str] = []
    for arg in argv[1:]:
        p = pathlib.Path(arg)
        if not p.exists():
            continue
        problems.extend(check_file(p))

    if problems:
        sys.stderr.write("Disallowed ASCII control characters found:\n")
        for msg in problems:
            sys.stderr.write(msg + "\n")
        sys.stderr.write("Remove control chars; use LF (0x0A) line endings and spaces for indentation.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

