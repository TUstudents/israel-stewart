#!/usr/bin/env python3
"""
Pre-commit hook: fail if files contain the Unicode replacement character (U+FFFD)
or are not valid UTF-8.

Usage (pre-commit passes filenames):
    python tools/check_replacement_chars.py file1 file2 ...
"""
from __future__ import annotations

import pathlib
import sys

REPLACEMENT_CHAR = "\uFFFD"


def check_file(path: pathlib.Path) -> list[str]:
    issues: list[str] = []
    try:
        data = path.read_bytes()
    except Exception as exc:
        issues.append(f"{path}: error reading file: {exc}")
        return issues

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        issues.append(f"{path}: not valid UTF-8 ({exc})")
        return issues

    if REPLACEMENT_CHAR in text:
        # Report each line containing U+FFFD
        for lineno, line in enumerate(text.splitlines(), start=1):
            if REPLACEMENT_CHAR in line:
                issues.append(f"{path}:{lineno}: contains U+FFFD (�)")
    return issues


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        # pre-commit may invoke with no files when none match; treat as success
        return 0
    problems: list[str] = []
    for arg in argv[1:]:
        p = pathlib.Path(arg)
        if not p.exists():
            # If file was deleted in the change set, skip
            continue
        problems.extend(check_file(p))

    if problems:
        sys.stderr.write("Unicode replacement characters or invalid UTF-8 detected:\n")
        for msg in problems:
            sys.stderr.write(msg + "\n")
        sys.stderr.write("Fix encoding to UTF-8 and remove U+FFFD occurrences.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

