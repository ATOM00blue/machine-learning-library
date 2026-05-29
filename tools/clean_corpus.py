#!/usr/bin/env python3
"""Normalize corpus markdown files to clean UTF-8 / LF text.

Removes extraction artifacts left by PDF->text conversion WITHOUT touching
meaning: CRLF/CR line endings -> LF, and C0 control characters (form feeds,
ETX, device-control codes, etc.) except TAB and LF. Multibyte UTF-8 (em
dashes, bullets, math symbols) is preserved untouched.

Idempotent: running twice changes nothing the second time. Safe because every
byte removed is non-printable junk; the file's visible text is unchanged.
"""
from __future__ import annotations
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")

# C0 controls to delete: 0x00-0x1F and 0x7F, EXCEPT tab(0x09) and LF(0x0A).
_DELETE = bytes(b for b in range(0x20) if b not in (0x09, 0x0A)) + bytes([0x7F])
_DELETE_TABLE = {b: None for b in _DELETE}


def clean_bytes(data: bytes) -> bytes:
    # Line endings first: CRLF -> LF, lone CR -> LF.
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    # Drop remaining C0 controls (CR already gone) + DEL.
    return data.translate(None, _DELETE)


def main() -> int:
    files = sorted(glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True))
    changed = 0
    bytes_removed = 0
    for path in files:
        original = open(path, "rb").read()
        cleaned = clean_bytes(original)
        if cleaned != original:
            # sanity: result must still decode as UTF-8
            cleaned.decode("utf-8")
            open(path, "wb").write(cleaned)
            changed += 1
            bytes_removed += len(original) - len(cleaned)
    print(f"scanned {len(files)} files; cleaned {changed}; removed {bytes_removed} junk bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
