#!/usr/bin/env python3
"""Проверка описания PR: строки `Blok: Ж-NNN` и `Punkt: <документ>#<раздел>`.

Текст — из файла (аргумент) или stdin. Код 0 — есть обе; 2 — нет. Вход для хука на
`gh pr create` / `gh pr merge` (Э3).
"""
import re
import sys

BLOK_RE = re.compile(r"^[ \t]*Blok:[ \t]*Ж-\d{3}(?!\d)", re.M)
PUNKT_RE = re.compile(r"^[ \t]*Punkt:[ \t]*[^\s#]+#\S+", re.M)


def oshibki(text):
    errs = []
    if not BLOK_RE.search(text):
        errs.append("нет строки `Blok: Ж-NNN`")
    if not PUNKT_RE.search(text):
        errs.append("нет строки `Punkt: <документ>#<раздел>`")
    return errs


def main(argv):
    text = open(argv[1], encoding="utf-8").read() if len(argv) > 1 else sys.stdin.read()
    errs = oshibki(text)
    for e in errs:
        print("Ж-025 п. 1.4: " + e, file=sys.stderr)
    return 2 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
