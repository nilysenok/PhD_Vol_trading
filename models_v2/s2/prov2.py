"""Шапка таблиц этапа 2: статус «до аудита», sha кода этапа 2 и входов."""
from __future__ import annotations

import hashlib
from pathlib import Path

from models_v2 import prov

S2 = Path(__file__).resolve().parent


def code_sha2() -> str:
    h = hashlib.sha256()
    for f in sorted(S2.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def header2(extra: str = "") -> str:
    return ("# разработка; до аудита; оценка 2017–02.2026 по Ж-197 (2022+ — просмотр контроля гл. 3 по SBER); "
            "код models_v2/s2 sha256[:16]=%s, этап 1 sha256[:16]=%s; sha входов: bars=%s rvi=%s %s\n"
            % (code_sha2(), prov.code_sha(), prov.sha_file(prov.BARS)[:16], prov.sha_file(prov.RVI)[:16], extra))


def write_csv(df, path, extra: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header2(extra))
        df.to_csv(f, index=False, lineterminator="\n")
