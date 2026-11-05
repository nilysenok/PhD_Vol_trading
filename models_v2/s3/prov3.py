"""Шапка таблиц этапа 3: «до аудита», sha кода s3, s2 и этапа 1, sha входов."""
from __future__ import annotations

import hashlib
from pathlib import Path

from models_v2 import prov
from models_v2.s2 import prov2

S3 = Path(__file__).resolve().parent


def code_sha3() -> str:
    h = hashlib.sha256()
    for f in sorted(S3.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def header3(extra: str = "") -> str:
    return ("# разработка; до аудита; оценка 2017–02.2026 по Ж-197 (2022+ — просмотр контроля гл. 3 по SBER); "
            "код models_v2/s3 sha256[:16]=%s, s2=%s, этап 1=%s; sha входов: bars=%s rvi=%s %s\n"
            % (code_sha3(), prov2.code_sha2(), prov.code_sha(), prov.sha_file(prov.BARS)[:16],
               prov.sha_file(prov.RVI)[:16], extra))


def write_csv(df, path, extra: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header3(extra))
        df.to_csv(f, index=False, lineterminator="\n")
