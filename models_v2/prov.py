"""Происхождение таблиц: sha256 входов и кода, шапка О11."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data/iss_check/bars_10m_full/SBER.parquet"
RVI = ROOT / "data/fastbt/s9/rvi_1850.parquet"


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "models_v2").glob("*.py")):
        if f.name.startswith("run_"):
            continue
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def header(extra: str = "", status: str = "после аудита этапа 1 проверяющим") -> str:
    return ("# разработка; %s; оценка 2017–02.2026 по Ж-197 (2022+ — просмотр контроля гл. 3 по SBER); " % status +
            "код models_v2 sha256[:16]=%s; sha входов: bars=%s rvi=%s %s\n"
            % (code_sha(), sha_file(BARS)[:16], sha_file(RVI)[:16], extra))


def write_csv(df, path, extra: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header(extra))
        df.to_csv(f, index=False, lineterminator="\n")
