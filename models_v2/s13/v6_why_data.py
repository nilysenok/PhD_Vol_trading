"""V6 лист Z3 (Ж-230): диагностики причин «почему гибриды V6 не обгоняют K2/K3 устойчиво» — данные и общие определения.
Ничего не обучает: ряды — СОХРАНЁННЫЕ прогнозы, те же загрузчики, что в V6 FIN (`v6_fin_data.load`, импорт), 2271 общих дат. Диагностики объявлены до счёта (задание `2026-11-09_harj_v6_Z3.md`) и после просмотра не меняются.
Δ = mean QLIKE(гибрид) − mean QLIKE(база): минус = гибрид лучше; в таблицах — в QLIKE и в процентах уровня QLIKE K1 (владельцу — проценты, не bps)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_io as IO, v6_fin_data as FD

HYB = ["A2", "E2", "C2", "A0g", "A0"]
BOOST = ["K2", "K3"]
OTHER = ["K1", "K8", "M", "V4:NTP2"]
SER = HYB + BOOST + OTHER
PAIRS = [(h, b) for b in BOOST[::-1] for h in HYB]
MAIN = [("A2", "K3"), ("E2", "K3")]
YEARS = [str(y) for y in range(2017, 2027)]
WHY = IO.REP / "why"
DOC = IO.ROOT / "docs/otchety/HARJ_SBER_V6_PRICHINY.md"
CP_FILE = WHY / "CHECKPOINT_WHY.json"
TABLES = ["WHY_YEARS", "WHY_TAILS", "WHY_REGIME", "WHY_MZ", "WHY_CAPTURE", "WHY_ROLL", "WHY_GSIZE"]
CODE_VERSION = "v6.why"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, диагностика, не отбор; первая цифра — потолок; Ж-230, лист Z3 (SBER, h = 1; сохранённые прогнозы, обучения и новых конфигураций нет)"
NOTE = "причины: Δ гибридов к K2/K3 по годам, дням, режимам, калибровке, поправке; 2271 общих дат"


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_why_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def header(note: str = NOTE) -> str:
    return "# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), IO.inputs_hash(), IO.CAVEAT, note)


def write_csv(df: pd.DataFrame, name: str) -> None:
    WHY.mkdir(parents=True, exist_ok=True)
    with open(WHY / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write(header())
        df.to_csv(f, index=False, lineterminator="\n")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(WHY / (name + ".csv"), comment="#")


def load() -> dict:
    """-> dict(dates, tp, rv, f): f — имя ряда -> прогноз (SER) на общих датах; пропусков нет (иначе ошибка)."""
    d = FD.load()
    w = d["w"]
    f = {k: np.asarray(d["fs"][k], float) for k in SER}
    bad = [k for k, v in f.items() if not np.isfinite(v).all() or (v <= 0).any()]
    if bad or not np.isfinite(w["rv"].to_numpy()).all():
        raise ValueError("пропуск или неположительный прогноз в рядах: %s" % bad)
    return dict(dates=w.index, tp=pd.DatetimeIndex(w["t_plus"]), rv=w["rv"].to_numpy(float), f=f)
