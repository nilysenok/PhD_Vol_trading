"""V5 лист FIN, G23: FIN_COST — стоимость вычислений. (1) section=time: время пересчёта каждой стадии, замер по её `--check` (подпроцесс, ТОЛЬКО ЧТЕНИЕ; прогнозы берутся из кэша, обучения нет),
пишется только стадией `FIN` (не `--check`); (2) section=params: число оцениваемых параметров конфигурации — из таблицы `SP_PARAMS` и объявления плана (GS_REPORT, SP_REPORT).
`--check` сверяет section=params целиком и в section=time — состав стадий и rc, но не секунды (замер недетерминирован)."""
from __future__ import annotations

import subprocess
import sys
import time

import pandas as pd

from models_v2.s12 import v5_0_io as IO

NAN = float("nan")
STAGES = [("0", ["models_v2.s12.run_v5", "--stage", "0", "--check"]), ("GS", ["models_v2.s12.run_v5", "--stage", "GS", "--check"]), ("SP", ["models_v2.s12.run_v5", "--stage", "SP", "--check"]),
          ("DX1", ["models_v2.s12.run_v5_dx", "--stage", "DX1", "--check"])]
FREE = {"GS1": 2, "GS2": 2, "GS3": 3, "GS4": 3, "GS5": 3, "SP1": 272}
NOTE = {"GS1": "θ = (ρ, α), β = 0 фиксирована", "GS2": "θ = (ρ, α), β = 0 фиксирована", "GS3": "θ = (ρ, α, β) общие для панели", "GS4": "θ = (ρ, α, β); веса W — три соседа на бумагу из прошлых score (без оценки)",
        "GS5": "θ = (ρ, α, β); λ — первый собственный вектор прошлой корреляции score", "SP1": "МНК по бумаге: 4 регрессии (3 полосы и сумма) × 4 коэффициента (свободный член и 3 лага) × 17 бумаг"}


def timing() -> pd.DataFrame:
    rows = []
    for stage, args in STAGES:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-W", "ignore", "-m"] + args, cwd=IO.ROOT, capture_output=True, text=True)
        rows.append(dict(section="time", stage=stage, command="python3 -W ignore -m " + " ".join(args), seconds=round(time.perf_counter() - t0, 1), rc=r.returncode, n_params=NAN, note="замер по --check, прогнозы из кэша"))
    return pd.DataFrame(rows)


def params() -> pd.DataFrame:
    sp = IO.read_csv("SP_PARAMS.csv")
    sp = sp[sp["section"] == "month"]
    rows = [dict(section="params", stage=g, command="", seconds=NAN, rc=-1, n_params=n, note=NOTE[g]) for g, n in FREE.items()]
    for g in ("SP2", "SP3", "SP4"):
        th = [c for c in sp.columns if c.startswith("th_") and sp.loc[sp["config"] == g, c].notna().any()]
        rows.append(dict(section="params", stage=g, command="", seconds=NAN, rc=-1, n_params=len(th), note="θ общий для панели 17 бумаг, ежемесячная переоценка (число столбцов th_* с оценкой в SP_PARAMS)"))
    return pd.DataFrame(rows)


def table(measure: bool, disk: pd.DataFrame | None = None) -> pd.DataFrame:
    """measure=True (запись) — замер; иначе строки времени берутся с диска (их секунды не пересчитываются)."""
    t = timing() if measure else disk[disk["section"] == "time"].reset_index(drop=True)
    return pd.concat([t, params()], ignore_index=True)
