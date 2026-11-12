"""S-Z-EVAL (Ж-258; `S_Z_SPEC.md`, разд. 7, 8): константы критериев и загрузка рядов. Константы — ЭТОГО файла (из спецификации), не из данных. Ничего не пишет.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS; механизм выбран после просмотра истории.
Публичное API:
- `THR` = {У1-S: 0,415; У2-S: 0,50; У3-S: 1,00} (%), `KTRIM` = 22, `N_HOLM` = 855, `N_WIDE` = 3415, `M_FAMILY` = 6; `KEY` = "S.Z.eval", `REP`, `CKPT`.
- `FILES`: ряд → parquet; `SHOWN` — 11 рядов первой таблицы; `MCS_SET` — 12 рядов MCS; `REAL_Z` = (S-RCKZ, S-RZ, S-RCKZs).
- `registered_n(ckpt) -> (N, N_wide)`: n, n_wide из СВОЕГО ключа `S.Z.registered` (не максимум по `S.*.registered`: позднейшие регистрации не меняют N оценки; при записи значения те же, что у прежнего максимума — регистрация этапа была последней); `load_inputs(ckpt)`: ряды F и слои X1_R на одних датах D_M (2198 пар);
  расхождение d, rv, cfg8 с окном M или нечисловой прогноз — SystemExit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s18 import r_reg as RR

ROOT, CKPT, REP = RR.ROOT, RR.CKPT, RR.ROOT / "docs/otchety/harj_struct"
KEY, FIRST, LAST, N_D = "S.Z.eval", "2018-01-04", "2026-11-10", 2198
THR, KTRIM, N_HOLM, N_WIDE, M_FAMILY = {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00}, 22, 855, 3415, 6
H = ROOT / "data/regen/harj_struct"
FILES = {"S-RCK": H / "R/rck.parquet", "S-RCKc": H / "R/rckc.parquet", "S-RCKm": H / "D/rckm.parquet", "S-RL1": H / "L/rl1.parquet",
         "S-RCKZ": H / "Z/rckz.parquet", "S-RZ": H / "Z/rz.parquet", "S-RCKZs": H / "Z/rckzs.parquet"}
REAL_Z = ("S-RCKZ", "S-RZ", "S-RCKZs")
SHOWN = ("K1", "K2", "K3", "B_mix", "X1", "W5-G", "S-RCK", "S-RL1") + REAL_Z
MCS_SET = ("S-RCKZ", "S-RZ", "S-RCKZs", "S-RCK", "S-RCKm", "S-RCKc", "S-RL1", "W5-G", "X1", "K2", "K3", "B_mix")
BASES = ("S-RCK", "W5-G", "X1", "K2", "K3")


def registered_n(ckpt: Path = CKPT) -> tuple:
    cp = json.loads(Path(ckpt).read_text(encoding="utf-8"))
    own = cp.get("S.Z.registered")
    if not (isinstance(own, dict) and isinstance(own.get("n"), int) and isinstance(own.get("n_wide"), int)):
        raise SystemExit("нет S.Z.registered с N — оценка отказана")
    return own["n"], own["n_wide"]


def load_inputs(ckpt: Path = CKPT) -> dict:
    x = K.load()
    if not (len(x) == N_D and str(x["t_plus"].min().date()) == FIRST and str(x["t_plus"].max().date()) == LAST):
        raise SystemExit("окно M ≠ D_M (2198 пар, t⁺ %s…%s)" % (FIRST, LAST))
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "B_mix": x["M"], "X1": x["F"], "W5-G": x["W5G"]}
    frames = {}
    for k, p in FILES.items():
        a = pd.read_parquet(p)
        ok = len(a) == N_D and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy()) and np.allclose(a["cfg8"], x["C"], rtol=1e-12, atol=0)
        if not (ok and np.isfinite(a[["F", "X1_R"]].to_numpy()).all()):
            raise SystemExit("%s: даты, rv, cfg8 расходятся с окном M или есть нечисловые прогнозы" % k)
        frames[k], F[k], F["X1_R(%s)" % k] = a, a["F"], a["X1_R"]
    n, nw = registered_n(ckpt)
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), d=pd.DatetimeIndex(x["d"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames=frames, N=n, N_wide=nw)
