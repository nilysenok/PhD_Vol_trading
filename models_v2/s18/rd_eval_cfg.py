"""S-RCD-EVAL (Ж-259; `S_RCD_SPEC.md`, разд. 6–8): константы критериев и загрузка рядов. Константы — ЭТОГО файла (из спецификации), не из данных. Ничего не пишет.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS; гипотеза выбрана после просмотра результатов S-R, S-D, S-L, S-Z.
Публичное API:
- `THR` = {У1-S: 0,415; У2-S: 0,50; У3-S: 1,00} (%), `KTRIM` = 22, `N_HOLM` = 858, `N_WIDE` = 3421, `M_FAMILY` = 6, `SHARE_HALF` = 0,5, `ACTIVE_FROM` = 2020-08-03; `KEY` = "S.RCD.eval".
- `SERIES` — ряд → (parquet, колонка F, колонка слоя X1); `SHOWN` — 14 рядов первой таблицы; `MCS_SET` — 15 рядов; `LAYERS` — слои X1 (имя → (ряд, колонка)); `BASES`.
- `registered_n(ckpt) -> (N, N_wide)`: n, n_wide из СВОЕГО обязательного ключа `S.RCD.registered` (не максимум по `S.*.registered`; при записи значения те же, что у прежнего максимума — регистрация этапа была последней); `load_inputs(ckpt)`: F и слои на одних датах D_M (2198 пар); расхождение d, rv, cfg8 с окном M или нечисловой прогноз — SystemExit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s18 import r_reg as RR

ROOT, CKPT, REP = RR.ROOT, RR.CKPT, RR.ROOT / "docs/otchety/harj_struct"
KEY, FIRST, LAST, N_D = "S.RCD.eval", "2018-01-04", "2026-11-10", 2198
THR, KTRIM, N_HOLM, N_WIDE, M_FAMILY, SHARE_HALF = {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00}, 22, 858, 3421, 6, 0.5
ACTIVE_FROM = "2020-08-03"
H = ROOT / "data/regen/harj_struct"
SERIES = {"S-RCD": (H / "RD/rcd.parquet", "F", "X1_R"), "S-CDF": (H / "RD/cdf.parquet", "F", "X1_R"), "S-CKF": (H / "RD/ckf.parquet", "F", "X1_R"),
          "S-CD": (H / "CM/cd.parquet", "F", "X1_V"), "S-CK": (H / "CM/ck.parquet", "F", "X1_V"), "S-RCK": (H / "R/rck.parquet", "F", "X1_R"),
          "S-RCKf": (H / "D/rckf.parquet", "F", "X1_R"), "S-RCKb": (H / "R/rckb.parquet", "F", "X1_R"), "S-RCKZ": (H / "Z/rckz.parquet", "F", "X1_R"), "S-RL1": (H / "L/rl1.parquet", "F", "X1_R")}
LAYERS = {"X1_R(S-RCD)": "S-RCD", "X1_V(S-CDF)": "S-CDF", "X1_V(S-CKF)": "S-CKF", "X1_R(S-RCK)": "S-RCK"}
SHOWN = ("K1", "K2", "K3", "B_mix", "X1", "W5-G", "S-CD", "S-CK", "S-CDF", "S-CKF", "S-RCK", "S-RCD", "S-RCKZ", "S-RL1")
MCS_SET = ("S-RCD", "S-CDF", "S-CKF", "S-CD", "S-CK", "S-RCK", "S-RCKf", "S-RCKb", "S-RCKZ", "S-RL1", "W5-G", "X1", "K2", "K3", "B_mix")
BASES = ("S-CDF", "S-RCK", "W5-G", "X1", "K2", "K3")


def registered_n(ckpt: Path = CKPT) -> tuple:
    cp = json.loads(Path(ckpt).read_text(encoding="utf-8"))
    own = cp.get("S.RCD.registered")
    if not (isinstance(own, dict) and isinstance(own.get("n"), int) and isinstance(own.get("n_wide"), int)):
        raise SystemExit("нет S.RCD.registered с N — оценка отказана")
    return own["n"], own["n_wide"]


def load_inputs(ckpt: Path = CKPT) -> dict:
    x = K.load()
    if not (len(x) == N_D and str(x["t_plus"].min().date()) == FIRST and str(x["t_plus"].max().date()) == LAST):
        raise SystemExit("окно M ≠ D_M (2198 пар, t⁺ %s…%s)" % (FIRST, LAST))
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "B_mix": x["M"], "X1": x["F"], "W5-G": x["W5G"]}
    frames = {}
    for k, (p, fc, lc) in SERIES.items():
        a = pd.read_parquet(p)
        ok = len(a) == N_D and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy()) and np.allclose(a["cfg8"], x["C"], rtol=1e-12, atol=0)
        if not (ok and np.isfinite(a[[fc, lc]].to_numpy()).all()):
            raise SystemExit("%s: даты, rv, cfg8 расходятся с окном M или есть нечисловые прогнозы" % k)
        frames[k], F[k] = a, a[fc]
    for lay, s in LAYERS.items():
        F[lay] = frames[s][SERIES[s][2]]
    n, nw = registered_n(ckpt)
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), d=pd.DatetimeIndex(x["d"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames=frames, N=n, N_wide=nw)
