"""S-A-IMPL-3 (Ж-254; `S_A_SPEC.md`, разд. 3.4): слои X1 и W5-G над новой компонентой C_A. Код слоёв — прежний, импорт без правок (`s15/de_e1_state.run`, `s16/x1_w5_gap.walk`).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Только чтение рядов X1, W5-G, K2, K3.
Публичное API:
- `inputs() -> (x, gap)`: окно M (`x1_w3_core.load`: d, t_plus, rv, M = B_mix, C = F cfg8, s = e_d, F, K1, K2, K3; n = 2198) и gap^PIT (`x1_w5_gap_cal.pit_table`, numpy int).
- `layers(x, c_a, gap) -> DataFrame`: X1_A = `de_e1_state.run(rv, M, C_A, s, d, t⁺)` (α по состоянию из сетки X1, помесячно) и F_A = `x1_w5_gap.walk(d, t⁺, rv, X1_A, gap)` (β_gap по закрытым парам, месяцы ≥ 2018-07-02).
  Колонки: d, t_plus, rv, X1_A, pi, a0, a1, F, beta_gap, refit_date. c_a — массив C_A на датах x (n = 2198).
- `check_nest() -> list` (CLI `--check-nest`, только чтение): `run(B_mix, cfg8 F)` = x1 (F, π, α₀, α₁) ≤ 1e-12 отн.; `walk` над ним = `data/regen/harj_x1/W5/w5g.parquet` (F, β) ≤ 1e-12;
  нулевые параметры (C_A из `a_model.forward` при θ = 0) ⇒ F_A ≡ W5-G ≤ 1e-12.
Запуск: `python3 -W ignore -m models_v2.s18.a_layers --check-nest`."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from models_v2.s15 import de_e1_state as E1
from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_w5_gap as W5
from models_v2.s16 import x1_w5_gap_cal as CAL

TOL = 1e-12


def inputs() -> tuple:
    x = C.load()
    tbl = CAL.pit_table(x, CAL.load_calendar(), CAL.canonical_saturdays())
    return x, tbl["gap_pit"].to_numpy(int)


def layers(x: pd.DataFrame, c_a, gap) -> pd.DataFrame:
    c_a = np.asarray(c_a, float)
    r = E1.run(x["rv"].to_numpy(float), x["M"].to_numpy(float), c_a, x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy())
    w = W5.walk(x["d"], x["t_plus"], x["rv"], r["F"], gap, base_name="X1_A")
    return pd.DataFrame(dict(d=x["d"].to_numpy(), t_plus=x["t_plus"].to_numpy(), rv=x["rv"].to_numpy(float), X1_A=r["F"], pi=r["pi"], a0=r["a0"], a1=r["a1"],
                             F=w["F"].to_numpy(), beta_gap=w["beta"].to_numpy(), refit_date=w["refit_date"].to_numpy()))


def _rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)))


def check_nest() -> list:
    from models_v2.s18 import a_model as M
    x, gap = inputs()
    base = layers(x, x["C"], gap)
    w5g = pd.read_parquet(C.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    bad, out = [], {}
    out["X1_A(cfg8) vs x1.F"] = _rel(base["X1_A"], x["F"])
    out["π"], out["α₀"], out["α₁"] = _rel(base["pi"], x["pi"]), _rel(base["a0"], x["a0"]), _rel(base["a1"], x["a1"])
    out["F_A(cfg8) vs w5g.F"] = _rel(base["F"], w5g["F"])
    out["β vs w5g.β (абс.)"] = float(np.max(np.abs(base["beta_gap"].to_numpy() - w5g["beta"].to_numpy())))
    D = M.load_data()
    import torch
    _, lnC = M.forward(torch.zeros(M.NP, dtype=torch.float64), D.view(np.arange(len(D.d))), "A2")
    c0 = pd.Series(np.exp(lnC.numpy()), index=pd.DatetimeIndex(D.d)).reindex(pd.DatetimeIndex(x["d"])).to_numpy()
    zero = layers(x, c0, gap)
    out["θ = 0: F_A vs W5-G"] = _rel(zero["F"], w5g["F"])
    for k, v in out.items():
        print("%s: %.2e" % (k, v))
        bad += [] if v <= TOL else ["%s: %.2e" % (k, v)]
    return bad


if __name__ == "__main__":
    if "--check-nest" not in sys.argv:
        raise SystemExit("укажите --check-nest")
    b = check_nest()
    print("--check-nest: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
    sys.exit(1 if b else 0)
