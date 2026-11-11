"""S-C-IMPL-3 (Ж-254; `S_C_SPEC.md`, разд. 1.4): слои X1 и W5-G над каналом C_V = cfg8·𝓜_V. Код слоёв — прежний, импорт без правок (`s15/de_e1_state.run`, `s16/x1_w5_gap.walk`).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Только чтение рядов X1, W5-G, K2, K3.
Публичное API:
- `inputs() -> (x, gap)`: окно M (`x1_w3_core.load`: d, t_plus, rv, M = B_mix, C = F cfg8, s = e_d, F, K1, K2, K3; n = 2198) и gap^PIT (`x1_w5_gap_cal.pit_table`, numpy int).
- `layers(x, c_v, gap) -> DataFrame`: X1_V = `de_e1_state.run(rv, M, C_V, s, d, t⁺)` (канал доверия над НОВЫМ C_V, B_mix — прежний) и F = `x1_w5_gap.walk(d, t⁺, rv, X1_V, gap)` (поправка перерыва над X1_V).
  Колонки: d, t_plus, rv, X1_V, pi, a0, a1, F, beta_gap, refit_date. c_v — массив C_V на датах x.
- `check_nest() -> list` (CLI `--check-nest`, только чтение): `run(B_mix, cfg8)` = x1 (F, π, α₀, α₁) ≤ 1e-12 отн.; `walk` над ним = `data/regen/harj_x1/W5/w5g.parquet` (F, β) ≤ 1e-12;
  нулевые параметры (𝓜 из `cm_model.forward` при v = 0, все четыре обучаемых варианта) ⇒ F_V ≡ W5-G ≤ 1e-12.
Запуск: `python3 -W ignore -m models_v2.s18.cm_layers --check-nest`."""
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


def layers(x: pd.DataFrame, c_v, gap) -> pd.DataFrame:
    c_v = np.asarray(c_v, float)
    r = E1.run(x["rv"].to_numpy(float), x["M"].to_numpy(float), c_v, x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy())
    w = W5.walk(x["d"], x["t_plus"], x["rv"], r["F"], gap, base_name="X1_V")
    return pd.DataFrame(dict(d=x["d"].to_numpy(), t_plus=x["t_plus"].to_numpy(), rv=x["rv"].to_numpy(float), X1_V=r["F"], pi=r["pi"], a0=r["a0"], a1=r["a1"],
                             F=w["F"].to_numpy(), beta_gap=w["beta"].to_numpy(), refit_date=w["refit_date"].to_numpy()))


def _rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)))


def check_nest() -> list:
    import torch

    from models_v2.s18 import cm_feat as CF
    from models_v2.s18 import cm_model as M
    x, gap = inputs()
    base = layers(x, x["C"], gap)
    w5g = pd.read_parquet(C.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    out = {"X1_V(cfg8) vs x1.F": _rel(base["X1_V"], x["F"]), "π": _rel(base["pi"], x["pi"]), "α₀": _rel(base["a0"], x["a0"]), "α₁": _rel(base["a1"], x["a1"]),
           "F_V(cfg8) vs w5g.F": _rel(base["F"], w5g["F"]), "β vs w5g.β (абс.)": float(np.max(np.abs(base["beta_gap"].to_numpy() - w5g["beta"].to_numpy())))}
    feat = CF.load_feat()
    for var in M.TRAINED:
        V = M.build_data(x, feat, lag=var == "CKL").view(np.arange(len(x)))
        m = M.forward(torch.zeros(M.NP, dtype=torch.float64), V, var)["M"].numpy()
        out["θ = 0, %s: F_V vs W5-G" % var] = _rel(layers(x, x["C"].to_numpy(float) * m, gap)["F"], w5g["F"])
    bad = []
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
