"""V6-3, G31: диагностика обращения времени (B). Обращённый день: порядок баров обратный, доходности с обратным знаком (`put`) или со своим знаком (`poryadok`), объём в обратном порядке; RV = Σr² не меняется.
Известные ответы: при обращении порядка баров все 12 признаков Σ меняют знак, s1–s7 инвариантны, s8 и s9 меняются местами, ρ6 инвариантен; при обращении пути s5 меняет знак. Мутация «порядок баров не обращён» ловится известным ответом.
Обращённые признаки во вход прогноза не попадают: `v6_3_tr` импортирует только оркестратор `run_v6` (AST), дизайн E1–E3 не содержит TR."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_m10 as M
from models_v2.s13 import v6_1_feat as FE, v6_3_io as IO3, v6_3_r as RF, v6_3_stage as ST, v6_3_tr as TR


def make_bars(seed: int = 3, days: int = 70, nb: int = 53) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for k in range(days):
        day = pd.Timestamp("2016-01-04") + pd.offsets.BDay(k)
        for j in range(nb):
            o = px
            px = px * np.exp(rng.normal(0, 0.002))
            rows.append((day + pd.Timedelta(minutes=600 + 10 * j), o, max(o, px), min(o, px), px, float(rng.integers(100, 1000))))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume"])


def daily_of(bars):
    return D.build_daily(bars, index_days=None)[0]


def setup(days: int = 70):
    bars = make_bars(days=days)
    daily = daily_of(bars)
    return bars, daily, daily["rv"].astype(float)


def test_reversed_bars_have_reversed_returns_and_same_rv():
    bars = make_bars(days=6)
    r0 = M.bar_returns(bars)
    v0 = D.main_session(bars).sort_values("begin")["volume"].to_numpy(float)
    for mode, sign in (("put", -1.0), ("poryadok", 1.0)):
        rb = TR.reverse_bars(bars, mode)
        r1 = M.bar_returns(rb)
        for d, ix in r0.groupby("date").indices.items():
            assert np.allclose(r1["r"].to_numpy()[ix], sign * r0["r"].to_numpy()[ix][::-1], rtol=0, atol=1e-12)
            assert np.array_equal(rb["volume"].to_numpy()[ix], v0[ix][::-1])
        a, b = FE.day_aggregates(bars)["rv"], FE.day_aggregates(rb)["rv"]
        assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=0)                   # RV = Σr² инвариантна


def test_known_answers_order_reversal():
    bars, daily, h = setup()
    t = TR.diag(bars, daily, h).set_index(["mode", "group", "feature"])["verdict"]
    assert (t.loc["poryadok", "Sigma"] == "меняет знак").all() and len(t.loc["poryadok", "Sigma"]) == 12
    assert all(t.loc[("poryadok", "S", "s%d" % i)] == "инвариантен" for i in range(1, 8))
    assert t.loc[("poryadok", "S", "s8")] == "меняется" and t.loc[("poryadok", "S", "s9")] == "меняется"
    assert t.loc[("poryadok", "R", "rho6")] == "инвариантен" and t.loc[("poryadok", "R", "rho1")] == "меняется"
    assert IO3.tr_bad(TR.diag(bars, daily, h)) == []


def test_known_answers_path_reversal():
    bars, daily, h = setup()
    d = TR.diag(bars, daily, h).set_index(["mode", "group", "feature"])
    assert d.loc[("put", "S", "s5"), "verdict"] == "меняет знак" and d.loc[("put", "S", "s4"), "verdict"] == "меняется"
    assert d.loc[("put", "Sigma", "a_VQ"), "verdict"] == "меняет знак" and d.loc[("put", "R", "rho6"), "verdict"] == "инвариантен"
    assert abs(d.loc[("put", "S", "s5"), "corr_real_reversed"] + 1.0) < 1e-6 and len(d) == 54


def test_mutation_order_not_reversed_is_caught(monkeypatch):
    bars, daily, h = setup()
    monkeypatch.setattr(TR, "reverse_bars", lambda b, m: D.main_session(b).sort_values("begin").reset_index(drop=True))         # МУТАЦИЯ: «обращённый» день = прямой
    assert IO3.tr_bad(TR.diag(bars, daily, h)) != []


def test_real_sigma_from_arrays_equals_stage_1b_features():
    from models_v2.s13 import v6_1b_sig as SG
    bars, daily, _ = setup()
    a = TR.sigma_areas(RF.day_arrays(bars), daily.index)
    b = SG.features(bars, daily)[0]
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12, equal_nan=True) and list(a.columns) == SG.SIGMA_NAMES


def test_diag_does_not_touch_inputs():
    bars, daily, h = setup()
    b0 = bars.copy()
    TR.diag(bars, daily, h)
    assert bars.equals(b0)


def test_tr_module_imported_only_by_orchestrator_and_not_in_design():
    here = Path(ST.__file__).parent
    for p in sorted(here.glob("*.py")):
        if p.name in ("run_v6.py", "v6_3_tr.py"):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "v6_3_tr" not in names and "models_v2.s13.v6_3_tr" not in names, p.name
    run = ast.parse((here / "run_v6.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.ImportFrom) and any(a.name == "v6_3_tr" for a in n.names) for n in ast.walk(run))
    bars, daily, _ = setup()
    rho = RF.features(bars, daily)[RF.RHO_NAMES]
    X = ST.design("E3", pd.DataFrame(index=rho.index), {"R": rho, "sigma": pd.DataFrame(index=rho.index)})
    assert list(X.columns) == RF.RHO_NAMES
