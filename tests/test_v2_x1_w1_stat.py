"""W1.6: Холм с N программы (751), не числом строк (№ 103); знак Δ и «лучше/хуже» (№ 125/140); «подтверждено» только при p Холма < 0,05 (№ 158). Синтетика с известным ответом."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w1_stat as SS
from models_v2.s16.x1_ze_crit import N_HOLM, N_WIDE


def wins(x1_scale: float = 0.25, zb_scale: float = 0.25, n: int = 700) -> dict:
    d = pd.bdate_range("2018-01-03", periods=n)
    tp = pd.DatetimeIndex(list(d[1:]) + [d[-1] + pd.offsets.BDay(1)])
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(0, 0.3, n))
    sd = dict(K1=0.4, K2=0.3, K3=0.3, X1=x1_scale, ZB=zb_scale)
    al = {k: pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=rv * np.exp(rng.normal(0.02, s, n)))) for k, s in sd.items()}
    return {"M": al, "H": al, "F": al}


def test_holm_uses_program_n_not_row_count_and_mutation_is_caught() -> None:
    s = SS.stats_table(wins())
    m = s[s["window"] == "M"]
    assert len(m) == len(SS.PAIRS) == 7 and N_HOLM == 751 and N_WIDE == 3207 and (s["n_holm"] == N_HOLM).all()
    p = m["p_one"].to_numpy()
    assert 0 < p.min() < 1e-3 and N_HOLM * p.min() < 1                                          # поправка не насыщается: различие N и числа строк видно
    assert m["p_holm_N"].min() == pytest.approx(min(1.0, N_HOLM * p.min()), abs=1e-12)          # наименьшее p умножается на N программы
    assert m["p_holm_Nwide"].min() == pytest.approx(min(1.0, N_WIDE * p.min()), abs=1e-12)
    rows_only = np.minimum(1.0, len(m) * p.min())                                                # мутант: N = числу строк
    assert rows_only < 0.5 * m["p_holm_N"].min()


def test_delta_sign_and_words_known_answer() -> None:
    d = pd.bdate_range("2018-01-03", periods=700)
    tp = pd.DatetimeIndex(list(d[1:]) + [d[-1] + pd.offsets.BDay(1)])
    rv = np.exp(np.random.default_rng(1).normal(0, 0.3, 700))
    noise = np.exp(np.random.default_rng(2).normal(0, 0.01, 700))
    al = {k: pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=rv * f)) for k, f in dict(K1=3.0 * noise, K2=2.0 * noise, K3=2.0 * noise, X1=noise, ZB=noise).items()}
    s = SS.stats_table({"M": al, "H": al, "F": al}).query("window == 'M'").set_index(["hybrid", "base"])
    r = s.loc[("X1", "K2")]
    l2 = 0.5 - np.log(0.5) - 1.0                                                                          # l при F = 2·RV
    assert r.qlike_b == pytest.approx(l2, abs=1e-3) and r.qlike_h < 1e-3 and r.delta_pct == pytest.approx(100 * (r.qlike_h - r.qlike_b) / r.qlike_b, abs=1e-9) and r.delta_pct < -99   # Δ в % QLIKE базы, минус — лучше
    assert r.verdict.startswith("численно лучше базы")
    rev = SS.stats_table({"M": {**al, "X1": al["K2"], "K2": al["X1"]}, "H": al, "F": al}).query("window == 'M'").set_index(["hybrid", "base"]).loc[("X1", "K2")]
    assert rev.delta_pct > 0 and rev.verdict.startswith("численно не лучше") and "подтверждено (" not in rev.verdict.replace("не подтверждено (", "")


def confirmed_property(fn) -> bool:
    """«подтверждено» (без «не») допустимо только при Δ < 0 и p Холма < 0,05; на сетке значений."""
    for delta in (-5.0, -0.1, 0.0, 0.1, 5.0):
        for p in (0.0, 0.049, 0.05, 0.2, 1.0):
            v = fn(delta, p)
            yes = " подтверждено" in (" " + v) and "не подтверждено" not in v
            if yes != (delta < 0 and p < 0.05) or (delta >= 0 and "лучше базы" in v.split(";")[0] and "не лучше" not in v):
                return False
    return True


def test_confirmed_only_when_holm_below_005_and_mutations_are_caught() -> None:
    assert confirmed_property(SS.verdict)
    assert SS.verdict(-1.0, 0.049).endswith("подтверждено (p Холма < 0,05)") and "не подтверждено" in SS.verdict(-1.0, 0.05)
    assert not confirmed_property(lambda d, p: "численно лучше базы; подтверждено" if d < 0 else "численно не лучше базы; не подтверждено")      # мутант: p не учитывается
    assert not confirmed_property(lambda d, p: ("численно лучше базы" if d > 0 else "численно не лучше базы") + "; " + ("подтверждено" if p < 0.05 else "не подтверждено"))   # мутант: знак Δ перевёрнут
