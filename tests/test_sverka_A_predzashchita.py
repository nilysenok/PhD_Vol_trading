"""Синтетика для scripts/analysis/sverka_A_predzashchita.py: переключатели меняют результат, ошибка ловится."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import sverka_A_predzashchita as S  # noqa: E402

N = 55


def panel(drop_pair=False, scale=1.0):
    idx = pd.bdate_range("2019-01-01", periods=N)
    rows = []
    for t in ("X", "Y"):
        for s in ("S1", "S2"):
            if drop_pair and (t, s) == ("Y", "S2"):
                continue
            pos = np.where(np.arange(N) % 10 < 5, 1.0, 0.0) * (1 if t == "X" else -1) * scale
            gr = np.random.default_rng(7 + len(t + s)).normal(0, .01, N) * pos
            rows.append(pd.DataFrame(dict(date=idx, ticker=t, strategy=s, position=pos, gross=gr)))
    return pd.concat(rows), pd.Series(.1, index=idx)


def test_selftest_runs():
    S.selftest()


def test_komissiya_znak():
    pos, rate = panel()
    m0 = S.portfolio(pos, rate, 0.0)["excess"].mean()
    assert S.portfolio(pos, rate, .01)["excess"].mean() < m0
    assert S.portfolio(pos, rate, -.01)["excess"].mean() > m0      # перепутанный знак виден


def test_agregirovanie_polnaya_panel_ravna_i_neravna_pri_dyre():
    pos, rate = panel()
    assert np.allclose(S.portfolio(pos, rate, .01, "pairs")["r"], S.portfolio(pos, rate, .01, "tickw")["r"])
    hole, rate = panel(drop_pair=True)
    assert not np.allclose(S.portfolio(hole, rate, 0, "pairs")["r"], S.portfolio(hole, rate, 0, "tickw")["r"])


def test_stavka_dni_i_plecho():
    pos, rate = panel()
    assert not np.allclose(S.portfolio(pos, rate, 0, fwd=True)["excess"], S.portfolio(pos, rate, 0, fwd=False)["excess"])
    big, rate = panel(scale=3.0)
    assert not np.allclose(S.portfolio(big, rate, 0, lev_cal=True)["total"], S.portfolio(big, rate, 0, lev_cal=False)["total"])
    flat = S.portfolio(pos, rate, 0, lev_cal=True)                  # экспозиция ≤ 1: плечо вклада не даёт
    assert np.allclose(flat["total"], S.portfolio(pos, rate, 0, lev_cal=False)["total"])


def test_vyhod_n5_platit_na_kontse():
    pos, rate = panel()
    a, b = S.portfolio(pos, rate, .01, exit_fee=True), S.portfolio(pos, rate, .01, exit_fee=False)
    assert a["r"].iloc[-1] < b["r"].iloc[-1] and np.allclose(a["r"].iloc[:-1], b["r"].iloc[:-1])
