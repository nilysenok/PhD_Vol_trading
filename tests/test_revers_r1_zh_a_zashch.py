"""R-ж для A_zashch (scripts/revers/r1_zh_zashch.py): синтетика с известным ответом + сверка с a_cd_test1.csv (нужны данные и кэш s9).
Без модульных monkeypatch (mutations.md № 43): мутации — передачей другого входа/объекта. Тест в сеть не ходит."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(p))
import r1_zh_zashch as Z  # noqa: E402

N = 400
DATES = pd.bdate_range("2021-01-04", periods=N)


class FakeEng:
    """Две бумаги × N баров, одна пара на бумагу; port = среднее по строкам дня pos × lr (без комиссии)."""
    tickers, nmax, n_t = ["AAA", "BBB"], N, np.array([N, N])

    def __init__(self):
        rng = np.random.default_rng(7)
        self.cache = {t: {"dates": DATES.to_numpy()} for t in self.tickers}
        self.r = dict(tick=np.repeat([0, 1], N), bar=np.tile(np.arange(N), 2), date=np.tile(np.arange(N), 2))
        self.lr = rng.normal(0, 0.01, (2, N))
        self.A_row = np.where(rng.random(2 * N) < 0.5, 1.0, -1.0)
        self.dates = DATES

    def g_row(self, G):
        return G[self.r["tick"], self.r["bar"]]

    def port(self, pos, comm=None):
        x = np.bincount(self.r["date"], pos * self.lr[self.r["tick"], self.r["bar"]], N) / 2
        e = np.bincount(self.r["date"], np.abs(pos), N) / 2
        return pd.DataFrame({"excess": x, "exposure": e}, index=DATES)


def ser_rastushchij() -> dict:
    return {t: pd.Series(np.arange(N, dtype=float) + k, index=DATES) for k, t in enumerate(FakeEng.tickers)}


def test_zatvor_vsegda_otkryt_daet_bazu():
    e = FakeEng()
    a, b = Z.ryad(e, e.A_row, 0.0), Z.ryad(e, e.A_row * e.g_row(np.ones((2, N))), 0.0)
    assert np.abs(a["x"] - b["x"]).max() <= 1e-12


def test_zatvor_zakryvaet_i_menyaet_ryad():
    e = FakeEng()
    G = Z.matrica_zatvora(e, ser_rastushchij())          # растущая σ̂ — ранг 1, закрыт после 126 значений
    assert G[:, :125].all() and not G[:, 126:].any()
    assert Z.ryad(e, e.A_row * e.g_row(G), 0.0)["delo"][130:].sum() == 0


def test_zatvor_ne_zavisit_ot_budushchego_i_mutaciya_sdviga_lovitsya():
    e, s = FakeEng(), ser_rastushchij()
    G = Z.matrica_zatvora(e, s)
    s2 = {t: v.copy() for t, v in s.items()}
    for v in s2.values():
        v.iloc[250:] = -1e9                                    # будущее после бара 249 переписано
    assert np.array_equal(Z.matrica_zatvora(e, s2)[:, :250], G[:, :250])
    # мутация «σ̂ сдвинута на +1 день» (mutations.md № 1): ряд, известный завтра, — решение в баре 249 меняется
    bad = {t: v.shift(-1) for t, v in s2.items()}
    assert not G[:, 249].any() and Z.matrica_zatvora(e, bad)[:, 249].all()


class Nul:
    def integers(self, lo, hi):
        return 0


def test_k2_sdvig_nul_daet_ishodnuyu_masku():
    e = FakeEng()
    G = Z.matrica_zatvora(e, ser_rastushchij())
    assert np.array_equal(Z.sdvig_k2(e, G, Nul()), G)             # mutations.md № 9


def test_k2_sdvig_menyaet_masku_i_sohranyaet_dolyu():
    e = FakeEng()
    s = {t: pd.Series(np.sin(np.arange(N) / 9.0) + k, index=DATES) for k, t in enumerate(e.tickers)}
    G = Z.matrica_zatvora(e, s)
    K = Z.sdvig_k2(e, G, np.random.default_rng(5))
    assert not np.array_equal(K, G)
    assert abs(K.sum() - G.sum()) <= 2                             # круговой сдвиг, последний день не сдвигается


def test_p_znachenie_nan_ne_podmenyaetsya():
    assert np.isnan(Z.p_znachenie(1.0, [np.nan] * 5))             # нет конечных повторов — nan, не 1/(n+1)
    assert np.isnan(Z.p_znachenie(np.nan, [1.0, 2.0]))
    assert Z.p_znachenie(1.5, [1.0, 2.0, np.nan, 3.0]) == pytest.approx(3 / 4)   # (1+2)/(3+1)


def test_stroki_pokazyvayut_dolyu_konechnyh_k2():
    e = FakeEng()
    r = Z.ryad(e, e.A_row, 0.0)
    rows = Z.stroki("x", "t", r, r, [r, r])
    assert all(x["n_k2"] == 2 and x["n_k2_konechnyh_sharpe"] == 2 for x in rows) and rows[0]["d_sharpe_vs_baza"] == 0


def test_dannye_vsegda_otkryt_raven_a_cd_test1():
    if not os.environ.get("FASTBT_Z2_R2FINAL") or not os.environ.get("FASTBT_MAIN"):
        pytest.skip("нужны env Z2 и FASTBT_MAIN (кэш s9 test1)")
    import r1_a_cd as C
    from fastbt import s9
    with s9.period_ctx("test1"):
        eng = C.s9.Engine(C.kesh("test1"))
        comm = Z.AS.COMM
        G1 = np.ones((len(eng.tickers), eng.nmax))
        r = Z.ryad(eng, eng.A_row * eng.g_row(G1), comm)
    t = pd.read_csv(ROOT / "docs/tables/revers_r1/a_cd_test1.csv", comment="#")
    a = t[t.portfel == "A (s9, база сравнения)"]
    for nm, g in (("(а) полные", Z.VSE), ("по году 2022", 2022), ("по году 2024", 2024)):
        row, m = a[a.stroka == nm].iloc[0], Z.metriki(r, g)
        assert abs(row.sharpe - m["sharpe"]) <= 1e-12 and abs(row.msr - m["msr"]) <= 1e-12
