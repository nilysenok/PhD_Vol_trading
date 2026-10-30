"""Тестировщик #737: настоящий свод Д8 (docs/issledovaniya/div_sverka_svod.csv), календарь подменён (рабочие дни до 2026-02-28).
Дополняет tests/test_wf_dividendy.py (там синтетический свод): число годных строк, две негодные, снятие гэпа на трёх реальных выплатах,
ключ кэша на настоящем своде, dev побайтно не зависит от свода. Мутации (godna вместо godna_korr; без фильтра; свод в dev; свод не в ключе)
проверены в отдельной копии — все пойманы tests/test_wf_dividendy.py; здесь — то же на реальных строках."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy as dv
from fastbt import zona

CAL = pd.bdate_range("2014-06-02", "2026-02-28").values.astype("datetime64[ns]")
SVOD = pd.read_csv(dv.SVOD)
S22 = SVOD[pd.to_datetime(SVOD["record_date"]) >= "2022-01-01"]


@pytest.fixture
def wf(monkeypatch):
    monkeypatch.setattr(dv, "_calendar", lambda: CAL)
    monkeypatch.delenv("FASTBT_Z2_DIV", raising=False)
    with zona.zone_ctx("wf"):
        yield


def _suma(ticker: str, lo: str, hi: str) -> float:
    d = dv.gap_divs(ticker, CAL)
    return float(d[np.searchsorted(CAL, np.datetime64(lo)):np.searchsorted(CAL, np.datetime64(hi))].sum())


def test_svod_2022_plus_74_stroki_72_godny_korr():
    assert len(SVOD) == 247 and len(S22) == 74 and int(S22["godna_korr"].sum()) == 72


def test_dve_negodnye_ne_koryectiruyut(wf):
    bad = S22[S22["godna_korr"] == 0]
    assert sorted(bad["ticker"]) == ["HYDR", "TATN"]
    for _, r in bad.iterrows():
        g = int(np.searchsorted(CAL, np.datetime64(dv.last_cum(pd.Series([pd.Timestamp(r["record_date"])])).iloc[0]), side="right"))
        assert dv.gap_divs(r["ticker"], CAL)[g] == 0.0, r["ticker"]


def test_summa_2022_plus_po_tikeram_ravna_godnym_korr(wf):
    for t, g in S22.groupby("ticker"):
        ozhid = float(g.loc[(g["godna_korr"] == 1) & (g["dps_rub"] > 0), "dps_rub"].sum())
        assert _suma(t, "2022-01-01", "2026-03-01") == pytest.approx(ozhid), t


@pytest.mark.parametrize("ticker,record,level", [("ALRS", "2023-10-18", 70.0), ("IRAO", "2024-06-03", 4.0), ("AFLT", "2025-07-18", 55.0)])
def test_gep_snyat_na_realnoj_vyplate(wf, ticker, record, level):
    row = SVOD[(SVOD["ticker"] == ticker) & (SVOD["record_date"] == record)].iloc[0]
    assert row["godna_korr"] == 1
    D = float(row["dps_rub"])
    d = dv.gap_divs(ticker, CAL)
    g = int(np.flatnonzero(d)[np.flatnonzero(d) >= np.searchsorted(CAL, np.datetime64(record)) - 5][0])
    assert d[g] == pytest.approx(D)
    close = np.full(len(CAL), level)
    close[g:] -= D                                                   # сырой ряд: цена упала на D в день гэпа
    after = g + 1
    (adj,) = dv.adjust_ohlc(ticker, CAL, close, close.copy())
    # прочие выплаты тикера имеют свои гэпы в другие дни: сравниваем только соседние бары вокруг этого гэпа
    assert close[g] / close[g - 1] < 1 - 0.5 * D / level             # в сыром ряду гэп есть
    assert adj[g] / adj[g - 1] == pytest.approx(1.0, abs=1e-9)       # в скорректированном — снят
    assert adj[after] / adj[g] == pytest.approx(1.0, abs=1e-9)


def test_klyuch_kesha_realnyj_svod_godna_korr(wf, monkeypatch, tmp_path):
    k = dv.digest()
    s = SVOD.copy()
    i = s.index[(s["ticker"] == "HYDR") & (s["record_date"] == "2022-07-10")][0]
    s.loc[i, "godna_korr"] = 1                                      # одна строка 2022+ стала годной: другая корректировка — другой кэш
    p = tmp_path / "svod.csv"
    s.to_csv(p, index=False)
    monkeypatch.setattr(dv, "SVOD", p)
    assert dv.digest() != k


def test_dev_realnyj_svod_ne_vliyaet_na_sobytiya_i_klyuch(monkeypatch, tmp_path):
    assert zona.name() == "dev"
    k, e = dv.digest(), dv.events("SBER")
    p = tmp_path / "svod.csv"
    SVOD.assign(godna_korr=0).to_csv(p, index=False)
    monkeypatch.setattr(dv, "SVOD", p)
    assert dv.digest() == k
    pd.testing.assert_frame_equal(dv.events("SBER"), e)
    assert (e["last_cum_date"] < pd.Timestamp("2022-01-01")).all()
