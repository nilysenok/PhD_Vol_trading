"""Таблица по годам §5–§7: синтетика с известным ответом. Мутации: ddof=0 (№ 12), Δ не из таблицы (№ 13),
медиана с 2026, mSR по календарным неделям вместо W-FRI."""
import math

import numpy as np
import pandas as pd
import pytest

from fastbt import tablica_godov_wf as T
from fastbt.metriki_e import msr

IDX = pd.bdate_range("2017-01-02", "2026-02-27")
UZOR = np.tile([1.0, -0.5, 0.3, -0.2, 0.4], len(IDX) // 5 + 1)[: len(IDX)] * 0.01


def _ryad(sdvig=0.0, shum=None):
    x = UZOR + sdvig
    if shum is not None:
        x = x + np.random.RandomState(shum).normal(0, 0.004, len(IDX))
    return pd.Series(x, index=IDX)


def _vse(**izm):
    r = {n: _ryad(2e-4 * i, shum=i) for i, n in enumerate(T.ANSAMBLI)}
    r.update(izm)
    return r


def _sr(a):
    a = np.asarray(a, float)
    return a.mean() / a.std(ddof=1) * math.sqrt(252)


def test_sharpe_ddof1_sqrt252_po_godam():
    r = _vse()
    t = T.tablica(r)
    n = T.ANSAMBLI[0]
    for g in (2017, 2021, 2025):
        assert t.loc[str(g), (n, "sharpe")] == pytest.approx(_sr(r[n][r[n].index.year == g]), rel=1e-12)
    assert t.loc["итог", (n, "sharpe")] == pytest.approx(_sr(r[n]), rel=1e-12)
    assert t.loc["итог", (n, "sharpe")] != pytest.approx(_sr(r[n]) * math.sqrt((len(r[n]) - 1) / len(r[n])), rel=1e-9)  # не ddof=0


def test_msr_nedelnyj_w_fri():
    r = _vse()
    n = T.ANSAMBLI[1]
    x = r[n][r[n].index.year == 2019]
    w = x.groupby(x.index.to_period("W-FRI")).sum().to_numpy()
    want = msr(w, 52)
    assert T.tablica(r).loc["2019", (n, "msr_ned")] == pytest.approx(want, rel=1e-12, nan_ok=True)
    kal = x.groupby(x.index.to_period("W-SUN")).sum().to_numpy()  # W-SUN — те же недели Пн–Пт: сверка группировки
    assert np.allclose(w, kal)
    mesyac = x.groupby(x.index.to_period("M")).sum().to_numpy()
    assert not (len(mesyac) == len(w))


def test_mediana_9_let_bez_2026():
    n = T.ANSAMBLI[0]
    x = pd.Series(0.0, index=IDX)
    for i, g in enumerate(T.GODY):  # годовой Sharpe растёт с i
        m = IDX.year == g
        x[m] = UZOR[m] + 1e-4 * i
    m26 = IDX.year == 2026
    x[m26] = UZOR[m26] - 0.05  # 2026 резко отрицательный
    t = T.tablica(_vse(**{n: x}))
    god = [t.loc[str(g), (n, "sharpe")] for g in T.GODY]
    assert t.loc["медиана 9 лет", (n, "sharpe")] == pytest.approx(np.median(god), rel=1e-12)
    s26 = t.loc["янв–фев 2026", (n, "sharpe")]
    assert t.loc["медиана 9 лет", (n, "sharpe")] != pytest.approx(np.median(god + [s26]), rel=1e-6)  # мутация: 10 значений


def test_delta_raznost_tablichnyh():
    r = _vse()
    t = T.tablica(r)
    out = T.SW.sravnit(r, (("C", "D"),), reps=50)["rows"][0]
    assert out["delta"] == pytest.approx(t.loc["итог", ("C", "sharpe")] - t.loc["итог", ("D", "sharpe")], abs=1e-9)


def test_pravilo_chteniya_oba_usloviya():
    n = T.ANSAMBLI[0]
    plus = _vse(**{n: _ryad(5e-4)})
    assert T.pravilo(T.tablica(plus))[n] == "выполнено"
    minus = _vse(**{n: _ryad(-3e-3)})
    assert T.pravilo(T.tablica(minus))[n] == "нет"
    # итог > 0 за счёт 2026, медиана по годам < 0 → «нет»
    x = _ryad(-2.2e-3)
    x[IDX.year == 2026] = 0.5 * np.tile([1, -0.9], 100)[: (IDX.year == 2026).sum()]
    t = T.tablica(_vse(**{n: x}))
    assert t.loc["итог", (n, "sharpe")] > 0 and t.loc["медиана 9 лет", (n, "sharpe")] < 0
    assert T.pravilo(t)[n] == "нет"


def test_granitsy_i_kontrakt():
    r = _vse()
    r["C"] = pd.Series(1.0, index=pd.bdate_range("2017-01-02", "2026-03-10"))
    with pytest.raises(ValueError):
        T.tablica(r)
    r = _vse()
    del r["D"]
    with pytest.raises(KeyError):
        T.tablica(r)
    r = _vse()
    r["C"] = r["C"].copy()
    r["C"].iloc[5] = np.nan
    with pytest.raises(ValueError):
        T.tablica(r)


def test_pustoj_god_i_nulevoj_ryad_nan_ne_nol():
    r = _vse(C=pd.Series(0.0, index=IDX))
    t = T.tablica(r)
    assert np.isnan(t.loc["2020", ("C", "sharpe")]) and np.isnan(t.loc["итог", ("C", "sharpe")])
    assert T.pravilo(t)["C"] == "нет"


def test_zapis_shapka_i_predzashchita(tmp_path, monkeypatch):
    r = _vse()
    monkeypatch.setattr(T, "PREDZASHCHITA", tmp_path / "net.csv")
    out = T.zapisat(r, tmp_path / "o", pary=(("C", "D"), ("A_strogaya", "D")))
    assert out["sravneniya"]["m"] == 2
    tx = (tmp_path / "o" / "tablica_godov.csv").read_text(encoding="utf-8")
    assert tx.startswith("# v: 1") and "SHA предрегистрации" in tx and "хэш входов" in tx and "правило чтения" in tx
    assert "не найден" in (tmp_path / "o" / "predzashchita.csv").read_text(encoding="utf-8")
    assert len(T.sha_predreg()) == 40
