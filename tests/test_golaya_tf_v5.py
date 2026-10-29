"""Т-1: scripts/server/golaya_tf_v5.py — центр S1–S6 × ТФ и импульс. Только синтетика, без сети и данных.

Мутации (каждая ловится тестом): вход в день сигнала (Д2; mutations.md № 1) — test_mutaciya_vhod_v_den_signala;
неполный бар 18:00–18:50 учтён в сигнале — test_mutaciya_nepolnyj_bar; чтение ≥ 2020 без отказа — test_chtenie_*;
сторона издержек не взята — test_impuls_stroka_izderzhek.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "server"))
import golaya_tf_v5 as G  # noqa: E402

from fastbt import bars_agg as ba  # noqa: E402


def b10_iz(cena, ndays, start="2016-01-04"):
    """10-минутки: cena(день, слот) → цена (open = high = low = close); слоты 0..52 (10:00..18:40)."""
    days = pd.bdate_range(start, periods=ndays)
    rows = [(d, s, cena(i, s)) for i, d in enumerate(days) for s in range(ba.SLOTS)]
    df = pd.DataFrame(rows, columns=["date", "slot", "p"])
    for k in ("open", "high", "low", "close"):
        df[k] = df["p"]
    df["volume"] = 100.0
    return df.drop(columns="p")


def cena_proboj(posle_18=False):
    """Дни 0–8: 100; день 8 — прорыв вверх на баре 16:00–18:00 (слоты 36–47) или только после 18:00 (слоты 48–52);
    день 9: open 111, close 112; день 10: 120."""
    def f(i, s):
        if i == 8:
            return 110.0 if (s >= 48 if posle_18 else 36 <= s < 48) else 100.0
        if i == 9:
            return 112.0 if s == ba.SLOTS - 1 else 111.0
        return 120.0 if i >= 10 else 100.0
    return f


def test_tsentr_sredniy_element():
    assert G.tsentr({"a": [1, 2, 3], "b": [5, 10, 20, 30], "c": ["x", "net"]}) == {"a": 2, "b": 20, "c": "net"}


def test_impuls_signal_na_poslednem_polnom_bare():
    s = G.impuls_signal(b10_iz(cena_proboj(), 12))
    assert s.index[8] == pd.bdate_range("2016-01-04", periods=12)[8] and s.iloc[8] == 1
    assert (s.drop(s.index[8]).iloc[:8] == 0).all()


def test_mutaciya_nepolnyj_bar(monkeypatch):
    """Прорыв только в неполном баре 18:00–18:50: сигнала нет. Мутация POLNYH = 5 (неполный бар как последний) — ловится."""
    b = b10_iz(cena_proboj(posle_18=True), 12)
    assert (G.impuls_signal(b).iloc[:9] == 0).all()
    monkeypatch.setattr(G, "POLNYH", 5)
    assert G.impuls_signal(b).iloc[8] == 1          # мутант даёт сигнал — значит, предыдущая проверка его бы поймала


def test_impuls_vhod_open_dnya_d_plus_1_vyhod_close_d_plus_tau():
    b = b10_iz(cena_proboj(), 12)
    d, sig = ba.daily(b), G.impuls_signal(b)
    assert G.sdelki_impulsa(d, sig, 1, "2016-01-01", "2020-01-01")[0] == (1, 111.0, 112.0)
    assert G.sdelki_impulsa(d, sig, 2, "2016-01-01", "2020-01-01")[0] == (1, 111.0, 120.0)


def test_mutaciya_vhod_v_den_signala(monkeypatch):
    b = b10_iz(cena_proboj(), 12)
    d, sig = ba.daily(b), G.impuls_signal(b)
    monkeypatch.setattr(G, "LAG", 0)
    with pytest.raises(ValueError, match="Д2"):
        G.sdelki_impulsa(d, sig, 1, "2016-01-01", "2020-01-01")


def test_impuls_stroka_izderzhek(monkeypatch):
    b = b10_iz(cena_proboj(), 12)
    monkeypatch.setitem(G.PERIODY, "train4", dict(a="2016-01-01", b="2017-01-01", warm="2016-01-01"))
    rows = G.yacheyki_impuls({"SYN": b}, "train4")
    assert [r["tf"] for r in rows] == ["2h"] * 3 and [r["tochka"] for r in rows][0] == str(dict(kanal=20, tau=1))
    d, sig = ba.daily(b), G.impuls_signal(b)
    for r, tau in zip(rows, G.TAU):
        tr = G.sdelki_impulsa(d, sig, tau, "2016-01-01", "2017-01-01")
        yrs = 366 / 365.25
        assert r["sdelok"] == len(tr) > 0 and r["komissiya_storona_pct"] == pytest.approx(0.04)
        assert r["izderzhki_pct_god"] == pytest.approx(2 * 0.0004 * len(tr) * 100 / yrs)
        assert r["brutto_pct_god"] == pytest.approx(sum(s * (x / e - 1) for s, e, x in tr) * 100 / yrs)
        assert r["netto_pct_god"] == pytest.approx(r["brutto_pct_god"] - r["izderzhki_pct_god"])


def test_yacheyki_tf_vse_semyi_i_tf_imeyut_stroku_izderzhek(monkeypatch):
    rng = np.random.default_rng(1)
    w = np.exp(np.cumsum(rng.normal(0, 0.002, 40 * ba.SLOTS))) * 100
    b = b10_iz(lambda i, s: w[i * ba.SLOTS + s], 40)
    monkeypatch.setitem(G.PERIODY, "train4", dict(a="2016-01-04", b="2016-03-01", warm="2016-01-04"))
    for tf in G.TFS:
        tds = G.bary_tf({"SYN": b}, tf)
        for sid in G.SIDS:
            (r,) = G.yacheyki_tf(tds, sid, tf, "train4")
            assert (r["sid"], r["tf"], r["period"]) == (sid, tf, "train4")
            assert r["komissiya_storona_pct"] == pytest.approx(0.04) and np.isfinite(r["izderzhki_pct_god"])
            assert r["netto_pct_god"] == pytest.approx(r["brutto_pct_god"] - r["izderzhki_pct_god"])


def test_chtenie_ne_dalshe_2020(monkeypatch):
    with pytest.raises(ValueError, match="2020"):
        G.chtenie("SBER", "2022-01-01")
    ends = []
    monkeypatch.setattr(G.ba, "load_10m", lambda t, end, main=None: ends.append(end) or b10_iz(lambda i, s: 1.0, 2))
    for p in G.PERIODY:
        G.zagruzit(p, tickers=["SBER"])
    assert max(pd.Timestamp(e) for e in ends) <= G.GRANITSA
