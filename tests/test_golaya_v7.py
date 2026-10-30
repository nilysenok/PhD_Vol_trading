"""Л1.6 (Ж-146): драйвер голых идей S7–S12 — тесты на синтетике (прогон Р1 не делается, сеть закрыта)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_s7_s9 as ref  # noqa: E402
import golaya_v7 as gv  # noqa: E402
from fastbt import bars_agg, daily_1850 as d18, kross_v7 as kv, metriki_diag as md  # noqa: E402
from fastbt.golaya_v7_isp import ispolnit  # noqa: E402

TK = gv.TICKERS[:8]
SID = ("S7", "S8", "S9", "S10", "S11", "S12")


def bary(seed=0, dni=420, start="2015-03-02", tk=TK):
    """Синтетические 10-минутки 10:00–18:40: случайное блуждание с пробоями; dict тикер → bars."""
    out, days = {}, pd.bdate_range(start, periods=dni)
    mins = pd.to_timedelta(np.arange(10 * 60, 18 * 60 + 50, 10), unit="min")
    for j, t in enumerate(tk):
        r = np.random.default_rng([seed, j])
        ts = (days.values[:, None] + mins.values[None, :]).ravel()
        c = 100 * np.exp(np.cumsum(r.normal(0, 0.004, len(ts))))
        o = np.r_[c[0], c[:-1]]
        out[t] = pd.DataFrame(dict(timestamp=ts, open=o, high=np.maximum(o, c) * 1.001, low=np.minimum(o, c) * 0.999,
                                   close=c, volume=r.integers(100, 1000, len(ts))))
    im = pd.DataFrame({"OPEN": 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, .01, dni))) }, index=days)
    im["CLOSE"] = im["OPEN"] * 1.001
    return out, im


@pytest.fixture(scope="module")
def D():
    raw, im = bary()
    return gv.iz_barov(raw, im, "train4", TK)


def test_centry_csv_sovpadaet_s_kodom():
    t = pd.read_csv(ROOT / "docs/tables/golaya_v7/centry.csv", dtype=str).set_index("sid")
    for s in ("S7", "S8", "S9"):
        c = gv.CENTR[s]
        assert (t.loc[s, "rebal"], float(t.loc[s, "porog"]), float(t.loc[s, "bufer"]), int(t.loc[s, "cap"])) == \
            (c["rezhim"], c["porog"], c["bufer"], c["cap"])
    for s in ("S10", "S11", "S12"):
        assert (int(t.loc[s, "tau"]), t.loc[s, "vhod"], t.loc[s, "t_sig"]) == (2, "d1", "18:30")


def test_chtenie_ne_dalshe_2020(monkeypatch):
    raw, im = bary(dni=60, start="2016-03-01", tk=gv.TICKERS)
    kontsy = []

    def fake(path, end=None):
        kontsy.append(end)
        return raw[Path(path).stem][lambda b: b["timestamp"] < end]

    monkeypatch.setattr(d18, "read_bars", fake)
    monkeypatch.setattr(gv, "chitat_imoex", lambda period: im)
    gv.zagruzit("v2")
    assert kontsy and all(e <= pd.Timestamp("2020-01-01") for e in kontsy)
    kontsy.clear()
    with pytest.raises(ValueError):
        gv.zagruzit("wf")
    assert not kontsy


def test_k2_lag0_ravno_iskhodnomu(D):
    for s in SID:
        a, b = gv.signaly(D, s), gv.signaly(D, s, ("k2", 0))
        assert np.array_equal(a, b), s
        m = lambda x: gv.stroka(gv.schet(D, x, **gv.shag(s)), D)["mSR"]  # noqa: E731
        assert m(a) == m(b) or (np.isnan(m(a)) and np.isnan(m(b)))
    assert not np.array_equal(gv.signaly(D, "S8", ("k2", 100)), gv.signaly(D, "S8"))


def _utechka(D, s, d, fn):
    """Сигнал дней ≤ d не должен зависеть от данных после d (M-d1)."""
    D2 = dict(D)
    D2["c"] = D["c"].copy()
    D2["c"].iloc[d + 1:] *= 1.5
    D2["cr"], D2["_sig"] = D2["c"], {}
    if s in ("S10", "S11", "S12"):
        D2["ist"] = {t: b.assign(close=np.where(b["date"] > D["daty"][d], b["close"] * 1.5, b["close"]),
                                 high=np.where(b["date"] > D["daty"][d], b["high"] * 1.6, b["high"])) for t, b in D["ist"].items()}
    a, b = fn(D, s)[:d + 1], fn(D2, s)[:d + 1]
    return np.array_equal(a, b)


def test_m_d1_signaly_ne_vidyat_budushchee(D):
    for s in SID:
        assert _utechka(D, s, 330, lambda x, y: gv.signaly(x, y)), s


def test_m_d1_myutaciya_lovitsya(D, monkeypatch):
    orig = kv.panel
    monkeypatch.setattr(kv, "panel", lambda c, im, m, sid: orig(c.shift(-1), im, m, sid))
    assert not _utechka(D, "S8", 330, lambda x, y: gv.signaly(x, y))


def test_ispolnitel_d_i_ceny_odnoy_shkaly(D):
    """Замечание 1: цены нескорректированные + D на день отсечки (после НДФЛ); сигналы S7–S9 — на скорректированных."""
    assert D["div_nalog"] == 0.87 and "нескорр" in gv.shapka(D, "train4")["ceny_i_D"]
    D2 = dict(D)
    D2["div"] = D["div"].copy()
    D2["div"].iloc[200, 0] = 5.0
    sig = np.zeros(D["c"].shape, np.int8)
    sig[198:200, 0] = 1
    a = gv.schet(D, sig, vhod="d1", tau=3)["pnl"]
    b = gv.schet(D2, sig, vhod="d1", tau=3)["pnl"]
    assert (b - a)[200] == pytest.approx(5.0 / D["o"].iloc[199, 0] / len(D["tk"]))
    assert np.abs(b - a)[np.arange(len(a)) != 200].max() < 1e-12
    b = gv.schet(D2, -sig, vhod="d1", tau=3)["pnl"]
    assert (b - gv.schet(D, -sig, vhod="d1", tau=3)["pnl"])[200] < 0


def test_exit_flagi_obertki_ne_vliyayut_pri_d1():
    """Замечание 2: флаги выхода обёртки S10–S12 (exit на d+τ у любого сигнала) при vhod=d1 и τ не меняют сделки."""
    r, T = np.random.default_rng(1), 120
    c = 100 * np.exp(np.cumsum(r.normal(0, .01, T)))
    o, dts = np.r_[c[0], c[:-1]], pd.bdate_range("2016-01-04", periods=T)
    for _ in range(20):
        sig = (r.random(T) < .2) * r.choice([-1, 1], T)
        ex = np.zeros(T, bool)
        idx = np.flatnonzero(sig) + 2
        ex[idx[idx < T]] = True
        kw = dict(sig=sig.astype(np.int8), o=o, c=c, vhod="d1", tau=2, dates=dts, rate=np.full(T, .07))
        a, b = ispolnit(**kw), ispolnit(**kw, exit_l=ex, exit_s=ex)
        assert a["sdelki"] == b["sdelki"] and np.array_equal(a["excess"], b["excess"])


def test_stroka_polna_i_diagnostika(D):
    s = gv.signaly(D, "S11", None, gv.shag("S11"))
    r = gv.stroka(gv.schet(D, s, **gv.shag("S11")), D)
    for k in ("mSR", "msr_ned", "beta", "dolya_dispersii_rynka", "sdelok_v_god", "pnl_pct_god", "kom_pct_god", "excess_pct_god",
              md.K_NORM, md.K_SDELOK, md.K_CENA, "mSR_2016"):
        assert k in r, k
    h = gv.shapka(D, "train4")
    for k in ("v", "status", "okno_s", "okno_po", "zaderzhka", "chestnaya", "kod", "vhody", "ceny_i_D"):
        assert h[k], k
    assert h["v"] == "7.0-gol" and h["chestnaya"] == "net"


def test_pustaya_lenta_i_znak(D):
    nul = np.zeros(D["c"].shape, np.int8)
    r = gv.stroka(gv.schet(D, nul, vhod="d1", tau=2), D)
    assert r["sdelok_v_god"] == 0 and not np.isfinite(r["mSR"])
    s = gv.signaly(D, "S11", None, gv.shag("S11"))
    p, m = (gv.schet(D, x, **gv.shag("S11")) for x in (s, -s))
    assert np.allclose(p["pnl"] + m["pnl"], 0, atol=1e-10) and p["kom"].sum() == pytest.approx(m["kom"].sum())


def test_kontroli_20_i_p(D):
    for s in ("S8", "S11"):
        k2 = gv.kontrol(D, s, "k2")
        pl = gv.kontrol(D, s, "placebo")
        assert len(k2) == len(pl) == 20
        assert not np.allclose(k2, pl)


def test_spravka_kniga_ravna_ref(D):
    """G5 на синтетике: справочная книга == golaya_s7_s9.schet/metriki на тех же входах (на реальных — тест ниже)."""
    r = {"c": D["c"], "o": D["o"], "d": D["div"], "im": D["im"]}
    sig = ref.sigmy(D["c"]).to_numpy()
    for s in ("S7", "S8", "S9"):
        o_ref = ref.schet(s, ref.skory(r, D["metki"])[s].to_numpy(), r, D["metki"], sig, D["stavka"], D["i0"])
        o = gv.kniga_spravka(D, s)
        for k in ("excess", "pnl", "kom", "zaem", "stavka"):
            assert np.array_equal(o[k], o_ref[k]), (s, k)


def test_spravka_kniga_ravna_svod_csv_train4():
    f = ROOT / "docs/tables/golaya_s7_s9/svod.csv"
    if not (d18.default_main() / "moex_discovery/data/raw/candles_10m/AFLT.parquet").exists():
        pytest.skip("нет 10-минуток (< 2020) на этой машине")
    D = gv.zagruzit("train4")
    sv = pd.read_csv(f).set_index("strategiya")
    for s in ("S7", "S8", "S9"):
        assert abs(gv.stroka(gv.kniga_spravka(D, s), D)["mSR"] - sv.loc[s, "mSR"]) <= 1e-12, s
