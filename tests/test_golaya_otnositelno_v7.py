"""Л2.2 (Ж-146 п. 2): голая идея S1–S4 × otnositelno ∈ {net, sektor, imoex} — тесты на синтетике (прогон не делается, сеть закрыта).

G1 (выкл.): net == центр без оси побайтно. G2: чтение < 2020. M-d1: фон дня d+1 в сигнале d ловится.
Контроли (k2, плацебо) двигают только входы; выход — правило семьи (флаги exit), как есть.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "server")]
import golaya_otnositelno_v7 as go  # noqa: E402
import golaya_tf_v5 as gtf  # noqa: E402
import golaya_v7 as gv  # noqa: E402
from fastbt import daily_1850 as d18, otnositelno_v7 as ot  # noqa: E402
from fastbt.indicators import compute_base  # noqa: E402
from fastbt.signals.interface import get_family  # noqa: E402

TK = gv.TICKERS[:8]
SIDS = go.SIDS


def bary(seed=0, dni=420, start="2015-03-02", tk=TK):
    out, days = {}, pd.bdate_range(start, periods=dni)
    mins = pd.to_timedelta(np.arange(10 * 60, 18 * 60 + 50, 10), unit="min")
    for j, t in enumerate(tk):
        r = np.random.default_rng([seed, j])
        ts = (days.values[:, None] + mins.values[None, :]).ravel()
        c = 3000 * np.exp(np.cumsum(r.normal(0, 0.004, len(ts))))
        o = np.r_[c[0], c[:-1]]
        out[t] = pd.DataFrame(dict(timestamp=ts, open=o, high=np.maximum(o, c) * 1.001, low=np.minimum(o, c) * 0.999,
                                   close=c, volume=r.integers(100, 1000, len(ts))))
    im = pd.DataFrame({"OPEN": 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, .01, dni)))}, index=days)
    im["CLOSE"] = im["OPEN"] * 1.001
    return out, im


@pytest.fixture(scope="module")
def D():
    raw, im = bary()
    return go.iz_barov(raw, im, "train4", TK)


def test_centry_csv_ravny_centram_v5_i_kodu():
    t = pd.read_csv(ROOT / "docs/tables/golaya_otnositelno_v7/centry.csv", dtype=str).set_index("sid")
    for s in SIDS:
        assert go.CENTR[s] == gtf.tsentr(gtf.Z.OSI[s]), s
        assert t.loc[s, "otnositelno"] == "|".join(ot.OTNOSITELNO) and t.loc[s, "vhod"] == "d1"


def test_g1_net_pobajtno_ravno_centru_bez_osi(D):
    n_sig = 0
    for s in SIDS:
        S, EL, ES = go.signaly(D, s, "net")
        ref = np.zeros(S.shape, np.int8)
        for j, t in enumerate(D["tk"]):
            td, ix = D["td"][t]
            ss = get_family(s, "v5")(td, compute_base(td.close, td.high, td.low, td.open, td.volume), [go.CENTR[s]])
            ref[ix, j] = ss.sig[0]
        assert S.tobytes() == ref.tobytes(), s
        n_sig += int(np.abs(S).sum())
        a, b = go.stroka_ryad(D, S, EL, ES), go.stroka_ryad(D, ref, EL, ES)
        assert a == b or all(x == y or (x != x and y != y) for x, y in zip(a.values(), b.values()))
    assert n_sig > 0
    assert ot.semya(get_family("S1", "v5"), "net", None) is get_family("S1", "v5")


def test_schet_bez_flagov_ravno_gv_schet(D):
    S = go.signaly(D, "S4", "net")[0]
    a, b = go.schet(D, S), gv.schet(D, S, vhod="d1")
    assert all(np.array_equal(a[k], b[k]) for k in ("pnl", "kom", "zaem", "stavka", "excess", "gross", "vhody"))
    assert a["sd"] == b["sd"]


def test_flag_vyhoda_zakryvaet_poziciyu(D):
    S, EL, ES = (np.zeros(D["c"].shape, a) for a in (np.int8, bool, bool))
    S[250, 0], EL[255, 0] = 1, True
    sd = go.schet(D, S, EL, ES)["sd"]
    assert len(sd) == 1 and sd[0][2] == 5                      # вход open 251, выход open 256 по флагу дня 255
    assert go.schet(D, S)["sd"][0][2] > 5                      # без флага — последний close
    S[250, 0], EL[255, 0], ES[255, 0] = -1, False, True
    assert go.schet(D, S, EL, ES)["sd"][0][2] == 5             # для шорта — exit_s


def test_chtenie_ne_dalshe_2020(monkeypatch):
    raw, im = bary(dni=60, start="2016-03-01", tk=gv.TICKERS)
    kontsy = []

    def fake(path, end=None):
        kontsy.append(end)
        return raw[Path(path).stem][lambda b: b["timestamp"] < end]

    monkeypatch.setattr(d18, "read_bars", fake)
    monkeypatch.setattr(gv, "chitat_imoex", lambda period: im)
    go.zagruzit("v2")
    assert kontsy and all(e <= pd.Timestamp("2020-01-01") for e in kontsy)
    kontsy.clear()
    with pytest.raises(ValueError):
        go.zagruzit("wf")
    assert not kontsy


def _udar(D, d):
    """Копия D, где всё после дня d изменено: цены тикеров, IMOEX; кэши сброшены."""
    D2 = dict(D)
    k = np.linspace(1.1, 0.92, len(D["tk"]))
    D2["c"] = D["c"].copy()
    D2["c"].iloc[d + 1:] *= k
    D2["im"] = D["im"].copy()
    D2["im"].iloc[d + 1:] *= 1.04
    D2["td"], D2["_sig"], D2["_fon"] = {}, {}, {}
    for j, t in enumerate(D["tk"]):
        td, ix = D["td"][t]
        m = (ix > d)
        D2["td"][t] = (td.__class__(**{**td.__dict__, **{a: getattr(td, a) * np.where(m, k[j], 1.0)
                                                         for a in ("open", "high", "low", "close")}}), ix)
    return D2


def _utechka(D, d):
    D2 = _udar(D, d)
    return all(np.array_equal(go.signaly(D, s, r)[0][:d + 1], go.signaly(D2, s, r)[0][:d + 1])
               for s in SIDS for r in ot.OTNOSITELNO)


def test_m_d1_signaly_ne_vidyat_budushchee(D):
    assert _utechka(D, 330)


def test_m_d1_myutaciya_fon_dnya_d_plus_1_lovitsya(D, monkeypatch):
    orig = ot.fon_ryad
    monkeypatch.setattr(ot, "fon_ryad", lambda c, im, m, r: tuple(x.shift(-1, fill_value=0.0 if x.dtypes.iloc[0] != bool else False)
                                                                  for x in orig(c, im, m, r)))
    D3 = dict(D, _sig={}, _fon={})
    assert not _utechka(D3, 330)


def test_k2_lag0_i_plasebo(D):
    s = "S1"
    a = go.signaly(D, s, "sektor")[0]
    assert np.array_equal(a, go.signaly(D, s, "sektor", ("k2", 0))[0])
    assert not np.array_equal(a, go.signaly(D, s, "sektor", ("k2", 100))[0])
    p = go.signaly(D, s, "sektor", ("placebo", 7))[0]
    assert np.abs(p).sum() == np.abs(a).sum() and not np.array_equal(a, p)
    assert np.array_equal(go.signaly(D, s, "sektor", ("k2", 0))[1], go.signaly(D, s, "sektor")[1])   # выход — правило семьи


def test_tablicy_pishut_12_strok_i_net_ravno_centru(D, tmp_path):
    go.tablicy(D, "train4", tmp_path, n=2)
    sv = pd.read_csv(tmp_path / "svod_train4.csv", comment="#")
    assert len(sv) == len(SIDS) * 3 and set(sv["otnositelno"]) == set(ot.OTNOSITELNO)
    for nm in ("k2", "placebo"):
        assert len(pd.read_csv(tmp_path / f"{nm}_train4.csv", comment="#")) == len(SIDS) * 3
    assert (tmp_path / "sdelki_train4.csv").read_text().startswith("# ")
    S, EL, ES = go.signaly(D, "S3", "net")
    ref = go.stroka_ryad(D, S, EL, ES)
    r = sv[(sv["strategiya"] == "S3") & (sv["otnositelno"] == "net")].iloc[0]
    assert r["mSR"] == pytest.approx(ref["mSR"], nan_ok=True)
