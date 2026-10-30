"""Тестировщик PR #881: проверка починки голых идей S7–S12 на реальных 10-минутках (< 2020) и мутации (monkeypatch).

Мутации: М1 корпсобытия после агрегации (10-минутки и open 10:00 без поправки); М2 плацебо снова ежедневное; М3 к2 с прогревом.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7 as gv  # noqa: E402
from fastbt import data as fd, dividendy, kross_v7 as kv  # noqa: E402
from test_golaya_v7 import TK, bary  # noqa: E402
from test_golaya_v7_anomalii import _s_konsolidaciej  # noqa: E402
from test_golaya_v7_pochinka import _smen  # noqa: E402

SRC = ROOT / "moex_discovery" / "data" / "raw" / "candles_10m"
REAL = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "SBER"]
END = pd.Timestamp("2020-01-01")
nodata = pytest.mark.skipif(not (SRC / "IRAO.parquet").exists(), reason="нет 10-минуток в рабочей копии")


def _skachok(D, t, data):
    i = int(np.flatnonzero(D["daty"] == data)[0])
    return max(abs(np.log(D[k][t].to_numpy()[i] / D[k][t].to_numpy()[i - 1])) for k in ("o", "cr", "c"))


@pytest.fixture(scope="module")
def real():
    pytest.importorskip("pyarrow")
    if not (SRC / "IRAO.parquet").exists():
        pytest.skip("нет данных")
    raw = {}
    for t in REAL:
        b = pq.read_table(SRC / f"{t}.parquet", filters=[("timestamp", "<", END.to_pydatetime())]).to_pandas()
        raw[t] = b[b["timestamp"] >= gv.PERIODY["v2"].progrev]
    day = pd.bdate_range(gv.PERIODY["v2"].progrev, END - pd.Timedelta(days=1))
    im = pd.DataFrame({"OPEN": 100.0, "CLOSE": 100.0}, index=day)
    return gv.iz_barov(raw, im, "v2", REAL), raw


@nodata
def test_real_irao_konsolidaciya_kak_u_dvizhka(real):
    """IRAO 2014–2019: множество событий и день — как у движка (data.corp_actions); цены драйвера = дневка × factor до даты; скачка нет."""
    D, _ = real
    ev = fd.corp_actions("IRAO")
    assert list(ev["date"].dt.date.astype(str)) == ["2015-01-20"]
    assert _skachok(D, "IRAO", pd.Timestamp("2015-01-20")) < 0.3
    d = pq.read_table(ROOT / "data/fastbt/daily/ohlcv_daily_1850.parquet",
                      filters=[("date", "<", END.to_pydatetime())]).to_pandas()
    for t in ("IRAO", "LKOH", "AFLT"):
        x = d[d["ticker"] == t].assign(date=lambda z: pd.to_datetime(z["date"])).set_index("date").sort_index()["close"]
        for e in fd.corp_actions(t).itertuples():
            x[x.index < e.date] *= float(e.factor)
        j = D["cr"][t].dropna().to_frame("a").join(x.rename("b"), how="inner")
        assert len(j) > 1000 and np.allclose(j["a"], j["b"], rtol=1e-9), t


@nodata
def test_real_dividendy_na_den_otsechki_ne_zadety(real):
    """D = gap_divs × НДФЛ на тех же днях, что и без корпсобытий (№ 31): ни сдвига, ни умножения на factor."""
    D, _ = real
    for t in ("IRAO", "LKOH", "SBER"):
        dts = D["cr"][t].dropna().index.values
        ozhid = pd.Series(dividendy.gap_divs(t, dts) * dividendy.DIV_NALOG, dts).reindex(D["daty"]).fillna(0.0)
        assert np.allclose(D["div"][t].to_numpy(), ozhid.to_numpy()), t
    assert (D["div"]["LKOH"] > 0).any() and (D["div"]["IRAO"] > 0).any()


def test_placebo_sohranyaet_stolbcy_i_rvet_privyazku():
    """Плацебо — перестановка столбцов: ряд каждого столбца цел (оборот, автокорреляция), но тикер получает чужой ряд."""
    raw, im = bary()
    D = gv.iz_barov(raw, im, "train4", TK)
    c = gv.CENTR["S8"]
    sk = kv.rebal(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], "S8"), D["daty"], c["rezhim"])
    for seed in (1, 2, 3):
        pl = gv.placebo_ryad(sk, seed)
        perm = [next(k for k in range(sk.shape[1]) if np.array_equal(pl[:, j], sk[:, k], equal_nan=True)) for j in range(sk.shape[1])]
        assert sorted(perm) == list(range(sk.shape[1]))
        assert perm != list(range(sk.shape[1]))
        assert np.array_equal(np.isfinite(pl).sum(axis=0), np.isfinite(sk).sum(axis=0)[perm])


def test_placebo_menyaet_ne_tolko_otdelnyj_den():
    """Одна перестановка на ряд: строки разных дней переставлены одинаково (ежедневная перестановка эту связь рвёт)."""
    sk = np.tile(np.arange(8.0), (30, 1))
    pl = gv.placebo_ryad(sk, 5)
    assert (pl == pl[0]).all()


def _mut_korp_posle_agregacii(monkeypatch):
    orig = gv.d18.aggregate

    def agg(b, t):
        r = orig(b, t)
        for _, ev in fd.corp_actions(t).iterrows():
            m = r["date"] < ev["date"]
            r.loc[m, ["open", "high", "low", "close"]] *= float(ev["factor"])
        return r
    monkeypatch.setattr(gv, "korpsobytiya", lambda b, t: b)
    monkeypatch.setattr(gv.d18, "aggregate", agg)


def test_mutaciya_m1_korpsobytiya_posle_agregacii_lovitsya(tmp_path, monkeypatch):
    raw, im, t, data = _s_konsolidaciej(tmp_path, monkeypatch)
    assert _skachok(gv.iz_barov(raw, im, "train4", TK), t, data) < 0.2
    _mut_korp_posle_agregacii(monkeypatch)
    assert _skachok(gv.iz_barov(raw, im, "train4", TK), t, data) > 1.0     # open 10:00 и сигнальный ряд 10-минуток — со скачком


def test_mutaciya_m2_placebo_ezhednevnoe_lovitsya(monkeypatch):
    raw, im = bary()
    D = gv.iz_barov(raw, im, "train4", TK)
    for sid in ("S7", "S8", "S9"):
        c = _smen(gv.signaly(D, sid))
        assert max(_smen(gv.signaly(D, sid, ("placebo", gv.SEED + s))) for s in range(3)) <= 2 * c
    monkeypatch.setattr(gv, "placebo_ryad", kv.placebo_sechenie)
    assert max(_smen(gv.signaly(D, "S8", ("placebo", gv.SEED + s))) for s in range(3)) > 2 * _smen(gv.signaly(D, "S8"))


@pytest.mark.parametrize("sid", ["S7", "S8", "S9"])
def test_mutaciya_m3_k2_s_progrevom_lovitsya(monkeypatch, sid):
    raw, im = bary()
    D = gv.iz_barov(raw, im, "train4", TK)
    c = gv.CENTR[sid]
    sk, w = kv.rebal(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], sid), D["daty"], c["rezhim"]), D["win"]
    lags = [660, 700, 800]
    dolya = lambda x: np.isfinite(x[w]).any(axis=1).mean()  # noqa: E731
    assert all(dolya(gv.skor_k2(D, sk, g)) >= dolya(sk) - 1e-9 for g in lags)
    monkeypatch.setattr(gv, "skor_k2", lambda D_, s, g: kv.k2(s, g))
    assert any(dolya(gv.skor_k2(D, sk, g)) < dolya(sk) - 0.01 for g in lags)
