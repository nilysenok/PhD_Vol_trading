"""Л6: счёт под замком (`otkrytie_raschet.run_nodes`) на синтетических узлах: сверка архивных строк (допуск 1e-12, иначе стоп), ↺-сверки календаря, запечатанной RV,
единиц объёма и n; новые строки отдельно от архивных; таблицы из общих пар. Реальные узлы и данные не вызываются."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/fwd"))
import otkrytie_raschet as RS  # noqa: E402

AE = pd.Timestamp("2026-02-27")


class World:
    def __init__(self, seed=5, lots=1.0):
        rng = np.random.default_rng(seed)
        self.days = pd.bdate_range("2026-02-16", "2026-04-30")
        n = len(self.days)
        rv = np.exp(rng.normal(0, 0.4, n))
        self.daily = pd.DataFrame({"rv": rv}, index=self.days)
        d, t, r = self.days[:-1], self.days[1:], rv[1:]
        f = lambda s: r * np.exp(rng.normal(0, s, n - 1))  # noqa: E731
        self.t = dict(
            k1=pd.DataFrame(dict(d=d, target_date=t, rv=r, sigma2=f(0.8), gap=False)),
            k23=pd.DataFrame(dict(date=d, t_plus=t, rv=r, sigma2_k2=f(0.5), sigma2_k3=f(0.5))),
            cfg8=pd.DataFrame(dict(d=d, t_plus=t, rv=r, F=f(0.4))),
            x1=pd.DataFrame(dict(d=d, t_plus=t, rv=r, M=f(0.45), C=f(0.4), s=(rng.random(n - 1) < 0.2).astype(int), F=f(0.35))),
            zb=pd.DataFrame(dict(d=d, t_plus=t, rv=r, F=f(0.3))), zb0=pd.DataFrame(dict(d=d, t_plus=t, rv=r, F=f(0.31))))
        self.lots, self.fz = lots, dict(calendar=dict(days=[str(x.date()) for x in self.days if x > AE], n_new=int((self.days > AE).sum())))
        self.arch = {k: v.copy() for k, v in self.t.items()}
        self.sealed = pd.DataFrame(dict(date=self.days[self.days > AE], rv=rv[self.days > AE]))

    def archive(self, name):
        v = self.arch[name]
        col = "target_date" if name == "k1" else "t_plus"
        return v[v[col] <= AE].reset_index(drop=True)

    def nodes(self):
        w, t = self, self.t
        src = types.SimpleNamespace(sealed_rv=lambda: w.sealed, bars=lambda inst, cut: w.bars())
        tab = pd.DataFrame(dict(K2=t["k23"]["sigma2_k2"].to_numpy(), K3=t["k23"]["sigma2_k3"].to_numpy()), index=pd.DatetimeIndex(t["k23"]["date"]))
        return types.SimpleNamespace(
            Src=lambda: src, AE=AE, LAST=w.days[-1], daily=lambda cut, s: w.daily, k1=lambda cut, s: t["k1"], k23=lambda cut, s, dates: t["k23"] if isinstance(dates, str) else t["k23"][t["k23"]["date"].isin(dates)].reset_index(drop=True),
            store=lambda cut, s: {}, xw=lambda store: {}, states=lambda cut, s, store: {}, inputs=lambda cut, s, k23, store, xw: dict(tab=tab), cfg8=lambda cut, s, inp: t["cfg8"],
            x1=lambda cut, s, cfg8, tab_, st: t["x1"], zb=lambda cut, s, x1, st, store, kind: t["zb" if kind == "ZB" else "zb0"], archive=w.archive)

    def bars(self):
        b = np.repeat(self.days, 2) + pd.to_timedelta([10, 18] * len(self.days), unit="h")
        px = np.full(len(b), 300.0)
        return pd.DataFrame(dict(begin=b, close=px, volume=np.full(len(b), 1000.0), value=px * 1000.0 * self.lots))


def go(w):
    return RS.run_nodes(None, w.fz, w.nodes())


def status(res):
    return dict(zip(res["tables"]["checks"]["check"], res["tables"]["checks"]["status"]))


def test_chestnyj_progon_novye_stroki_otdelno_ot_arkhivnykh():
    w = World()
    res = go(w)
    assert set(status(res).values()) == {"ok"} and len(status(res)) == 10
    for k, df in res["new"].items():
        assert (df["target_date" if k == "k1" else "t_plus"] > AE).all() and len(df) == w.fz["calendar"]["n_new"]
    fam = res["tables"]["family"]
    assert len(fam) == 7 and (fam["n"] == w.fz["calendar"]["n_new"]).all()
    assert fam["tp_first"].iloc[0] == "2026-03-02" and fam["tp_last"].iloc[0] == str(w.days[-1].date())
    assert int(res["tables"]["states"].set_index("state").loc["E", "n"]) == int(w.t["x1"]["s"][w.t["x1"]["t_plus"] > AE].sum())


def test_kalendar_raskhozhdenie_eto_pomet_a_ne_stop():
    w = World()
    w.fz["calendar"]["days"] = w.fz["calendar"]["days"][1:] + ["2026-12-31"]  # потеряна первая дата и лишняя последняя
    st = status(go(w))
    k = "календарь: полный build_daily против замороженного"
    assert st[k] == "↺" and all(v == "ok" for c, v in st.items() if c != k and not c.startswith("число пар"))


def test_zapechatannaya_rv_edinitsy_obyoma_i_n():
    w = World()
    w.sealed.loc[3, "rv"] *= 1.001
    assert status(go(w))["RV новых дней против rv_fwd_SEALED"] == "↺"
    assert status(go(World(lots=10.0)))["единицы объёма новых файлов"] == "↺"  # лоты вместо штук: value/volume ÷ close = 10
    w3 = World()
    w3.fz["calendar"]["n_new"] += 1
    assert status(go(w3))["число пар n против замороженного"] == "↺"


@pytest.mark.parametrize("name", ["k1", "cfg8", "x1", "zb", "zb0"])
def test_arkhivnaya_stroka_ne_ravna_zamorozhennoj_stop(name):
    w = World()
    col = "sigma2" if name == "k1" else "F"
    w.arch[name].loc[2, col] *= 1 + 1e-6
    with pytest.raises(RS.ArchiveMismatch):
        go(w)
    w.arch[name].loc[2, col] /= 1 + 1e-6
    w.arch[name].loc[2, col] *= 1 + 1e-14  # допуск 1e-12 (torch/XGBoost/LightGBM): проходит
    assert set(status(go(w)).values()) == {"ok"}


@pytest.mark.parametrize("pos", [0, 3, -1])
def test_k23_lyubaya_arkhivnaya_stroka_lovitsya(pos):
    """Расхождение в первой, средней или последней архивной строке K2/K3 — стоп (не только последние дни)."""
    w = World()
    idx = w.arch["k23"].index[w.arch["k23"]["t_plus"] <= AE]
    w.arch["k23"].loc[idx[pos], "sigma2_k3"] *= 1.001
    with pytest.raises(RS.ArchiveMismatch):
        go(w)


def test_k23_sdvig_daty():
    w = World()
    w.arch["k23"].loc[0, "t_plus"] += pd.Timedelta(days=1)
    with pytest.raises(RS.ArchiveMismatch):
        go(w)
    w2 = World()
    w2.arch["x1"].loc[4, "t_plus"] = w2.arch["x1"].loc[4, "t_plus"] + pd.Timedelta(days=1)  # дата t⁺ — нефлот, допуска нет
    with pytest.raises(RS.ArchiveMismatch):
        go(w2)


def test_obshchie_pary_dyra_v_odnom_ryade_umenshaet_n():
    w = World()
    i = int(np.flatnonzero((w.t["k1"]["target_date"] > AE).to_numpy())[5])
    w.t["k1"].loc[i, "sigma2"] = np.nan
    res = go(w)
    assert (res["tables"]["family"]["n"] == w.fz["calendar"]["n_new"] - 1).all()
    assert status(res)["число пар n против замороженного"] == "↺"


def test_compare_nan_ne_na_meste_i_forma():
    a = pd.DataFrame(dict(x=[1.0, np.nan]))
    assert RS.compare("t", a, a.copy())["status"] == "ok"
    with pytest.raises(RS.ArchiveMismatch):
        RS.compare("t", a, pd.DataFrame(dict(x=[1.0, 2.0])))
    with pytest.raises(RS.ArchiveMismatch):
        RS.compare("t", a, a.iloc[:1])
