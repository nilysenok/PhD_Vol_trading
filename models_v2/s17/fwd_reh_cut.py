"""Л5: одна обрезка репетиции — все узлы дня d по данным <= d 18:40 против архива (протокол, раздел 7). Только замороженный код через модули Л1–Л4.

Режим `obrezka`: K2/K3 строк < d — сохранённый `k23.parquet` (побитно равен обрезке, L2_K23_CUTS38.csv), cfg8 строк < d — архив; строка d, блок cfg8 месяца,
X1, ZB, ZB0, z, e_d, s_d, рыночные признаки — с нуля по обрезанным данным. Режим `skvoznaya`: K2/K3 всех строк <= d и cfg8 всех строк (полный ход) считаются с нуля,
архивных промежуточных файлов нет. Новых цен не читает (cut <= 2026-02-27).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_k23_calc as KC, hp_k23_world as KW
from models_v2.s11.v4_ntp_state import qlike
from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1m_feat as FE, v7_1m_wf as WF, v7_3_cfg as C8
from models_v2.s15 import de_e1_state as E1
from models_v2.s17 import fwd_cal as FC, fwd_cfg8 as F8, fwd_k23 as K, fwd_store as FS, fwd_x1 as X1, fwd_zb as ZB
from models_v2.s17.fwd_reh_ref import EXPECT_REFIT, TOL, Ref, rec
from models_v2.s17.fwd_src import Src, limit_of

SKIP = ["rv", "d", "t_plus"]


def vals(df: pd.DataFrame, cols: list) -> np.ndarray:
    return df[cols].to_numpy(float)


def month_changes(df: pd.DataFrame, cols: list, n: int = 200) -> tuple:
    """(даты изменения параметров в последних n строках; те из них, что — первые дни месяцев индекса): равны, если переоценка только в первый день месяца."""
    t = df.tail(n).reset_index(drop=True)
    ch = [t["d"].iloc[i] for i in range(1, len(t)) if not np.array_equal(vals(t.iloc[[i]], cols), vals(t.iloc[[i - 1]], cols), equal_nan=True)]
    first = set(df["d"][~df["d"].dt.to_period("M").duplicated()])
    return np.array(ch, "datetime64[ns]"), np.array([x for x in ch if x in first], "datetime64[ns]")


def avail(c, src: Src) -> tuple:
    """Максимальный begin бара основной сессии каждого из 18 инструментов (17 бумаг пула + IMOEX), вошедшего в признаки дня d: все <= d 18:40."""
    mx = {t: D.main_session(src.bars(t, c))["begin"].max() for t in KW.POOL + ["IMOEX"]}
    ok = [bool(pd.notna(v) and v <= limit_of(c)) for v in mx.values()]
    stale = [t for t, v in mx.items() if pd.isna(v) or v.normalize() < pd.Timestamp(c)]
    return ok, "max begin: " + "; ".join(f"{t} {v}" for t, v in mx.items()) + f"; без бара в день d (NaN): {len(stale)} {stale}"


class Cut:
    """Контекст обрезки: копит строки сверки."""

    def __init__(self, c, src: Src, R: Ref, through: bool, procs: int):
        self.c, self.src, self.R, self.through, self.procs = pd.Timestamp(c), src, R, through, procs
        self.mode, self.rows, self.t0 = ("skvoznaya" if through else "obrezka"), [], time.time()
        self.w = K.build_world_at(self.c, src)
        self.tp = pd.Timestamp(K.next_day(pd.Timestamp(self.w["F"].dates[-1]), src))

    def add(self, node, got, ref, tol=0.0, detail=""):
        self.rows.append(rec(self.c, self.tp, node, got, ref, tol, self.mode, detail))

    def at(self, df, col):
        return df.loc[self.c, col]

    def base(self) -> tuple:
        """Календарь, доступность, рыночные признаки, K1. -> (k1, daily)."""
        c, R = self.c, self.R
        cal = FC.build_cal(c, self.src)
        self.add("cal_t_plus", self.tp, self.at(R.har, "target_date"))
        self.add("cal_days", cal["date"].to_numpy("datetime64[ns]"), R.days.to_numpy("datetime64[ns]")[R.days <= c])
        ok, det = avail(c, self.src)
        self.add("avail", ok, [True] * len(ok), 0.0, det)
        self.add("mkt17", self.w["dz0"].X[-1, -17:], R.dz0_row(c)[-17:], 0.0, "BASE 11 + IDX 6 против строки dz0 архива")
        self.add("dz0_full", self.w["dz0"].X[-1], R.dz0_row(c))
        k1, daily = F8.k1_at(c, self.src)
        self.add("K1", k1["sigma2"].iloc[-1], self.at(R.har, "sigma2"))
        self.add("prefix_K1", k1["sigma2"].iloc[:-1].to_numpy(float), R.har["sigma2"].reindex(pd.DatetimeIndex(k1["d"].iloc[:-1])).to_numpy(float))
        return k1, daily

    def k23(self) -> pd.DataFrame:
        """K2, K3, И2, M дня d; рамка K2/K3 для cfg8."""
        c, R, wp, i2 = self.c, self.R, K.pad_world(self.w, self.tp), K.load_i2()
        ds = None if self.through else pd.DatetimeIndex([c])
        fr = KC.forecast_frame(wp, i2, KC.select_rows(wp, min(i2), ds), self.procs)
        one = fr[fr["date"] == c]
        row = one.iloc[0]
        self.add("K2", row["sigma2_k2"], self.at(R.k23, "sigma2_k2"), TOL, "xgboost, допуск 1e-12 отн.")
        self.add("K3", row["sigma2_k3"], self.at(R.k23, "sigma2_k3"), TOL, "lightgbm, допуск 1e-12 отн.")
        hp, tx = ("hp_year", "hp_k2", "hp_k3"), lambda v: str(int(v)) if isinstance(v, (int, float, np.number)) else str(v)
        self.add("I2", [tx(row[k]) for k in hp], [tx(self.at(R.k23, k)) for k in hp], 0.0, "И2 из k23_meta_SBER.json, не пересчитывался")
        self.add("M", np.sqrt(row["sigma2_k2"] * row["sigma2_k3"]), self.at(R.x1, "M"), TOL)
        if self.through:
            pre = fr[fr["date"] < c]
            for k in ("k2", "k3"):
                self.add("prefix_" + k.upper(), pre[f"sigma2_{k}"].to_numpy(float), R.k23[f"sigma2_{k}"].reindex(pd.DatetimeIndex(pre["date"])).to_numpy(float), TOL, "строки < d с нуля")
            return fr
        sv = F8.mask_k23(pd.read_parquet(F8.K23), c)
        return pd.concat([sv[sv["date"] < c], one], ignore_index=True)

    def cfg8(self, tab, zm) -> pd.DataFrame:
        """cfg8: строка d, λ блока по seeds, дата переоценки; skvoznaya — полный ход, obrezka — блок месяца + архивные строки < d."""
        c, R = self.c, self.R
        d0 = tab.index[W6.refit_positions(tab.index, WF.FIRST_REFIT)[-1]]
        self.add("refit_cfg8", str(d0.date()), EXPECT_REFIT[(c.year, c.month)], 0.0, "первый день месяца по индексу (раздел 3)")
        if self.through:
            cfg, per = C8.run_cfg(tab, FE.build_Z(tab).to_numpy(float), lambda s: zm[s], F8.SEEDS)
            lam = np.array([[f["lam"] for f in p["fits"] if pd.Timestamp(f["d0"]) == d0][0] for p in per], float)
            self.add("prefix_cfg8", vals(cfg.iloc[:-1], list(cfg.columns[3:])), vals(R.cfg8.loc[R.cfg8["d"] < c], list(cfg.columns[3:])), TOL, "строки < d с нуля (полный ход)")
        else:
            blk = F8.build_cfg8(c, self.src, dict(tab=tab, zmap=zm))
            cfg, lam = pd.concat([R.cfg8[R.cfg8["d"] < c], blk.iloc[[-1]]], ignore_index=True), np.array(blk.attrs["lam"], float)
        cols = list(cfg.columns[3:])
        self.add("cfg8", vals(cfg.iloc[[-1]], cols)[0], vals(R.cfg8[R.cfg8["d"] == c], cols)[0], TOL, "torch, допуск 1e-12")
        self.add("cfg8_lam", lam, R.lam(d0), 0.0, "(λ_g, λ_LB) по seeds блока месяца: полный ход архива")
        return cfg

    def states(self, tab) -> tuple:
        """Склад, z, e_d, s_d. -> (store, st, zm)."""
        c, R = self.c, self.R
        store = FS.build_store(c, self.src)
        xw, st = FS.build_xw5(store), FS.build_states(c, self.src, store)
        zm = F8.z_maps(tab.index, c, self.src, F8.SEEDS, store, xw)
        self.add("z", np.array([zm[s][2026][-1] for s in F8.SEEDS]), R.z_at(c), TOL, "z дня d, 5 seeds, против артефакта tcn_w5_2026_s*; torch, допуск 1e-12")
        self.add("e_d", st["e"].iloc[-1], self.at(R.x1, "s"))
        self.add("s_d", st["s"].iloc[-1], self.at(R.states, "s"))
        self.add("prefix_es", vals(st.iloc[:-1], ["e", "s"]), vals(R.states.reindex(pd.DatetimeIndex(st["d"].iloc[:-1])), ["e", "s"]))
        return store, st, zm

    def x1(self, cfg, tab, st) -> pd.DataFrame:
        c, R = self.c, self.R
        x1, r = X1.run_x1(cfg, tab, st)
        cols, s = [k for k in X1.COLS if k not in SKIP], X1.state_of(r)
        self.add("X1", vals(x1.iloc[[-1]], cols)[0], vals(R.x1.loc[[c]], cols)[0], TOL)
        m = (R.x1["t_plus"] <= c).to_numpy()
        q = np.concatenate([qlike(R.x1["rv"].to_numpy()[m], R.x1[k].to_numpy()[m]) for k in ("C", "M")])
        self.add("X1_state", [s["pi0"], s["pi1"], s["eta"], s["a0"], s["a1"]], [self.at(R.x1, k) for k in ("pi0", "pi1")][:2] + [1.0 / max(float(np.median(q)), E1.EPS)] + [self.at(R.x1, k) for k in ("a0", "a1")], TOL, "π₀, π₁, η, α₀, α₁; η = 1/медиана QLIKE по закрытым парам")
        self.add("refit_x1_alpha", *month_changes(x1, ["a0", "a1"]), 0.0, "α меняется только в первый день месяца индекса")
        self.add("prefix_X1", vals(x1.iloc[:-1], cols), vals(R.x1.loc[R.x1["d"] < c], cols), TOL if self.through else 0.0)
        return x1

    def zb(self, x1, st, store) -> None:
        c, R = self.c, self.R
        for kind, a in (("ZB", R.zb), ("ZB0", R.zb0)):
            zb = ZB.build_zb(c, self.src, x1, st, store, kind)
            cols = [k for k in zb.columns if k not in SKIP]
            self.add(kind, vals(zb.iloc[[-1]], cols)[0], vals(a.loc[[c]], cols)[0], TOL)
            if kind == "ZB":
                z = ZB.state_of(zb)
                self.add("ZB_state", [z["lam"], z["b"], z["a"]], [self.at(a, k) for k in ("lam", "b", "a")], TOL, "λ, b, a")
            self.add("refit_" + kind.lower(), *month_changes(zb, ["lam", "b", "a"]), 0.0, "λ, b, a меняются только в первый день месяца индекса")
            self.add("prefix_" + kind, vals(zb.iloc[:-1], cols), vals(a.loc[a["d"] < c], cols), TOL if self.through else 0.0)


def run_cut(c, src: Src, R: Ref, through: bool = False, procs: int = 2) -> list:
    """Строки (день × узел) обрезки c. Расхождение — ok = False (причину разбирает человек, архив не подгоняется)."""
    k = Cut(c, src, R, through, procs)
    k1, daily = k.base()
    tab = F8.make_tab(k1, k.k23(), daily)
    store, st, zm = k.states(tab)
    x1 = k.x1(k.cfg8(tab, zm), tab, st)
    k.zb(x1, st, store)
    sec = float(time.time() - k.t0)
    k.rows.append(rec(k.c, k.tp, "info", sec, sec, 1e9, k.mode, "секунды обрезки (не сверка)"))
    return k.rows
