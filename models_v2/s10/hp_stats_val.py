"""P3, G11/G12: `HARJ_VALUE.csv` (У-15), `LEVELS.csv` (R14) и `LOTO.csv` (У-13). Всё — из PER_TICKER (по бумагам) и TESTS (панельные DM / бутстрап / Холм); новых счётов моделей нет. Δ = QLIKE(модель) − QLIKE(база), < 0 — модель лучше."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s10 import hp_paths as P, hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T

GROUPS = ("M16", "SBER", "E11", "all28")
P3 = ("y2022", "p2023_2025", "y2026")


def _grp(pt: pd.DataFrame, g: str) -> pd.DataFrame:
    return pt if g == "all28" else pt[pt["group"] == g]


def harj_value(pt: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Δ QLIKE гибрида к K4/K5 СВОЕГО модуля (добавленная ценность HAR-J); A1 — и к K1 (вклад поправки). Среднее, медиана по бумагам, доля бумаг Δ < 0, по срезам и группам; DM / ДИ — окна all и confirm."""
    q = pt.pivot_table(index=["ticker", "group", "period"], columns="model", values="qlike")
    rows = []
    for h in T.HYB:
        comps = [(CM.own_ctrl(h), "HARJ_value" if not h.endswith("_A1") else "A1_vs_own_K45 (вклад offset HAR-J при той же панели)")] + ([("k1", "A1_vs_K1 (вклад поправки)")] if h.endswith("_A1") else [])
        for b, role in comps:
            d = (q[h] - q[b]).dropna().reset_index(name="d")
            for g in GROUPS:
                for per, x in _grp(d, g).groupby("period"):
                    r = dict(config=h, comparator=b, role=role, scope=g, period=per, n_tickers=len(x), delta_mean=x["d"].mean(), delta_median=x["d"].median(), frac_tickers_better=float((x["d"] < 0).mean()))
                    t = tests[(tests["config"] == h) & (tests["comparator"] == b) & (tests["scope"] == g) & (tests["window"] == per)]
                    if len(t):
                        r.update({k: t.iloc[0][k] for k in ("dm_p", "boot_lo", "boot_hi")})
                    rows.append(r)
    return pd.DataFrame(rows)


def _row(tests: pd.DataFrame, cfg: str, comp: str, scope: str, win: str) -> pd.Series | None:
    t = tests[(tests["config"] == cfg) & (tests["comparator"] == comp) & (tests["scope"] == scope) & (tests["window"] == win)]
    return t.iloc[0] if len(t) else None


def l1(tests: pd.DataFrame, cfg: str, comp: str, scope: str, win: str, holm: bool, pref: str = "") -> dict:
    """Условия уровня 1 к базе comp: Δ < 0; знак Δ < 0 в ≥ 2 из 3 периодов; DM p < 0,05; ДИ бутстрапа Δ < 0 (верхняя граница < 0); p Холма < 0,10 (иначе «только сырой»).
    pref — префикс окон (`k23_` для K2/K3: общие даты с ними; win и периоды берутся с префиксом)."""
    r = _row(tests, cfg, comp, scope, pref + win)
    if r is None:
        return dict(ok=False, raw=False)
    signs = [(_row(tests, cfg, comp, scope, pref + p) is not None and _row(tests, cfg, comp, scope, pref + p)["delta_mean"] < 0) for p in P3]
    raw = bool(r["delta_mean"] < 0 and sum(signs) >= 2 and r["dm_p"] < 0.05 and r["boot_hi"] < 0)
    return dict(ok=bool(raw and (not holm or r["p_holm"] < 0.10)), raw=raw, delta=r["delta_mean"], dm_p=r["dm_p"], boot_hi=r["boot_hi"], p_holm=r.get("p_holm"), n_neg_periods=int(sum(signs)))


def levels(pt: pd.DataFrame, tests: pd.DataFrame, loto_cfgs: set) -> pd.DataFrame:
    """R14 по каждому гибриду и области доказательств: confirm (M16 × 2022–02.2026) и hist (M16 × вся история). Уровень 2 — к K1, K4, K5 и K2, K3 (R14; K2/K3 — на общих датах с ними, окна `k23_*`, G15); 4 — E11 и LOTO."""
    rows = []
    for h in T.HYB:
        for ev, w in (("confirm_M16_2022_02.2026", "confirm_2022_2026"), ("hist_M16_2017_2026", "all")):
            a = l1(tests, h, "k1", "M16", w, True)
            c4, c5 = l1(tests, h, "k4", "M16", w, True), l1(tests, h, "k5", "M16", w, True)
            c2, c3 = l1(tests, h, "k2", "M16", w, True, CM.K23_PREF), l1(tests, h, "k3", "M16", w, True, CM.K23_PREF)
            l2 = a["ok"] and c4["ok"] and c5["ok"] and c2["ok"] and c3["ok"]       # G15: к K1, K4, K5 и K2, K3 (R14)
            n240, n5 = _row(tests, h, "k1", "M16", "no_240222"), _row(tests, h, "k1", "M16", "no_top5_adv")
            ptw = pt[(pt["group"] == "M16") & (pt["period"] == w)].pivot_table(index="ticker", columns="model", values="qlike")
            frac = float((ptw[h] < ptw["k1"]).mean())
            l3 = bool(l2 and n240 is not None and n240["delta_mean"] < 0 and n5 is not None and n5["delta_mean"] < 0 and frac >= 0.6)
            e11 = _row(tests, h, "k1", "E11", w)
            lt = _row(tests, h + "__LOTO", "k1", "M16", w) if h in loto_cfgs else None
            e_ok = bool(e11 is not None and e11["delta_mean"] < 0 and e11["dm_p"] < 0.05)
            lt_ok = bool(lt is not None and lt["delta_mean"] < 0 and lt["dm_p"] < 0.05)
            lvl = 4 if (l3 and e_ok and lt_ok) else 3 if l3 else 2 if l2 else 1 if a["ok"] else 0
            rk2 = _row(tests, h, "k2", "M16", CM.K23_PREF + w)
            rows.append(dict(config=h, evidence=ev, level=lvl, n_k23_tickers_m16=None if rk2 is None else rk2.get("n_k23_tickers"), l1_ok=a["ok"], l1_raw_only=bool(a["raw"] and not a["ok"]), d_k1=a.get("delta"), p_dm_k1=a.get("dm_p"), p_holm_k1=a.get("p_holm"), n_neg_periods=a.get("n_neg_periods"),
                             l2_k4=c4["ok"], l2_k5=c5["ok"], l2_k2=c2["ok"], l2_k3=c3["ok"], d_k2_m16=c2.get("delta"), d_k3_m16=c3.get("delta"), p_dm_k2=c2.get("dm_p"), p_dm_k3=c3.get("dm_p"),
                             p_holm_k2=c2.get("p_holm"), p_holm_k3=c3.get("p_holm"), l3_no240222=None if n240 is None else bool(n240["delta_mean"] < 0), l3_no_top5=None if n5 is None else bool(n5["delta_mean"] < 0), frac_m16_better_k1=frac,
                             l4_e11=e_ok, l4_loto=lt_ok if lt is not None else "не считалось", status="потолок; ACCEPTED не присваивается"))
    return pd.DataFrame(rows)


def loto_table(c: pd.DataFrame, loto: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """LOTO.csv: по бумагам — QLIKE LOTO, пула, K1; Δ к пулу и к K1; DM (НУ 5, описательно); по группам — среднее, медиана Δ, панельные DM / ДИ из TESTS."""
    rows = []
    for cfg, g in loto.groupby("config"):
        x = c.merge(g[S.KEY + ["sigma2"]].rename(columns={"sigma2": "loto"}), on=S.KEY, how="inner")
        x = x[np.isfinite(x["loto"]) & (x["loto"] > 0)]
        for tk, y in x.groupby("ticker"):
            q = {s: CM.qrow(y, s) for s in ("loto", cfg, "k1")}
            r = dict(config=cfg, scope="ticker", name=tk, group=y["group"].iloc[0], n=len(y), q_loto=q["loto"].mean(), q_pool=q[cfg].mean(), q_k1=q["k1"].mean())
            r.update(d_vs_pool=r["q_loto"] - r["q_pool"], d_vs_k1=r["q_loto"] - r["q_k1"], dm_p_vs_pool=dmod.dm(y["rv_tplus"], y[cfg], y["loto"])["p"], dm_p_vs_k1=dmod.dm(y["rv_tplus"], y["k1"], y["loto"])["p"])
            rows.append(r)
    t = pd.DataFrame(rows)
    gr = []
    for cfg, g in t.groupby("config"):
        for name in GROUPS:
            s = _grp(g.rename(columns={"group": "group"}), name)
            r = dict(config=cfg, scope="group", name=name, n=int(s["n"].sum()), n_tickers=len(s), q_loto=s["q_loto"].mean(), q_pool=s["q_pool"].mean(), q_k1=s["q_k1"].mean(), d_vs_pool=s["d_vs_pool"].mean(), d_vs_k1=s["d_vs_k1"].mean(),
                     d_vs_pool_median=s["d_vs_pool"].median(), d_vs_k1_median=s["d_vs_k1"].median(), frac_loto_better_k1=float((s["d_vs_k1"] < 0).mean()), frac_loto_better_pool=float((s["d_vs_pool"] < 0).mean()))
            for comp, key in (("k1", "k1"), (cfg + "(пул)", "pool")):
                p = _row(tests, cfg + "__LOTO", comp, name, "all")
                if p is not None:
                    r.update({"dm_p_vs_" + key: p["dm_p"], "boot_lo_vs_" + key: p["boot_lo"], "boot_hi_vs_" + key: p["boot_hi"]})
            gr.append(r)
    return pd.concat([pd.DataFrame(gr), t], ignore_index=True)


def write(hv: pd.DataFrame, lv: pd.DataFrame, lt: pd.DataFrame) -> None:
    P.write_csv(hv, "statistics/HARJ_VALUE.csv", "добавленная ценность HAR-J: Δ QLIKE гибрида к K4/K5 СВОЕГО модуля; A1 — также к K1; по бумагам (среднее, медиана, доля Δ < 0), срезам и группам; %s" % S.STAT)
    P.write_csv(lv, "statistics/LEVELS.csv", "уровни R14 (уровень 2 к K1, K4, K5 и K2, K3 — G15, K2/K3 на 28 бумагах, общие даты с ними); evidence: confirm (M16 × 2022–02.2026), hist (M16 × вся история); %s" % S.STAT)
    P.write_csv(lt, "statistics/LOTO.csv", "LOTO (У-13): по бумагам и группам — LOTO против пула и K1, DM (НУ 5, описательно; панельный — в TESTS); %s" % S.STAT)
