"""P1b, G8: контроли K4/K5 (панельные XGB/LGBM без HAR-J), K6 (HAR-J + E8, замороженная спецификация Ж-214), K7 (компонентная HAR, C1 при λ = 0). Не trials (CHECKPOINT used = 0).
Панель движка — `engine_panel.parquet`: та же сборка hp_panel.build, но K1 на сетке с первой даты, где есть >= 252 закрытых пар (а не с 03.01.2017), — иначе walk-forward «с 2016» (У-7) не имеет обучающих пар.
Общие строки (бумага, дата) побитно равны panel_base во всех колонках, кроме построенных из K1 (x_hdev, x_err*) — их z меняется (больше прошлого), см. `overlap_diff`; K1 на общих датах не меняется (OLS по парам s⁺ <= d)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from models_v2 import har as H, metrics
from models_v2.s10 import hp_engine as E, hp_feat as F, hp_har as HH, hp_paths as P, hp_panel as PN
from models_v2.s9 import hv2_dec as DC, hv2_mkt as MK, hv2_msfeat as MS, hv2_world as W

HAR_BASED = ("x_hdev", "x_err1", "x_err5", "x_err22")          # построены из прогноза HAR (R10: K4/K5 без них)
K45_FEATS = ["z_" + c for c in F.X_COLS + F.M_COLS + F.C_COLS if c not in HAR_BASED]
ENGINE_PANEL = P.OUT / "engine_panel.parquet"
E8_BLOCKS, E8_WIN = ("semicov", "idio"), 22


CONTROLS = ("K4", "K5", "K6", "K7")


def registry_control_rows(path) -> list:
    """Ключи строк реестра `ALL_TRIALS.csv` с именем K4…K7 (контроли — не trials, R10; Ж-215 справка). Пусто = ок."""
    out = []
    for ln in pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str):
        parts = ln.split(":")
        if len(parts) >= 3 and parts[2].upper() in CONTROLS:
            out.append(ln)
    return out


def used_errors(registry_path) -> list:
    """Проверка «контроли не в бюджете»: строк K4–K7 в реестре нет (после P2 `used` = 20 — гибриды, это не ошибка)."""
    rows = registry_control_rows(registry_path)
    return ["контроли K4–K7 в реестре %s: %s" % (registry_path.name if hasattr(registry_path, "name") else registry_path, rows)] if rows else []


def k1_ext(daily: dict) -> dict:
    """K1 бумаг на сетке с первой даты с >= 252 закрытых пар (HAR-J, как hp_har.series)."""
    return {t: HH.series(d, HH.grid(H.make_panel(d), first_target="2000-01-01")) for t, d in daily.items()}


def build_panel_ext(daily: dict, imo: pd.DataFrame) -> pd.DataFrame:
    return PN.build(daily, imo, k1_ext(daily))


def overlap_diff(ext: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Максимальное |ext − panel_base| по общим строкам (бумага, дата) для числовых колонок; отдельно — колонки K1-ошибок (x_err*)."""
    m = ext.merge(base, on=["ticker", "date"], suffixes=("", "_b"))
    cols = [c for c in base.columns if c not in ("ticker", "date") and base[c].dtype.kind == "f" and c in ext.columns]
    diff = {}
    for c in cols:
        a, b = m[c].to_numpy(float), m[c + "_b"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        diff[c] = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
    har = {c for h in HAR_BASED for c in (h, "z_" + h)}
    return {"n_common": len(m), "n_base": len(base), "max_by_col": diff, "max_non_har": max(v for k, v in diff.items() if k not in har),
            "max_har_based": max(diff.get(k, 0.0) for k in har)}


def k45_inputs(panel: pd.DataFrame) -> tuple:
    """(строки панели, список признаков, base = RV_m дня d): offset ln RV_m, без величин из прогноза HAR (R10)."""
    return panel, K45_FEATS, panel["rv_m_t"].to_numpy(float)


def run_k45(panel: pd.DataFrame, module: str, hp=None, nthread=6, log=None) -> tuple:
    """K4 (xgb) / K5 (lgbm): два прохода глубины, QLIKE, λ ≡ 1 (У-6), c по R8 (У-7). -> (прогнозы, выбор по годам, время по проходам)."""
    pn, feats, base = k45_inputs(panel)
    runs, secs = {}, {}
    for depth in E.DEPTHS:
        runs[depth] = E.walk_forward(pn, feats, base, "qlike", module, depth, hp=hp, nthread=nthread, log=log)
        secs[depth] = float(runs[depth].groupby("retrain")["fit_sec"].first().sum())       # время обучения: одно значение на месячную модель
    fc, ch = E.finalize(pn, base, runs, "qlike", lams=(1.0,))
    return fc, ch, secs


# --- K6: HAR-J + E8 на бумаге (замороженная спецификация Ж-214: log-OLS A0 + блоки semicov, idio, окно 22, без Mb) ---
def e8_world(d: pd.DataFrame, imo: pd.DataFrame, rows: np.ndarray) -> DC.DWorld:
    """Мир бумаги по её дневной таблице (hp_daily.ticker_daily) и дневному ряду IMOEX — как hv2_msfeat.build_world, но без перечитывания баров."""
    p = H.make_panel(d)
    w = W.World(p, d["rq"].to_numpy(float), d["rv_rel"].to_numpy(float), rows, None)
    dw = DC.build(w, d, MS.KEYS)
    ix = imo.reindex(d.index)
    return dataclasses.replace(dw, comp={**dw.comp, "imo_rv": ix["rv"].to_numpy(float), "imo_w": ix["rv_w"].to_numpy(float), "imo_m": ix["rv_m"].to_numpy(float)})


def k6_series(d: pd.DataFrame, imo: pd.DataFrame, k1: pd.DataFrame) -> pd.DataFrame:
    """σ̂² E8 на датах сетки K1 бумаги (d, target_date); день с нефинитным признаком — A0 (fb = 1), как в Ж-214."""
    p = H.make_panel(d)
    rows = np.flatnonzero(np.isin(d.index.to_numpy(), k1["d"].to_numpy()))
    s2, _, _, fb = MK.run(e8_world(d, imo, rows), False, E8_BLOCKS, E8_WIN)
    return pd.DataFrame({"d": p.dates[rows], "target_date": p.dates[rows + 1], "sigma2_k6": s2, "k6_fb": fb.astype(float)})


# --- K7: компонентная HAR (C1 при λ = 0, R7): f = β²·HAR-J(IMOEX) + HAR-J(u) ---
BETA_WIN, EPS_U, MIN_U_PAIRS = 252, 0.01, 252


def beta_past(d: pd.DataFrame, win: int = BETA_WIN) -> pd.Series:
    """β_t = Σ cov / Σ RV_m,sync по последним win строкам бумаги, включая t (синхронные бары с IMOEX); β >= 0 (отрицательная — 0). Нет полного окна — NaN."""
    cov, rm = (d[c].rolling(win, min_periods=win).sum() for c in ("cov", "rv_m_sync"))
    return (cov / rm.where(rm > 0)).clip(lower=0.0)


def u_daily(d: pd.DataFrame, imo: pd.DataFrame, eps: float | None = None) -> pd.DataFrame:
    """Дневной ряд идиосинкратической части u = max(RV − β²·RV_IMOEX, ε·RV), ε = EPS_U = 0,01, в формате daily (rv, rv_w, rv_m — окна уровней u в сегменте; j — скачок бумаги без изменений)."""
    eps = EPS_U if eps is None else eps
    m = beta_past(d) ** 2 * imo["rv"].reindex(d.index)
    u = np.maximum(d["rv"] - m, eps * d["rv"]).where(m.notna() & d["rv"].notna())
    seg = d["seg"].to_numpy()
    return pd.DataFrame({"rv": u, "rv_w": DC.wmean(u.to_numpy(float), seg, 5), "rv_m": DC.wmean(u.to_numpy(float), seg, 22), "j": d["j"], "seg": seg}, index=d.index)


def harj_at(p: H.Panel, i: int, min_pairs: int = MIN_U_PAIRS) -> float:
    """σ̂² HAR-J (OLS, exp(f + s²/2)) строки i по парам t+1 <= i с конечными X и y; мало пар или нефинитный X — NaN."""
    cols = list(H.COLS_HARJ)
    Xt, yt = H.pairs(p, i, cols)
    ok = np.isfinite(yt)
    if ok.sum() < min_pairs or not np.isfinite(p.X[i, cols]).all():
        return float("nan")
    b, s2, _ = H.ols(Xt[ok], yt[ok])
    return H.predict(b, p.X[i, cols], s2)[1]


def k7_series(d: pd.DataFrame, imo: pd.DataFrame, k1: pd.DataFrame, k1_imo: pd.DataFrame) -> pd.DataFrame:
    """σ̂²_K7 на датах сетки K1 бумаги: β²_d·Ĥ(IMOEX, d) + Ĥ(u, d); Ĥ(IMOEX) — K1 индекса (har_IMOEX). Нет β, пар u или прогноза индекса — NaN."""
    du = u_daily(d, imo)
    pu = H.make_panel(du)
    rows = np.flatnonzero(np.isin(d.index.to_numpy(), k1["d"].to_numpy()))
    hu = np.array([harj_at(pu, int(i)) for i in rows])
    hm = k1_imo.set_index("d")["sigma2"].reindex(d.index[rows]).to_numpy(float)
    b2 = (beta_past(d) ** 2).to_numpy(float)[rows]
    return pd.DataFrame({"d": d.index[rows], "target_date": pu.dates[rows + 1], "sigma2_k7": b2 * hm + hu, "k7_beta2": b2})


# --- сборка k4567.parquet и таблица QLIKE ---
KEY = ["ticker", "date"]
SIG = ["sigma2_k1", "sigma2_k4", "sigma2_k5", "sigma2_k6", "sigma2_k7"]


def assemble(panel: pd.DataFrame, k4: pd.DataFrame, k5: pd.DataFrame, k6: dict, k7: dict) -> pd.DataFrame:
    """Строки панели × ряды K4–K7 и выбранные HP (k4_*, k5_*). K4/K5 там, где модель не обучена (g нет), — NaN."""
    out = panel[KEY + ["t_plus", "group", "gap", "rv_tplus", "sigma2_k1"]].copy()
    for name, fc in (("k4", k4), ("k5", k5)):
        f = fc[fc["g"].notna()].rename(columns={"sigma2": "sigma2_" + name, "depth": name + "_depth", "c": name + "_c", "lam": name + "_lam", "n_train": name + "_ntrain"})
        out = out.merge(f[KEY + ["sigma2_" + name, name + "_depth", name + "_c", name + "_lam", name + "_ntrain"]], on=KEY, how="left")
    for col, ser, extra in (("sigma2_k6", k6, ["k6_fb"]), ("sigma2_k7", k7, ["k7_beta2"])):
        s = pd.concat([v.assign(ticker=t) for t, v in ser.items()]).rename(columns={"d": "date"})
        out = out.merge(s[KEY + [col] + extra], on=KEY, how="left")
    return out.sort_values(KEY).reset_index(drop=True)


def qlike_tables(k: pd.DataFrame, first_target: str = P.FIRST_TARGET, sigs=SIG) -> pd.DataFrame:
    """QLIKE рядов на ОБЩИХ датах бумаги (все ряды конечны и > 0; R10), цели с first_target. Строки: бумага и группы (среднее по бумагам) × {all, all_no_gap, год цели}."""
    k = k[(k["t_plus"] >= pd.Timestamp(first_target))].copy()
    ok = np.logical_and.reduce([np.isfinite(k[s]) & (k[s] > 0) for s in sigs])
    k = k[ok]
    for s in sigs:
        k["q_" + s[7:]] = metrics.qlike(k["rv_tplus"], k[s])
    qc = ["q_" + s[7:] for s in sigs]
    per = []
    for name, m in [("all", np.ones(len(k), bool)), ("all_no_gap", ~k["gap"].to_numpy(bool))] + [(str(y), (k["t_plus"].dt.year == y).to_numpy()) for y in sorted(k["t_plus"].dt.year.unique())]:
        g = k[m].groupby(["ticker", "group"])
        a = g[qc].mean().join(g.size().rename("n")).reset_index()
        per.append(a.assign(period=name))
    t = pd.concat(per, ignore_index=True)
    port = pd.read_csv(P.REP / "P0_PORTABILITY.csv", comment="#").set_index("ticker")["K6_portable"]
    rows = [t.assign(scope="ticker", name=t["ticker"])]
    for gname, sel in (("M16", t["group"] == "M16"), ("SBER", t["group"] == "SBER"), ("E11", t["group"] == "E11"), ("all28", t["group"].notna()), ("K6_portable23", t["ticker"].map(port).fillna(False).astype(bool))):
        s = t[sel]
        a = s.groupby("period")[qc].mean().join(s.groupby("period")["n"].sum()).reset_index()
        rows.append(a.assign(scope="group", name=gname, ticker=gname, group=gname, n_tickers=s.groupby("period")["ticker"].nunique().reindex(a["period"]).to_numpy()))
    return pd.concat(rows, ignore_index=True)[["scope", "name", "period", "n", "n_tickers"] + qc]
