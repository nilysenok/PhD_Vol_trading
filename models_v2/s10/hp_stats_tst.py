"""P3, G11: `TESTS.csv` и `MCS.csv` (R13, У-14). Единица — ДАТА: Δ_d = среднее по бумагам дня d разности QLIKE (гибрид − база); DM — Ньюи–Уэст лаг 5, двусторонний p (`models_v2/s3/dm`);
блочный бутстрап ДАТ (блок 20, 5000, seed 42; `stats_d.block_idx`) с сохранением всех бумаг дня; MCS 90 % (`stats_d.mcs`) на среднем по панели. Холм — `stats_d.holm_n(p, N)`, N объявлено ДО p (CHECKPOINT).
Основное семейство Холма: 20 гибридов × {K1, K4, K5} на M16 × {2022–02.2026, вся история}; 20 × {K2, K3} на M16 в тех же двух окнах на общих датах с K2, K3 (G15); LOTO к K1 (M16, оба окна). Остальное — описательно (аux). Δ < 0 — гибрид лучше."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_paths as P, hp_stats_cmp as CM, hp_stats_sel as S

SCOPES = ("M16", "SBER", "E11", "all28")
WINS = ("all", "p2017_2021", "y2022", "p2023_2025", "y2026", "confirm_2022_2026", "no_240222", "no_top5_adv")
HYB = [c for c in S.model_cols() if "_" in c]
MAIN = ("M16", ("confirm_2022_2026", "all"))       # основное семейство: M16, подтверждение 2022–02.2026 и вся история (два окна — две гипотезы)


@lru_cache(maxsize=6)
def block_idx(n: int) -> np.ndarray:
    return SD.block_idx(n)


def resample(dl: np.ndarray) -> np.ndarray:
    """(B,) бутстрап-средние Δ по датам: блоки по 20 смежных ДАТ (все бумаги даты вместе — Δ_d уже панельное). Подменяется мутацией «ячейки независимо»."""
    return SD.boot_means(np.asarray(dl, float), block_idx(len(dl)))


def dm_panel(dl: np.ndarray) -> tuple:
    """DM на d = −Δ (плюс = модель лучше), как `dm.dm`: (stat, p двусторонний)."""
    d = -np.asarray(dl, float)
    v = dmod.nw_variance(d, 5)
    if len(d) < 3 or v <= 0:
        return float("nan"), float("nan")
    z = float(d.mean()) / math.sqrt(v)
    return z, math.erfc(abs(z) / math.sqrt(2))


def pair_stats(dl: np.ndarray, drop_top5: bool = False) -> dict:
    """Δ среднее, DM (НУ 5), бутстрап дат: ДИ 95 %, односторонний p. drop_top5 — без 5 дат наибольшего преимущества (наиболее отрицательных Δ_d)."""
    dl = np.asarray(dl, float)
    if drop_top5 and len(dl) > 5:
        dl = np.delete(dl, np.argsort(dl)[:5])
    if len(dl) < 30:
        return dict(n_dates=len(dl), delta_mean=float("nan"), dm_stat=np.nan, dm_p=np.nan, boot_lo=np.nan, boot_hi=np.nan, boot_p=np.nan)
    z, p = dm_panel(dl)
    bm = resample(dl)
    return dict(n_dates=len(dl), delta_mean=float(dl.mean()), dm_stat=z, dm_p=p, boot_lo=float(np.percentile(bm, 2.5)), boot_hi=float(np.percentile(bm, 97.5)), boot_p=float((np.sum(bm >= 0) + 1) / (len(bm) + 1)))


def date_tab(c: pd.DataFrame, Q: pd.DataFrame, scope_mask, win_mask) -> pd.DataFrame:
    """Панельные QLIKE по датам (среднее по бумагам дня): индекс — дата, колонки — ряды."""
    m = np.asarray(scope_mask) & np.asarray(win_mask)
    return Q[m].groupby(c["date"].to_numpy()[m]).mean().sort_index()


def qmatrix(c: pd.DataFrame, sigs: list) -> pd.DataFrame:
    return pd.DataFrame({s: CM.qrow(c, s) for s in sigs})


def scope_masks(c: pd.DataFrame) -> dict:
    g = c["group"].to_numpy()
    return {"M16": g == "M16", "SBER": g == "SBER", "E11": g == "E11", "all28": np.ones(len(c), bool)}


def rows_for(c: pd.DataFrame, sigs: list, pairs: list, scopes: dict, wins: dict, kind: str) -> list:
    """pairs — [(модель, база)]; на каждый (scope, window) — строка TESTS."""
    Q, out = qmatrix(c, sigs), []
    for sn, sm in scopes.items():
        for wn, wm in wins.items():
            tab = date_tab(c, Q, sm, wm if wn != "no_top5_adv" else wins["all"])
            for a, b in pairs:
                if a in tab and b in tab and len(tab):
                    out.append(dict(config=a, comparator=b, scope=sn, window=wn, kind=kind, **pair_stats(tab[a].to_numpy() - tab[b].to_numpy(), wn == "no_top5_adv")))
    return out


def holm_main(df: pd.DataFrame, n_declared: int) -> pd.DataFrame:
    """p Холма (`holm_n`, N объявлено до p) для основного семейства: kind == 'main'; N < числа p — ошибка."""
    sel = (df["kind"] == "main") & df["dm_p"].notna()
    df["p_holm"] = np.nan
    df.loc[sel, "p_holm"] = SD.holm_n(df.loc[sel, "dm_p"].to_numpy(), n_declared)
    return df


def run_tests(k: pd.DataFrame, loto: pd.DataFrame, n_declared: int, log=print) -> pd.DataFrame:
    """TESTS.csv: гибриды × {K1, K4, K5} (все срезы), K2/K3 (28 бумаг, `k23_*`), контроли K4–K7 к K1, LOTO к K1 и к пулу."""
    c = S.common(k, CM.SIGS)
    wm = CM.slice_masks(c)
    rows = []
    pairs = [(h, b) for h in HYB for b in ("k1", "k4", "k5")]
    rows += rows_for(c, CM.SIGS, pairs, scope_masks(c), {w: wm.get(w, wm["all"]) for w in WINS}, "aux")
    c2 = S.common(CM.with_k23(k)[k["ticker"].isin(CM.K23).to_numpy()], CM.SIGS + ["k2", "k3"])
    rows += k23_rows(c2)
    rows += rows_for(c, CM.SIGS, [(s, "k1") for s in ("k4", "k5", "k6", "k7")], scope_masks(c), {w: wm.get(w, wm["all"]) for w in WINS}, "control")
    df = pd.DataFrame(rows)
    sel = (df["kind"] == "aux") & (df["scope"] == MAIN[0]) & df["window"].isin(MAIN[1])
    df.loc[sel, "kind"] = "main"
    lt = loto_rows(c, loto)
    df = holm_main(pd.concat([df, pd.DataFrame(lt)], ignore_index=True), n_declared)
    log("TESTS: %d строк, основное семейство %d p, N=%d" % (len(df), int((df["kind"] == "main").sum()), n_declared))
    return df


def k23_rows(c2: pd.DataFrame) -> list:
    """G15: гибриды × {K2, K3} на ОБЩИХ датах бумаги со всеми рядами и K2, K3 (те же области и окна, что к K1/K4/K5); окна с префиксом `k23_`.
    Основное семейство — M16 × (подтверждение, вся история), как у K1/K4/K5; остальное — описательно. N Холма не меняется (K2/K3 — контроли)."""
    wm = CM.slice_masks(c2)
    rows = rows_for(c2, CM.SIGS + ["k2", "k3"], [(h, b) for h in HYB for b in ("k2", "k3")], scope_masks(c2), {w: wm.get(w, wm["all"]) for w in WINS}, "aux")
    for r in rows:
        r["kind"] = "main" if (r["scope"] == MAIN[0] and r["window"] in MAIN[1]) else "aux"
        r["window"] = CM.K23_PREF + r["window"]
    return rows


def loto_rows(c: pd.DataFrame, loto: pd.DataFrame) -> list:
    """LOTO к K1 (M16 × 2022–02.2026, семейство main) и к пулу (aux) для каждой конфигурации LOTO; общие даты LOTO-ряда и 25 рядов."""
    out, wm = [], CM.slice_masks(c)
    for cfg, g in loto.groupby("config"):
        x = c.merge(g[S.KEY + ["sigma2"]].rename(columns={"sigma2": "loto"}), on=S.KEY, how="inner")
        x = x[np.isfinite(x["loto"]) & (x["loto"] > 0)].reset_index(drop=True)
        xm = CM.slice_masks(x)
        for b, kind in (("k1", "main"), (cfg, "aux")):
            sc = scope_masks(x)
            for sn in SCOPES:
                for wn in ("all", "confirm_2022_2026", "p2017_2021"):
                    tab = date_tab(x, qmatrix(x, ["loto", b]), sc[sn], xm[wn])
                    if len(tab):
                        out.append(dict(config=cfg + "__LOTO", comparator=b if b == "k1" else cfg + "(пул)", scope=sn, window=wn, kind=kind if (sn == MAIN[0] and wn in MAIN[1]) else "aux",
                                        **pair_stats(tab["loto"].to_numpy() - tab[b].to_numpy())))
    return out


MCS_SETS = {"xgb": ["k1", "k4", "k6", "k7"], "lgbm": ["k1", "k5", "k6", "k7"]}


def run_mcs(k: pd.DataFrame, log=print) -> pd.DataFrame:
    """MCS 90 % (T_max, блочный бутстрап дат) на панельном QLIKE по датам: наборы XGB и LGBM (контроли + 10 гибридов), области M16 / все 28, окна выбора / подтверждения / все."""
    c = S.common(k, CM.SIGS)
    wm, sm, rows = CM.slice_masks(c), scope_masks(c), []
    Q = qmatrix(c, CM.SIGS)
    for mod, ctl in MCS_SETS.items():
        ms = ctl + [h for h in HYB if h.startswith(mod + "_")]
        for sn in ("M16", "all28"):
            for wn in ("p2017_2021", "confirm_2022_2026", "all"):
                tab = date_tab(c, Q, sm[sn], wm[wn])[ms]
                r = SD.mcs(tab.to_numpy(), block_idx(len(tab)))
                rows += [dict(set=mod, scope=sn, window=wn, model=m, mean_qlike=float(tab[m].mean()), mcs_p=float(r["p"][i]), in_mcs90=bool(r["in"][i]), n_dates=len(tab)) for i, m in enumerate(ms)]
    log("MCS: %d строк" % len(rows))
    return pd.DataFrame(rows)


def write(tests: pd.DataFrame, mcs: pd.DataFrame, n_declared: int) -> None:
    note = "%s; Δ = QLIKE(гибрид) − QLIKE(база) по датам, среднее по бумагам дня; N Холма = %d (объявлено до p, CHECKPOINT); два статуса: «снижение QLIKE (описательно)» — delta_mean < 0, «подтверждённое преимущество» — LEVELS.csv" % (S.STAT, n_declared)
    P.write_csv(tests, "statistics/TESTS.csv", note)
    P.write_csv(mcs, "statistics/MCS.csv", "MCS 90 %%, T_max, бутстрап дат (блок 20, 5000, seed 42) на панельном QLIKE по датам; %s" % S.STAT)


def declared_n() -> int:
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    if "N_declared_before_stats" not in cp:
        raise SystemExit("N не объявлено в CHECKPOINT.json: сначала `stats --part sel`")
    return int(cp["N_declared_before_stats"])
