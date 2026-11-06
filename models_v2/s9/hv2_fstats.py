"""Ж-214, лист 5 (V14, V15, У30–У32): статистика финалистов — тесты A (к A0), B (к XJ), C (к LF), Холм с N = 541 + trials Ж-214 (объявлено в CHECKPOINT.json ДО расчёта p),
бутстрап 20/5000/seed 42, DM Ньюи–Уэст лаг 5, MCS 90 %, уровни 1–3 (уровень 4 — «не проверялся»), влияние 5 крупнейших дней. Новых trials нет; опора — Ж-212 `hs_gstage`/`hs_levels`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5.d import stats_d
from models_v2.s8 import hs_gstage as G, hs_levels as LV, hs_store
from models_v2.s9 import hv2_ledger as L, hv2_paths as P

CEIL = "потолок: финалист выбран по crit_wf на всей истории (2017–02.2026 просмотрена); HISTORICAL WALK-FORWARD, не OOS; уровень 4 (другие акции) не проверялся"
REFS = ("HAR-J", "XGBoost", "LightGBM")


def declare_n(path=L.CHECK) -> int:
    """N для Холма = 541 + число trials Ж-214; пишется в CHECKPOINT.json до любого p. Не число тестов."""
    n = L.n_final()
    L.write_check(L.load(), path=path, N_declared_before_final=n, N_for_corrections=n)
    return n


def build_s(df: pd.DataFrame, fin: pd.DataFrame, series: dict) -> dict:
    s = {"HAR-J": df["HAR"].to_numpy(float), "XGBoost": df["XJ"].to_numpy(float), "LightGBM": df["LF"].to_numpy(float)}
    for _, r in fin.iterrows():
        s["%s:%s" % (r["step"], r["model"])] = np.asarray(series[r["model"]], float)
    for k, v in s.items():
        if len(v) != len(df) or not np.isfinite(v).all() or (v <= 0).any():
            raise SystemExit("стоп: ряд %s не конечен/не положителен" % k)
    return s


def stats_all(df: pd.DataFrame, s: dict, n: int) -> tuple:
    """-> (STATISTICAL_TESTS с p_holm, LEVELS). Холм — семейство (финалист × тест A/B), период all, N по реестру."""
    t, lv = G.tests_and_levels(df, s, n)
    lv["ceiling_note"] = CEIL
    t["family_tests"] = int(t["p_holm"].notna().sum())
    t["N_holm"] = n
    return t, lv


def top5_table(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Δ к A0 и к XJ на всей истории и без 5 дней с наибольшим вкладом в преимущество; даты этих дней."""
    q = {k: np.asarray(metrics.qlike(df["rv"].to_numpy(float), v), float) for k, v in s.items()}
    rows = []
    for k in s:
        for ref, rk in (("A0", "HAR-J"), ("XJ", "XGBoost")):
            if k in REFS:
                continue
            dl = q[k] - q[rk]
            ix = np.argsort(dl)[:5]
            rows.append({"model": k, "ref": ref, "delta_all": float(dl.mean()), "delta_wo_top5": LV.top5_wo(dl), "top5_dates": ";".join(str(df["target_date"].iloc[i].date()) for i in ix),
                         "top5_share_of_gain": float(dl[ix].sum() / dl.sum()) if dl.sum() < 0 else np.nan})
    return pd.DataFrame(rows)


def mcs_table(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    rv = df["rv"].to_numpy(float)
    lm = np.column_stack([np.asarray(metrics.qlike(rv, v), float) for v in s.values()])
    a = stats_d.mcs(lm, stats_d.block_idx(len(lm)))
    keep = G.V.mask_of(df, "no_240222")
    b = stats_d.mcs(lm[keep], stats_d.block_idx(int(keep.sum())))
    return pd.DataFrame({"model": list(s), "p_mcs": a["p"], "in_mcs_90": a["in"], "p_mcs_no_240222": b["p"], "in_mcs_90_no_240222": b["in"]})


def write(df: pd.DataFrame, name: str, note: str) -> None:
    P.write_csv(df, name, note, "statistical_validation")
