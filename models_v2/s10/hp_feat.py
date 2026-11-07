"""P0, G5 (R5): признаки X (индивидуальные), M (рыночные), C (межакционные и структурные) бумаги на дату t; формулы — P0_FEATURES.md.
Все величины строки t известны к 18:50 t. Нормировка z — ТОЛЬКО по прошлому самой бумаги: расширяющиеся медиана и (q75 − q25)/1.349 по строкам s < t (сдвиг на 1 строку), >= 252 наблюдений."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_dec as DC

MIN_HIST = 252
EPS_IDIO = 0.01     # ε доли RV: нижняя граница идиосинкратической части (R7, C1)
X_COLS = ["x_d_m", "x_w_m", "x_j_m", "x_hdev", "x_err1", "x_err5", "x_err22", "x_jshare", "x_open", "x_mid", "x_close",
          "x_open_dev", "x_mid_dev", "x_close_dev", "x_ln_close_mid", "x_rs_neg_pos", "x_ln_rel"]
M_COLS = ["m_imo_d_m", "m_imo_w_m", "m_med", "m_disp", "m_share_up"]
C_COLS = ["c_corr22", "c_sys_share", "c_ln_idio", "c_semi_asym", "c_rank", "c_dev_med"]


def ln(x) -> pd.Series:
    x = pd.Series(x)
    return np.log(x.where(x > 0))


def past_stats(x: pd.Series, min_hist: int = MIN_HIST) -> tuple:
    """(медиана, шкала) по строкам s < t (не включая t); шкала = (q75 − q25)/1.349. МУТАНТ «нормировка с днём t»/«с будущим» подменяет эту функцию в тесте."""
    e = x.expanding(min_periods=min_hist)
    return e.median().shift(1), ((e.quantile(0.75) - e.quantile(0.25)) / 1.349).shift(1)


def zscore(x: pd.Series, min_hist: int = MIN_HIST) -> pd.Series:
    med, sc = past_stats(x, min_hist)
    return ((x - med) / sc.where(sc > 0)).astype(float)


def _wm(d: pd.DataFrame, col: str, k: int) -> pd.Series:
    return pd.Series(DC.wmean(d[col].to_numpy(float), d["seg"].to_numpy(), k), index=d.index)


def individual(d: pd.DataFrame, k1: pd.DataFrame) -> pd.DataFrame:
    """X: d — дневная таблица бумаги (hp_daily.ticker_daily), k1 — ряд K1 (hp_har.series). Ошибка HAR известна в дату цели t⁺ (закрыта к t)."""
    lm = ln(d["rv_m"])
    k = k1.set_index("d")
    err = pd.Series(np.log(k["rv"].to_numpy()) - np.log(k["sigma2"].to_numpy()), index=pd.DatetimeIndex(k["target_date"]))
    out = pd.DataFrame(index=d.index)
    out["x_d_m"], out["x_w_m"] = ln(d["rv"]) - lm, ln(d["rv_w"]) - lm
    out["x_j_m"] = np.log1p(d["j"] / d["rv_m"])
    out["x_hdev"] = ln(k["sigma2"].reindex(d.index)) - lm
    out["x_err1"] = err.reindex(d.index)
    out["x_err5"] = err.rolling(5).mean().reindex(d.index)
    out["x_err22"] = err.rolling(22).mean().reindex(d.index)
    out["x_jshare"] = d["j"] / d["rv"]
    for nm, c in (("open", "rv_open"), ("mid", "rv_mid"), ("close", "rv_close")):
        sh = d[c] / d["rv"]
        out["x_" + nm] = sh
        out["x_%s_dev" % nm] = sh - sh.shift(1).rolling(22).mean()
    out["x_ln_close_mid"] = ln(d["rv_close"]) - ln(d["rv_mid"])
    out["x_rs_neg_pos"] = ln(d["rs_neg"]) - ln(d["rs_pos"])
    out["x_ln_rel"] = ln(d["rv_rel"])
    return out[X_COLS]


def market_own(d: pd.DataFrame, imo: pd.DataFrame) -> pd.DataFrame:
    """M (часть IMOEX): ln RV_d/RV_m и ln RV_w/RV_m индекса на собственных датах бумаги (нет дня индекса — NaN)."""
    out = pd.DataFrame(index=d.index)
    out["m_imo_d_m"] = (ln(imo["rv"]) - ln(imo["rv_m"])).reindex(d.index)
    out["m_imo_w_m"] = (ln(imo["rv_w"]) - ln(imo["rv_m"])).reindex(d.index)
    return out


def structural(d: pd.DataFrame) -> pd.DataFrame:
    """C (часть бумаги): корреляция с IMOEX (22 дня), доля системной части, ln идиосинкратической части, асимметрия семиковариаций."""
    cov, rs, rm = (_wm(d, c, 22) for c in ("cov", "rv_s_sync", "rv_m_sync"))
    beta = cov / rm
    sys_ = beta ** 2 * d["rv_m_sync"]
    idio = np.maximum(d["rv_s_sync"] - sys_, EPS_IDIO * d["rv_s_sync"])
    out = pd.DataFrame(index=d.index)
    out["c_corr22"] = cov / np.sqrt(rs * rm)
    out["c_sys_share"] = sys_ / d["rv_s_sync"]
    out["c_ln_idio"] = ln(idio) - ln(d["rv_m"])
    out["c_semi_asym"] = (d["cov_pp"] - d["cov_nn"]) / (d["cov_pp"] + d["cov_nn"] + d["cov_mix"].abs())
    return out


def add_z(f: pd.DataFrame) -> pd.DataFrame:
    z = f.apply(zscore)
    z.columns = ["z_" + c for c in f.columns]
    return f.join(z)
