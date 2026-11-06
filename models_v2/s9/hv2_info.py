"""Ж-214, лист 1: ОПИСАТЕЛЬНАЯ диагностика — связаны ли M10-величины дня d с ln-ошибкой A0 e = ln RV_{d+1} − ln σ̂²_HAR(d).
Выполняется ПОСЛЕ фиксации правил GATES_V2 (V1–V17), на сетки и выбор не влияет. Признаки стандартизованы по всей выборке (диагностика, не прогноз).
Не trials. p: HAC (Ньюи–Уэст, лаг 5) для МНК e ~ z; Холм — на число проверок (17) и консервативно с N = 541."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from models_v2.s5.d import stats_d

PERIODS = {"2017-21": ("2017-01-01", "2021-12-31"), "2022": ("2022-01-01", "2022-12-31"), "2023-26": ("2023-01-01", "2026-12-31")}


def candidates(t: pd.DataFrame) -> pd.DataFrame:
    rv = t["rv"]
    f = pd.DataFrame(index=t.index)
    f["ln_rv_rel (ошибка измерения)"] = np.log(t["rv_rel"])
    f["ln_sqrtRQ_minus_lnRV"] = 0.5 * np.log(t["rq"]) - np.log(rv)
    f["ln_RSneg_over_RSpos"] = np.log(t["rs_neg"] / t["rs_pos"])
    f["jump_share J/RV"] = t["j"] / rv
    f["bv_over_rv"] = t["bv"] / rv
    f["ln_open_over_mid"] = np.log(t["rv_open"].where(t["rv_open"] > 0) / t["rv_mid"])
    f["ln_close_over_mid"] = np.log(t["rv_close"] / t["rv_mid"])
    f["open_share"] = t["rv_open"] / rv
    f["close_share"] = t["rv_close"] / rv
    f["corr_sber_imoex"] = t["corr"]
    f["beta_sber_imoex"] = t["beta"]
    f["semicov_asym (pp-nn)/(|.|)"] = (t["cov_pp"] - t["cov_nn"]) / (t["cov_pp"] + t["cov_nn"] + t["cov_mix"].abs())
    f["mix_share |mix|/(|.|)"] = t["cov_mix"].abs() / (t["cov_pp"] + t["cov_nn"] + t["cov_mix"].abs())
    f["ln_RSMneg_over_RSMpos (рынок)"] = np.log(t["rsm_neg"] / t["rsm_pos"])
    f["ln_rv_over_rv_mkt"] = np.log(t["rv_s_sync"] / t["rv_m_sync"])
    f["idio_share 1-corr2"] = 1 - t["corr"] ** 2
    f["ln_rv_sa_over_rv (дрейф профиля)"] = np.log(t["rv_sa"] / rv)
    return f


def nw_t(z: np.ndarray, e: np.ndarray, lag: int = 5) -> tuple:
    """МНК e = a + b z с HAC: (b, t, p). z, e без NaN."""
    z = (z - z.mean()) / z.std()
    X = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(X, e, rcond=None)[0]
    u = e - (X * beta).sum(axis=1)
    xu = X * u[:, None]
    S = xu.T @ xu / len(z)
    for k in range(1, lag + 1):
        G = xu[k:].T @ xu[:-k] / len(z)
        S += (1 - k / (lag + 1)) * (G + G.T)
    A = np.linalg.inv(X.T @ X / len(z))
    se = np.sqrt((A @ S @ A)[1, 1] / len(z))
    tt = beta[1] / se
    return float(beta[1]), float(tt), float(2 * (1 - stats.norm.cdf(abs(tt))))


def table(t: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """t — m10_daily; comp — components (d, rv, HAR). Строки — кандидаты."""
    f = candidates(t).reindex(pd.DatetimeIndex(comp["d"]))
    e = pd.Series(np.log(comp["rv"].to_numpy()) - np.log(comp["HAR"].to_numpy()), index=f.index)
    rows = []
    for name in f.columns:
        ok = f[name].notna() & np.isfinite(f[name])
        z, ee = f.loc[ok, name].to_numpy(float), e[ok].to_numpy(float)
        b, tt, p = nw_t(z, ee)
        row = {"feature": name, "n": int(ok.sum()), "spearman_all": float(stats.spearmanr(z, ee)[0]), "b_per_sd": b, "t_hac": tt, "p_raw": p}
        for lab, (a, c) in PERIODS.items():
            m = (f.index[ok] >= a) & (f.index[ok] <= c)
            row["rho_" + lab] = float(stats.spearmanr(z[m], ee[m])[0])
        yrs = pd.Series(ee).groupby(f.index[ok].year).apply(lambda s: np.sign(stats.spearmanr(z[(f.index[ok].year == s.name)], s.to_numpy())[0]))
        row["years_same_sign_as_all"] = int((yrs == np.sign(row["spearman_all"])).sum())
        row["years"] = int(len(yrs))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm_tests"] = stats_d.holm_n(out["p_raw"].to_numpy(), len(out))
    out["p_holm_N541"] = stats_d.holm_n(out["p_raw"].to_numpy(), 541)
    return out.sort_values("p_raw").reset_index(drop=True)
