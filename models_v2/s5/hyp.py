"""Гипотезы (а)–(е) «почему бустинг проигрывает в 2017–2021»: числа и вывод по правилам GATES_A.md (заданы до счёта)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from models_v2.s5 import common

YEARS = tuple(range(2017, 2026))   # 2026 неполный — в корреляции не входит
CONF, REJ = 0.6, 0.2
BIAS_DIFF = 0.05
TOP_N, SHARE_CONF, SHARE_REJ = 20, 0.5, 0.3


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(sps.spearmanr(x, y)[0])


def verdict_rho(r: float, sign: int, rej: float = REJ) -> str:
    """sign=-1: подтверждена при ρ <= −0,6, отвергнута при ρ >= −rej; sign=+1: зеркально."""
    s = sign * r
    if not np.isfinite(s):
        return "не различима"
    return "подтверждена" if s >= CONF else ("отвергнута" if s <= rej else "не различима")


def verdict_b(p_ks: float, r: float) -> str:
    if p_ks < 0.01 and r <= -CONF:
        return "подтверждена"
    if p_ks >= 0.05 or (np.isfinite(r) and r >= -REJ):
        return "отвергнута"
    return "не различима"


def verdict_c(diff: float) -> str:
    return "подтверждена" if diff >= BIAS_DIFF else ("отвергнута" if diff <= 0 else "не различима")


def verdict_d(s_all: float, s_ex: float, share: float) -> str:
    if np.sign(s_ex) != np.sign(s_all) or share >= SHARE_CONF:
        return "подтверждена"
    return "отвергнута" if share < SHARE_REJ else "не различима"


def years_table(df, npairs, overfit, choice) -> pd.DataFrame:
    rows = []
    for Y in range(2017, 2027):
        s = df[df["target_date"].dt.year == Y]
        of = overfit[(overfit.model == "fx_b2_g132") & (overfit.year == str(Y))]
        ch = choice[(choice.Y == Y) & choice.chosen]
        ok = s["aux_imoex_rv"].notna()
        rows.append({"year": Y, "n": len(s), "delta_qlike": s["qlike_b2"].mean() - s["qlike_harj"].mean(), "n_pairs_jan1": npairs[Y],
                     "median_ln_rv": float(np.log(s["rv"]).median()), "bias_ln_b2": s["err_ln_b2"].mean(), "bias_ln_harj": s["err_ln_harj"].mean(),
                     "trees_mean": float(of["trees_mean"].iloc[0]), "depth": int(ch["depth"].iloc[0]),
                     "es_over_train": float(of["ratio_mean"].iloc[0]),
                     "corr_sber_imoex": float(np.corrcoef(np.log(s.loc[ok, "rv_d"]), np.log(s.loc[ok, "aux_imoex_rv"]))[0, 1])})
    return pd.DataFrame(rows)


def top_days(df, n: int = TOP_N, until: int = 2021) -> pd.DataFrame:
    s = df[df["target_date"].dt.year <= until]
    t = s.reindex(s["dloss"].abs().sort_values(ascending=False).index[:n])
    return t[["d", "target_date", "rv", "s2_b2", "s2_harj", "qlike_b2", "qlike_harj", "dloss", "stress_g4", "quintile_w"]].reset_index(drop=True)


def hypotheses(df, yt, top) -> pd.DataFrame:
    y = yt[yt.year.isin(YEARS)]
    d = y["delta_qlike"].to_numpy()
    early = df[df["target_date"].dt.year <= 2021]
    r_a, r_b, r_g, r_e = rho(y.n_pairs_jan1, d), rho(y.median_ln_rv, d), rho(y.trees_mean, d), rho(y.corr_sber_imoex, d)
    a, b = df[df["target_date"].dt.year <= 2021]["rv"], df[(df["target_date"].dt.year >= 2023) & (df["target_date"].dt.year <= 2025)]["rv"]
    ks = sps.ks_2samp(np.log(a), np.log(b))
    e = y[y.year <= 2021]
    bdiff = float(e.bias_ln_b2.abs().mean() - e.bias_ln_harj.abs().mean())
    s_all, s_ex = float(early["dloss"].sum()), float(early.drop(index=early.index[early["d"].isin(top["d"])])["dloss"].sum())
    share = float(top["dloss"].abs().sum() / early["dloss"].abs().sum())
    n17, n23 = int(yt.loc[yt.year == 2017, "n_pairs_jan1"].iloc[0]), int(yt.loc[yt.year == 2023, "n_pairs_jan1"].iloc[0])
    R = [("а", "ρ(число пар на 1 янв., Δ_Y); пар 2017=%d, 2023=%d" % (n17, n23), r_a, "ρ<=-0.6", "ρ>=-0.2", verdict_rho(r_a, -1)),
         ("б", "KS ln RV 2017–2021 vs 2023–2025: p=%.3g, D=%.3f; ρ(медиана ln RV, Δ_Y)" % (ks.pvalue, ks.statistic), r_b, "KS p<0.01 и ρ<=-0.6", "KS p>=0.05 или ρ>=-0.2", verdict_b(ks.pvalue, r_b)),
         ("в", "mean|bias ln| 2017–2021: Б-2 минус HAR-J(б)", bdiff, ">=0.05", "<=0", verdict_c(bdiff)),
         ("г", "ρ(деревья Б-2, Δ_Y)", r_g, "ρ>=+0.6", "ρ<=0", verdict_rho(r_g, +1, 0.0)),
         ("д", "Σ dloss 2017–2021 = %.4f; без топ-20 дней = %.4f; доля топ-20 в Σ|dloss|" % (s_all, s_ex), share, "знак S меняется или доля>=0.5", "знак тот же и доля<0.3", verdict_d(s_all, s_ex, share)),
         ("е", "ρ(corr ln RV SBER–IMOEX по году, Δ_Y)", r_e, "ρ<=-0.6", "ρ>=-0.2", verdict_rho(r_e, -1))]
    return pd.DataFrame(R, columns=["id", "diagnostic", "number", "confirm_rule", "reject_rule", "verdict"])


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Чувствительность к выборке (диагностика §24.9 блока, не результат): без 2022, без топ-20 дней по |dloss|, без двух дней апреля 2018 (пост-фактум)."""
    y = df["target_date"].dt.year
    top_all = df["dloss"].abs().nlargest(TOP_N).index
    top_early = df[y <= 2021]["dloss"].abs().nlargest(TOP_N).index
    apr = df["target_date"].isin(pd.to_datetime(["2018-04-09", "2018-04-10"]))   # прогнозы с d = 06.04 и 09.04
    cases = [("all", np.ones(len(df), bool)), ("without_2022", (y != 2022).to_numpy()), ("without_top20_abs_dloss_all", ~df.index.isin(top_all)),
             ("2017_2021", (y <= 2021).to_numpy()), ("2017_2021_without_top20_of_2017_2021", ((y <= 2021) & ~df.index.isin(top_early)).to_numpy()),
             ("2017_2021_without_2018_04_06_and_09 (post hoc)", ((y <= 2021) & ~apr).to_numpy()), ("2023_2025", ((y >= 2023) & (y <= 2025)).to_numpy())]
    rows = []
    for lab, m in cases:
        s = df[m]
        rows.append({"sample": lab, "n": len(s), "qlike_b2": s["qlike_b2"].mean(), "qlike_harj": s["qlike_harj"].mean(),
                     "delta": s["qlike_b2"].mean() - s["qlike_harj"].mean(), "win_share_b2": float(((s["dloss"] < 0).sum() + 0.5 * (s["dloss"] == 0).sum()) / len(s))})
    return pd.DataFrame(rows)


def load_inputs() -> tuple:
    ov = common.read_comment_csv(common.TAB / "fix_overfit_c.csv")
    ov["year"] = ov["year"].astype(str)
    return ov, common.read_comment_csv(common.TAB / "fix_i2_choice_g132.csv")
