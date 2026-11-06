"""Ж-214, лист 4 (V10, ступень Л8): рыночная общность SBER–IMOEX. Log-OLS на A0-спецификации (ln RV_{t+1} на [ln RV_d, ln RV_w, ln RV_m, ln(1+J)] + столбцы), σ̂² = exp(f + s²/2), расширяющееся окно,
пары t+1 <= d. Mb (контроль рынка) = A0 + [ln RV_IMOEX d/w/m, corr_win, beta_win], win = 22 (E9 — 63); corr_win = Σcov/√(ΣRV_s·ΣRV_m) по win строкам сегмента, включая t (синхронные бары).
Блоки (У29): semicov, idio, relrv, dmkt, mktsemi. День с нефинитным признаком → прогноз A0 (`n_fallback`)."""
from __future__ import annotations

import numpy as np

from models_v2.s9 import hv2_dec as DC, hv2_harq as HQ, hv2_msfeat as MS

WIN = 22
BLOCKS = ("semicov", "idio", "relrv", "dmkt", "mktsemi")


def mb_cols(dw, win: int = WIN) -> list:
    c = dw.comp
    with np.errstate(divide="ignore", invalid="ignore"):
        cov, rs, rm = (MS.wm(dw, c[k], win) for k in ("cov", "rv_s_sync", "rv_m_sync"))
        return [("imo_d", MS.ln(c["imo_rv"])), ("imo_w", MS.ln(c["imo_w"])), ("imo_m", MS.ln(c["imo_m"])),
                ("corr%d" % win, cov / np.sqrt(rs * rm)), ("beta%d" % win, cov / rm)]


def block_cols(dw, b: str) -> list:
    """Новые блоки V10. semicov — асимметрия (cov_pp − cov_nn)/(cov_pp + cov_nn + |cov_mix|) и доля |cov_mix| (d и окно 22); idio — ln идиосинкратической RV_s − β²RV_m и рыночной части β²RV_m
    (β = cov/RV_m дня; d и среднее 5 дней уровней); relrv — ln RV_s/RV_m (d, w); dmkt — ln RV_m,d/RV_m,w IMOEX; mktsemi — ln RSM⁻/RSM⁺."""
    c = dw.comp
    with np.errstate(divide="ignore", invalid="ignore"):
        if b == "semicov":
            out = []
            for tag, f in (("d", lambda k: np.asarray(c[k], float)), ("22", lambda k: MS.wm(dw, c[k], WIN))):
                pp, nn, mx = f("cov_pp"), f("cov_nn"), np.abs(f("cov_mix"))
                out += [("semi_asym_" + tag, (pp - nn) / (pp + nn + mx)), ("semi_mix_" + tag, mx / (pp + nn + mx))]
            return out
        if b == "idio":
            sys_ = c["cov"] ** 2 / c["rv_m_sync"]
            idio = c["rv_s_sync"] - sys_
            return [("ln_idio_d", MS.ln(idio)), ("ln_idio_w", MS.ln(MS.wm(dw, idio, 5))), ("ln_sys_d", MS.ln(sys_)), ("ln_sys_w", MS.ln(MS.wm(dw, sys_, 5)))]
        if b == "relrv":
            return [("ln_rel_d", MS.lnr(c["rv_s_sync"], c["rv_m_sync"])), ("ln_rel_w", MS.lnr(MS.wm(dw, c["rv_s_sync"], 5), MS.wm(dw, c["rv_m_sync"], 5)))]
        if b == "dmkt":
            return [("ln_dmkt", MS.lnr(c["imo_rv"], c["imo_w"]))]
        if b == "mktsemi":
            return [("ln_rsm_negpos", MS.lnr(c["rsm_neg"], c["rsm_pos"]))]
    raise ValueError("блок %r" % b)


def design(dw, use_mb: bool, blocks: tuple = (), win: int = WIN) -> tuple:
    cols = ([] if not use_mb else mb_cols(dw, win)) + [x for b in blocks for x in block_cols(dw, b)]
    names = ["ln_rv_d", "ln_rv_w", "ln_rv_m", "ln1p_J"] + [n for n, _ in cols]
    return np.column_stack([dw.p.X] + [MS.fin(v) for _, v in cols]), names


def run(dw, use_mb: bool, blocks: tuple = (), win: int = WIN, lead: int = 0) -> tuple:
    """-> (σ̂² сетки с подменой на A0 в днях с нефинитным признаком, коэффициенты, имена, маска fallback). `lead` > 0 — МУТАНТ «признаки дня d+lead» (должен ловиться обрезкой)."""
    Xd, names = design(dw, use_mb, blocks, win)
    if lead:
        Xd = np.vstack([Xd[lead:], np.full((lead, Xd.shape[1]), np.nan)])
        Xd[:, :4] = dw.p.X
    s2, B, _ = HQ.fit_design(dw, Xd, False)
    a0 = DC.a0_forecast(dw)
    fb = ~np.isfinite(s2) | (s2 <= 0)
    return np.where(fb, a0, s2), B, ["b0"] + names, fb


def informative(r: dict, crit: float = 0.0005, min_years: int = 6) -> bool:
    """Блок информативен сверх Mb: Δcrit_wf <= −0,0005, ΔQLIKE < 0 на всей истории и без 24.02.2022, лет лучше >= 6 из 10 (V10)."""
    return bool(r["d_crit_Mb"] <= -crit and r["d_qlike_Mb"] < 0 and r["d_qlike_no_240222_Mb"] < 0 and r["years_better_Mb"] >= min_years)
