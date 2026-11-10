"""Z2, H0 — только БАРЫ, без цели RV (план Z2 разделы 4, 6, L3): события по окнам, ступень 1 (помесячно с 2018-07-02 и одна на 2015–2021), LR против пуассоновской базы, таблица событий по годам и слотам,
пороги 0,98 / 0,995 (справочно), правило дубля: |corr| ξ_d (θ окна до 2018-07-02) с s_d, e_d, N_d, x_d, v_d, ρ1, ρ6, J_d/RV_d, EWMA₂₀ 1[J > 0] на днях 2015–2021. Файлов не пишет."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models_v2.s13 import v6_3_r as RH
from models_v2.s16 import x1_ya_feat as YF, x1_z2_ev as EV, x1_z2_hawkes as HK, x1_z_feat as ZF

LR_FROM, LR_TO, DUP_LIM = pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31"), 0.8
DUP_VARS = ("s_d", "e_d", "N_d", "x_d", "v_d", "rho1", "rho6", "J_d/RV_d", "EWMA20_J")


def real_ctx(permute: bool = False) -> dict:
    """Контекст склада M10 V7 (пороги и события по барам); permute — Hc (перестановка внутри дня, seed 0)."""
    days, r, qq = EV.real_thresholds()
    N, m = EV.events(r, qq[:, 1])
    if permute:
        N2, m2 = EV.permute_events(N, m, np.isfinite(r))
        EV.check_control(N, m, N2, m2, np.isfinite(r))
        N, m = N2, m2
    return dict(HK.make_ctx(days, r, qq[:, 1], N, m), r=r, q3=qq)


def fit_at(ctx: dict, pos: int) -> tuple:
    """Ступень 1 на переоценке D = склад[pos]: окно 500 сессий < D (+ 20 прогрева) -> (fit, f_D)."""
    f = HK.profile(ctx, pos)
    return HK.fit(HK.window(ctx, pos), f), f


def monthly(ctx: dict, Dpos: list) -> tuple:
    """Помесячные оценки -> (таблица LR, {pos: (fit, f)})."""
    rows, fits = [], {}
    for p in Dpos:
        fit, f = fit_at(ctx, p)
        fits[p] = (fit, f)
        rows.append(row_of(str(ctx["days"][p].date()), "monthly", fit))
    return pd.DataFrame(rows), fits


def row_of(key: str, kind: str, fit: dict) -> dict:
    lr = HK.lr_test(fit["ll"], fit["ll0"])
    return dict(window=key, kind=kind, nev=fit["nev"], nslots=fit["nslots"], mu=fit["mu"], n_hat=fit["n"], beta=fit["beta"], half_life_min=math.log(2) / fit["beta"], eta=fit["eta"], rho=fit["rho"], alpha=fit["alpha"],
                ll=fit["ll"], ll0=fit["ll0"], **lr, nit=fit["nit"])


def lr_full(ctx: dict) -> dict:
    """Одна ступень 1 на всём 2015–2021 (D для профиля — первая сессия после 2021)."""
    days = ctx["days"]
    a, b = int(days.searchsorted(LR_FROM)), int(days.searchsorted(LR_TO, side="right"))
    f = HK.profile(ctx, b)
    return HK.fit(HK.rows(ctx, a, b), f)


def events_table(ctx: dict) -> pd.DataFrame:
    """События по годам, слотам, годам × слотам; доля дней E с N_d ≥ 2; числа при Q 0,98 / 0,995 (справочно)."""
    days, pres, N, q3, r = ctx["days"], ctx["pres"], ctx["N"] > 0.5, ctx["q3"], ctx["r"]
    ok, yr = ctx["base_inlik"], days.year.to_numpy()
    ev = {q: EV.events(r, q3[:, i])[0] for i, q in enumerate(EV.QS)}
    rows = []
    for y in sorted(set(yr[ok])):
        k = ok & (yr == y)
        nd = N[k].sum(1)
        rows.append(dict(kind="year", key=str(y), n_days=int(k.sum()), n_bars=int(pres[k].sum()), n_events=int(N[k].sum()), rate=float(N[k].sum() / pres[k].sum()), n_E_days=int((nd >= 1).sum()),
                         share_E_N2=float((nd >= 2).sum() / max((nd >= 1).sum(), 1)), n_ev_q98=int(ev[0.98][k].sum()), n_ev_q995=int(ev[0.995][k].sum())))
    for s in range(EV.NS):
        rows.append(dict(kind="slot", key=str(s), n_days=int(ok.sum()), n_bars=int(pres[ok][:, s].sum()), n_events=int(N[ok][:, s].sum()), rate=float(N[ok][:, s].sum() / max(pres[ok][:, s].sum(), 1)),
                         n_ev_q98=int(ev[0.98][ok][:, s].sum()), n_ev_q995=int(ev[0.995][ok][:, s].sum())))
    for y in sorted(set(yr[ok])):
        for s in range(EV.NS):
            k = ok & (yr == y)
            rows.append(dict(kind="year_slot", key="%d|%d" % (y, s), n_bars=int(pres[k][:, s].sum()), n_events=int(N[k][:, s].sum()), rate=float(N[k][:, s].sum() / max(pres[k][:, s].sum(), 1))))
    return pd.DataFrame(rows)


def daily_vars(ctx: dict, idx: np.ndarray, xi: np.ndarray) -> pd.DataFrame:
    """Переменные дня на строках склада idx (2015–2021) для правила дубля; ξ — первой колонкой."""
    from models_v2.s11 import v4_i_comp as IC, v4_io as IO
    days, pres, r2 = ctx["days"], ctx["pres"], ctx["r2"]
    st = dict(days=days, r2=r2, present=pres)
    x, v, _ = ZF.x_v(st, days[idx])
    d = IO.load_daily("SBER")[["rv", "j", "seg"]]
    f = IC.daily_features(d).reindex(days[idx])
    rho = np.full((len(idx), 2), np.nan)
    for i, j in enumerate(idx):
        rr = ctx["r"][j][pres[j]]
        if len(rr) and (rr ** 2).sum() > 0:
            h = RH.day_rho(rr, np.ones(len(rr)))[0]
            rho[i] = (h[0], h[5])
    nd = (ctx["N"][idx] > 0.5).sum(1).astype(float)
    return pd.DataFrame(dict(xi=xi[idx], s_d=[YF.day_features(r2[j], pres[j])[0] for j in idx], e_d=(nd >= 1).astype(float), N_d=nd, x_d=x, v_d=v, rho1=rho[:, 0], rho6=rho[:, 1],
                             **{"J_d/RV_d": f["jr"].to_numpy(), "EWMA20_J": f["ewf"].to_numpy()}))


def dup_table(ctx: dict, fit: dict, f: np.ndarray) -> pd.DataFrame:
    """|corr| ξ_d с девятью переменными на днях 2015–2021; ξ — θ окна до 2018-07-02, f_D того же D."""
    days = ctx["days"]
    idx = np.arange(int(days.searchsorted(LR_FROM)), int(days.searchsorted(LR_TO, side="right")))
    xi = HK.xi_all(fit, ctx, f, ctx["dnext"])
    t = daily_vars(ctx, idx, xi)
    rows = [dict(var=v, corr=float(t["xi"].corr(t[v])), abs_corr=float(abs(t["xi"].corr(t[v]))), n=int(t[["xi", v]].dropna().shape[0]), over_limit=bool(abs(t["xi"].corr(t[v])) > DUP_LIM)) for v in DUP_VARS]
    return pd.DataFrame(rows)
