"""Этап Z, помесячная переоценка (план Z раздел 4, протокол Y-A): в первый торговый день D каждого месяца с 2018-07-02 по ВСЕМ закрытым к D парам (t⁺ ≤ D, расширяющееся окно); параметры действуют на дни D … конец месяца.
До 2018-07-02 F = X1 побитно, `active` = 0; ZB, ZB0 включаются, только когда закрытых пар e_d = 1 ≥ 30 (до этого active = 0, F = X1). Пара j: (d_j, t⁺_j, X1_j, RV(t⁺_j), признаки дня d_j).
ZA: g = ω(b + a·s); ZB: e·(b + a·s); ZB0: e·b; ZC, ZCp: σ(γ0 + γᵀz̃)·(b + a·s), z̃ — стандартизация по паре пула; ZCp — z_{d−252}."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_z_feat as FE, x1_z_fit as FT
from models_v2.s16.x1_ya_walk import ACTIVATION, MIN_PAIRS, check_activation, month_starts  # noqa: F401

KINDS = ("ZA", "ZB", "ZB0", "ZC", "ZCp")
OUT_COLS = ["d", "t_plus", "rv", "X1", "F", "g", "active", "lam", "omega", "b", "a", "gam0", "gam1", "gam2", "gam3", "gam4", "gate", "n_ext"]


def closed_date(tp, d):
    """День, с которого пара доступна для обучения: t⁺ (мутация «закрыто + 1»: пара видна в день прогноза d)."""
    return tp


def closed_pool(tp, d, D) -> np.ndarray:
    """Индексы (в порядке дат) пар, закрытых к дню D: closed_date ≤ D."""
    return np.flatnonzero(np.asarray(closed_date(np.asarray(tp, "datetime64[ns]"), np.asarray(d, "datetime64[ns]"))) <= np.datetime64(D))


def refit_za(pool, D) -> dict:
    u, s = D["u"], D["s"]
    v = FT.n_valid(len(pool), FT.V_FULL)
    fit = lambda i, lam: FT.fit_lin(u[i], s[i], lam)
    lam, om = FT.select(pool[: len(pool) - v], FT.val_idx(pool, v), fit, lambda th, i, o: FT.qlike_g(u[i], o * (th[0] + th[1] * s[i])), FT.OMEGAS)
    th = fit(pool, lam)
    return dict(lam=lam, omega=om, b=th[0], a=th[1])


def refit_zb(pool, D, free_a: bool) -> dict | None:
    """ZB / ZB0: только по экстремальным парам; None — меньше 30 закрытых экстремальных пар (не включена)."""
    u, s, e = D["u"], D["s"], D["e"]
    ext = FT.ext_pool(pool, e)
    if len(ext) < FT.MIN_EXT:
        return None
    v = FT.n_valid(len(ext), FT.V_EXT)
    fit = lambda i, lam: FT.fit_lin(u[i], s[i], lam, free_a)
    lam, _ = FT.select(ext[: len(ext) - v], FT.val_block_b(pool, ext, v), fit, lambda th, i, o: FT.qlike_g(u[i], e[i] * (th[0] + th[1] * s[i])))
    th = fit(ext, lam)
    return dict(lam=lam, omega=1.0, b=th[0], a=th[1])


def refit_zc(pool, D) -> tuple:
    """-> (параметры, z̃ для всех строк); z̃ стандартизован по парам пула."""
    u, s = D["u"], D["s"]
    Zt = FE.standardize(D["Z"], pool)
    v = FT.n_valid(len(pool), FT.V_FULL)
    fit = lambda i, lam: FT.fit_gate(u[i], s[i], Zt[i], lam)
    lam, _ = FT.select(pool[: len(pool) - v], FT.val_idx(pool, v), fit, lambda th, i, o: FT.qlike_g(u[i], FT.gate_g(th, s[i], Zt[i])))
    th = fit(pool, lam)
    return dict(lam=lam, omega=1.0, b=th[5], a=th[6], gam=th[:5]), Zt


def block(kind, par, D, Zt, a, b) -> tuple:
    """-> (g, затвор) для строк a:b."""
    s, e = D["s"][a:b], D["e"][a:b]
    if kind == "ZA":
        return par["omega"] * (par["b"] + par["a"] * s), np.full(b - a, np.nan)
    if kind in ("ZB", "ZB0"):
        return e * (par["b"] + par["a"] * s), np.full(b - a, np.nan)
    th = np.concatenate([par["gam"], [par["b"], par["a"]]])
    return FT.gate_g(th, s, Zt[a:b]), FT.gate_parts(th, s, Zt[a:b])[0]


def walk(x1: pd.DataFrame, D: dict, kind: str) -> pd.DataFrame:
    """x1: d, t_plus, rv, F (= X1); D: u, s, e (n,), Z (n, 4) — сырые z (для ZCp — сдвиг внутри). -> прогнозы (OUT_COLS)."""
    d, tp, X1, n = pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"]), x1["F"].to_numpy(float), len(x1)
    check_activation(d, tp)
    D = dict(D, Z=FE.shift_z(D["Z"]) if kind == "ZCp" else D["Z"])
    g, gate, act, nx = np.zeros(n), np.full(n, np.nan), np.zeros(n, int), np.full(n, np.nan)
    par = np.full((n, 9), np.nan)                      # λ, ω, b, a, γ0…γ4
    starts = [p for p in month_starts(d) if d[p] >= ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        pool = closed_pool(tp, d, d[a])
        nx[a:b] = FT.ext_pool(pool, D["e"]).size
        Zt = None
        if kind == "ZA":
            p = refit_za(pool, D)
        elif kind in ("ZB", "ZB0"):
            p = refit_zb(pool, D, kind == "ZB")
        else:
            p, Zt = refit_zc(pool, D)
        if p is None:
            continue
        g[a:b], gate[a:b] = block(kind, p, D, Zt, a, b)
        act[a:b] = 1
        par[a:b, :4] = [p["lam"], p["omega"], p["b"], p["a"]]
        par[a:b, 4:] = p.get("gam", np.full(5, np.nan))
    cols = {c: par[:, i] for i, c in enumerate(("lam", "omega", "b", "a", "gam0", "gam1", "gam2", "gam3", "gam4"))}
    return pd.DataFrame(dict(d=d, t_plus=tp, rv=x1["rv"].to_numpy(float), X1=X1, F=X1 * np.exp(g), g=g, active=act, gate=gate, n_ext=nx, **cols))[OUT_COLS]
