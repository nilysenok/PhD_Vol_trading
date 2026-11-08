"""V4-C, лист C-b: модели гейта для месячного/годового цикла `v4_cb_run`. fit(kind, tr, prev) -> (state, prev_next); predict(kind, s, pr) -> (F, w, extra). kind: lin (воспроизведение C1 — контроль тождества),
gam (C5), mono (C6), boost (C7), regime (C8), three (C9), eta (C10).
tr — обучение: rv, lH, lK2, lK3, ell, X (конечные признаки), tid (0..n−1), n, present (номера бумаг), names, nT. pr — прогноз: lH, lK2, lK3, ell, X, tix."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from models_v2.s11 import v4_cb_boost as BO, v4_cb_fit as FB, v4_c_fit as CF

MONO_COL = 2            # C6: столбец γ (EWMA прошлого преимущества M по QLIKE) в наборе [ell, abs_ell, ewqd]; γ ≥ 0
LINEAR = ("lin", "gam", "mono", "eta")


def _expand(v, present, nT):
    out = np.zeros((nT,) + v.shape[1:])
    out[present] = v
    return out


def _design(kind, d, st):
    return FB.spline_basis(d["ell"], st["knots"]) if kind == "gam" else d["X"]


def _fit_linear(kind, tr, prev):
    st = {"knots": FB.spline_knots(tr["ell"])} if kind == "gam" else {}
    D = _design(kind, tr, st)
    mu, sd = CF.standardize(D)
    x0 = None if prev is None else np.concatenate([[prev["alpha"]], [prev["delta"].get(t, 0.0) for t in tr["names"]], prev["beta"]])
    r = FB.fit_lin(tr["rv"], tr["lH"], tr["ell"], (D - mu) / sd, tr["tid"], tr["n"], x0, nonneg=(MONO_COL,) if kind == "mono" else (), eta=kind == "eta")
    s = {"alpha": r["alpha"], "delta": _expand(r["delta"], tr["present"], tr["nT"]), "beta": r["beta"], "mu": mu, "sd": sd, "st": st, "ok": r["ok"], "grad": r["grad"], "nit": r["nit"],
         "eta": None if r["eta"] is None else _expand(r["eta"], tr["present"], tr["nT"])}
    return s, {"alpha": r["alpha"], "beta": r["beta"], "delta": dict(zip(tr["names"], r["delta"]))}


def _pred_linear(kind, s, pr):
    D = _design(kind, pr, s["st"])
    z = np.where(np.isfinite(D), (D - s["mu"]) / s["sd"], 0.0)
    eta = s["alpha"] + s["delta"][pr["tix"]] + (CF.mv(z, s["beta"]) if z.shape[1] else 0.0)
    if s["eta"] is not None:
        eta = eta + (z * s["eta"][pr["tix"]]).sum(1)
    F, w = CF.forward(pr["lH"], pr["ell"], eta)
    return F, w, {}


def _fit_boost(kind, tr, prev):
    m = BO.fit_boost(tr["X"], tr["rv"], tr["lH"], tr["ell"])
    return {"model": m, "ok": True, "grad": float("nan"), "nit": BO.N_TREES, "n_splits": BO.n_splits(m)}, prev


def _pred_boost(kind, s, pr):
    F, w = CF.forward(pr["lH"], pr["ell"], BO.predict_eta(s["model"], pr["X"]))
    return F, w, {}


def _fit_regime(kind, tr, prev):
    mu, sd = CF.standardize(tr["X"])
    r = FB.fit_regime(tr["rv"], tr["lH"], tr["ell"], ((tr["X"] - mu) / sd)[:, 0], tr["tid"], tr["n"])
    r.update(mu=mu, sd=sd, d_calm=_expand(r["d_calm"], tr["present"], tr["nT"]), d_crisis=_expand(r["d_crisis"], tr["present"], tr["nT"]))
    return r, prev


def _pred_regime(kind, s, pr):
    x = pr["X"][:, 0]
    z = np.where(np.isfinite(x), (x - s["mu"][0]) / s["sd"][0], 0.0)
    p = expit(s["a"] + s["b"] * z)
    w = p * expit(s["a_calm"] + s["d_calm"][pr["tix"]]) + (1.0 - p) * expit(s["a_crisis"] + s["d_crisis"][pr["tix"]])
    return np.exp(pr["lH"] + w * pr["ell"]), w, {"pi": p}


def three_z(d):
    l2, l3 = d["lK2"] - d["lH"], d["lK3"] - d["lH"]
    return np.stack([l2, np.abs(l2), l3, np.abs(l3)], 1)


def _fit_three(kind, tr, prev):
    Z = three_z(tr)
    mu, sd = CF.standardize(Z)
    r = FB.fit_three(tr["rv"], np.stack([tr["lH"], tr["lK2"], tr["lK3"]], 1), (Z - mu) / sd, tr["tid"], tr["n"], tr["ell"])
    r.update(mu=mu, sd=sd, delta=_expand(r["delta"], tr["present"], tr["nT"]))
    return r, prev


def _pred_three(kind, s, pr):
    z = (three_z(pr) - s["mu"]) / s["sd"]
    w3 = FB.softmax(s["alpha"] + s["delta"][pr["tix"]] + np.einsum("ij,jk->ik", z, s["beta"]))
    F = np.exp((w3 * np.stack([pr["lH"], pr["lK2"], pr["lK3"]], 1)).sum(1))
    return F, w3[:, 1] + w3[:, 2], {"wH": w3[:, 0], "wK2": w3[:, 1], "wK3": w3[:, 2]}


_FIT = {**{k: _fit_linear for k in LINEAR}, "boost": _fit_boost, "regime": _fit_regime, "three": _fit_three}
_PRED = {**{k: _pred_linear for k in LINEAR}, "boost": _pred_boost, "regime": _pred_regime, "three": _pred_three}


def fit(kind, tr, prev):
    return _FIT[kind](kind, tr, prev)


def predict(kind, s, pr):
    return _PRED[kind](kind, s, pr)


def prior_state(kind, nT, k):
    """До первых 60 закрытых пар: вес на M = 0,5 (C9: 1/3 каждому эксперту)."""
    return {"prior": True, "ok": True, "grad": 0.0, "nit": 0}


def predict_prior(kind, pr):
    if kind == "three":
        F = np.exp((pr["lH"] + pr["lK2"] + pr["lK3"]) / 3.0)
        third = np.full(len(F), 1.0 / 3.0)
        return F, 2.0 * third, {"wH": third, "wK2": third, "wK3": third}
    F, w = CF.forward(pr["lH"], pr["ell"], np.zeros(len(pr["lH"])))
    return F, w, {"pi": np.full(len(F), np.nan)} if kind == "regime" else {}


def dev(kind, s, i) -> float:
    """Наибольшее по модулю отклонение параметров бумаги i от панельных (δ, H_i, d_calm, d_crisis…); 0 у исключённой бумаги (LOTO) и у бустинга (параметров по бумагам нет)."""
    if s.get("prior") or kind == "boost":
        return 0.0
    if kind in LINEAR:
        return float(max([abs(s["delta"][i])] + ([float(np.abs(s["eta"][i]).max())] if s["eta"] is not None else [])))
    if kind == "regime":
        return float(max(abs(s["d_calm"][i]), abs(s["d_crisis"][i])))
    return float(np.abs(s["delta"][i]).max())


def alpha_i(kind, s, i) -> float:
    return float(s["alpha"] + s["delta"][i]) if kind in LINEAR and not s.get("prior") else (0.0 if s.get("prior") else float("nan"))


def panel(kind, s, bnames):
    """Поля строки PANEL веса-таблицы (коэффициенты на стандартизованных признаках)."""
    if s.get("prior"):
        return {}
    if kind in LINEAR:
        return {"alpha": s["alpha"], **{"b_" + n: float(b) for n, b in zip(bnames, s["beta"])}}
    if kind == "regime":
        return {"alpha": s["a"], "b_pi_level": s["b"], "alpha_calm": s["a_calm"], "alpha_crisis": s["a_crisis"]}
    if kind == "three":
        return {"alpha_H": s["alpha"][0], "alpha_K2": s["alpha"][1], "alpha_K3": s["alpha"][2],
                **{"b_%s_%s" % (n, e): float(s["beta"][j, i]) for j, n in enumerate(("l2", "abs_l2", "l3", "abs_l3")) for i, e in enumerate(("H", "K2", "K3"))}}
    return {"n_splits": s["n_splits"]}
