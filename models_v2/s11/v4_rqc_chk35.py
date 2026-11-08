"""V4-RQC: проверки гейта G35 (прогнозы K8, RQC1–RQC3) на реальных данных: K8 побитно с независимым пересчётом, эталон a (медленный пересчёт), обрезка, мутации через боевой путь, свойства RQC3 (шаг, заморозка).
Возвращают список нарушений (пусто = OK). Независимый пересчёт заморозки — свой код (без `v4_rqc_r3.prep`)."""
from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pandas as pd

from models_v2.s11 import v4_rqc_calib as CAL, v4_rqc_r3 as R3, v4_rqc_run as R

DAY = pd.Timestamp("2019-06-03")
STEP_DECL, FREEZE_DECL = 0.02, 5            # объявленные значения (не импортируются из кода прогноза)


def sorted_panel(P: pd.DataFrame, Zl: pd.DataFrame) -> pd.DataFrame:
    return R.prepare(P, Zl)[0]


def g35_k8_and_forecasts(o: dict) -> list:
    bad, P = [], sorted_panel(o["P"], o["Zl"])
    k8 = o["fc"]["K8"]
    ref = np.array([math.sqrt(h * math.sqrt(a * b)) for h, a, b in zip(P["h"], P["k2"], P["k3"])])
    if not (np.array_equal(k8["ticker"].to_numpy(), P["ticker"].to_numpy()) and np.array_equal(k8["F"].to_numpy(), ref)):
        bad.append("G35: K8 не равен sqrt(H·sqrt(K2·K3)) побитно (независимый пересчёт)")
    for cfg in R.CFGS:
        f = o["fc"][cfg]
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all():
            bad.append("G35: %s — прогнозы неполны / не конечны" % cfg)
        if not f["a"].between(CAL.A_LO - 1e-12, CAL.A_HI + 1e-12).all():
            bad.append("G35: %s — a вне [0,6; 1,6]" % cfg)
    return bad


def ref_a(P: pd.DataFrame, cfg: str) -> np.ndarray:
    """Медленный эталон a по определению: для каждого дня — mean(RV/K8) по закрытым (t⁺ ≤ D) парам панели; RQC2 — по парам с годом t⁺ = Y−1."""
    x = (P["rv"] / np.array([math.sqrt(h * math.sqrt(a * b)) for h, a, b in zip(P["h"], P["k2"], P["k3"])])).to_numpy()
    tp, dt_ = P["t_plus"].to_numpy(), P["date"].to_numpy()
    out = np.ones(len(P))
    for d in np.unique(dt_):
        m = tp <= d if cfg == "RQC1" else (pd.DatetimeIndex(tp).year == pd.Timestamp(d).year - 1)
        out[dt_ == d] = x[m].mean() if m.sum() >= CAL.MIN_PAIRS else 1.0
    return np.clip(out, CAL.A_LO, CAL.A_HI)


def g35_reference(o: dict) -> list:
    bad, P = [], sorted_panel(o["P"], o["Zl"])
    for cfg in ("RQC1", "RQC2"):
        if not np.allclose(o["fc"][cfg]["a"].to_numpy(), ref_a(P, cfg), rtol=1e-12, atol=0):
            bad.append("G35: %s — a не совпадает с медленным эталоном по определению" % cfg)
    with mock.patch.object(CAL, "closed_date", lambda tp, d: d):          # мутация «закрыто + 1»: RQC2 на границе года отличается от эталона
        if np.allclose(R.forecast("RQC2", o["P"], o["Zl"])[0]["a"].to_numpy(), ref_a(P, "RQC2"), rtol=1e-12, atol=0):
            bad.append("G35: мутация «закрыто + 1» не поймана на RQC2")
    return bad


def g35_pit(o: dict) -> list:
    """Обрезка (подмена RV пар, закрытых после d) на реальных данных; мутации через боевой путь."""
    bad, P, Z = [], o["P"], o["Zl"]
    for cfg in ("RQC1", "RQC2", "RQC3"):
        if R.pit_violation(cfg, P, Z, DAY):
            bad.append("G35: %s — подмена RV после d меняет F дня ≤ d" % cfg)
    with mock.patch.object(CAL, "closed_date", lambda tp, d: d):
        bad += ["G35: мутация «закрыто + 1» не поймана на %s" % c for c in ("RQC1", "RQC3") if not R.pit_violation(c, P, Z, DAY)]
    with mock.patch.object(CAL, "ref_year", lambda y: y):
        bad += ["G35: мутация «год Y вместо Y−1» не поймана на RQC2"] if not R.pit_violation("RQC2", P, Z, DAY) else []
    with mock.patch.object(R3, "window", lambda hist, full: full):
        bad += ["G35: мутация «процентили по будущему» не поймана на RQC3"] if not R.pit_violation("RQC3", P, Z, DAY) else []
    return bad


def expected_frozen(P: pd.DataFrame) -> np.ndarray:
    """Независимо: день-скачок бумаги (доля J > p90 или |Δ ln RV| > p95 по прошлым дням, ≥ 60) и следующие 4 строки бумаги."""
    out = np.zeros(len(P), bool)
    for t, g in P.groupby("ticker"):
        js, dl, idx, hit = g["jshare"].to_numpy(float), np.abs(g["dlrv"].to_numpy(float)), g.index.to_numpy(), []
        for r in range(len(g)):
            pj, pd_ = js[:r][np.isfinite(js[:r])], dl[:r][np.isfinite(dl[:r])]
            up = (len(pj) >= 60 and np.isfinite(js[r]) and js[r] > np.quantile(pj, 0.90)) or (len(pd_) >= 60 and np.isfinite(dl[r]) and dl[r] > np.quantile(pd_, 0.95))
            if up:
                hit.append(r)
        for r in hit:
            out[idx[r: r + FREEZE_DECL]] = True
    return out


def rqc3_props(out: pd.DataFrame, exp_fz: np.ndarray) -> list:
    """Шаг |Δ ln a| ≤ 0,02 и неизменность ln a в дни заморозки (по независимому пересчёту) — на строках (date, ticker)."""
    bad, la = [], np.log(out["a"].to_numpy())
    o = out.reset_index(drop=True)
    for t, g in o.groupby("ticker"):
        l = la[g.index.to_numpy()]
        d = np.diff(l, prepend=0.0)
        if np.abs(d).max() > STEP_DECL + 1e-12:
            bad.append("шаг > 0,02 у " + t)
        if (np.abs(d[exp_fz[g.index.to_numpy()]]) > 0).any():
            bad.append("a меняется в дни заморозки у " + t)
    return bad


def g35_rqc3(o: dict) -> list:
    bad, P, Z = [], sorted_panel(o["P"], o["Zl"]), o["Zl"]
    fz, out = expected_frozen(P), o["fc"]["RQC3"]
    bad += ["G35: RQC3 — " + x for x in rqc3_props(out, fz)]
    if not np.array_equal(out["frozen"].to_numpy(), fz):
        bad.append("G35: RQC3 — флаги заморозки прогноза не равны независимому пересчёту")
    pre = R.make_pre(o["P"], Z)
    for name, val in (("FREEZE", 0), ("STEP", 0.05)):
        with mock.patch.object(R3, name, val):
            m = R.forecast("RQC3", o["P"], Z, pre=pre)[0]
        if not rqc3_props(m, fz):
            bad.append("G35: мутация RQC3 %s=%s не поймана" % (name, val))
    k8 = o["fc"]["K8"]["F"].to_numpy()
    for cfg in ("RQC1", "RQC2", "RQC3"):
        if not np.array_equal(R.forecast(cfg, o["P"], Z, zero_a=True, pre=pre)[0]["F"].to_numpy(), k8):
            bad.append("G35: %s при a ≡ 1 не равен K8 побитно" % cfg)
    if not o["loto"][o["loto"]["held_out"] != "M16"]["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G35: LOTO — исключённая бумага влияет на панельные величины")
    return bad


def all_g35(o: dict) -> list:
    return g35_k8_and_forecasts(o) + g35_reference(o) + g35_pit(o) + g35_rqc3(o)
