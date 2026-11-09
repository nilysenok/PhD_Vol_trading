"""V6-3, G30: признаки шока и восстановления R = ρ1…ρ6 дня d (план V6, п. 5) по барам M10 основной сессии ТОЛЬКО самого дня d (тиков нет). Единицы: r — десятичные логдоходности, RV_d = Σr² (десятичная), P_k = Σ_{i≤k} r_i / √RV_d, P_0 = 0.
Шок — бар k* с наименьшей r (первый при равенстве), k* = 1…n. ρ1 = (n − k*)/n; ρ2 = (P_n − P_{k*}) / (P_0 − min_k P_k), знаменатель 0 (нет падения ниже открытия) -> ρ2 = 0;
ρ3 = ln(RV после k* на бар / RV до k* на бар), бар k* — в «до»; «после» пусто (k* = n) или сумма в сегменте 0 -> ρ3 = 0; ρ4 = доля объёма баров после k* (объём дня 0 -> ρ4 = ρ1);
ρ5 = ln(RS⁻ второй половины / RS⁻ первой), первая половина — бары с 2·i < n (i с нуля, как у долей s8–s9), пустая половина (RS⁻ = 0) заменяется ε = 1e-12·RV_d; ρ6 = число баров с |r| > 3·√(RV_d / n).
Крайние случаи считаются флагами (число дней — в CHECKPOINT и отчёт). Только бары самого дня: МУТАНТ «бары дня t⁺ в R дня d» (`day_table(...).shift(-1)`) ловится тестом инвариантности к замене баров после d."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_m10 as M

RHO_NAMES = ["rho%d" % i for i in range(1, 7)]
FLAG_NAMES = ["flag_rho2_den0", "flag_rho3_empty", "flag_rho3_zero", "flag_rho4_vol0", "flag_rho5_eps"]
EPS_REL = 1e-12
DEN_TOL = 1e-12          # «знаменатель 0» ρ2 в числах с плавающей точкой: накопленная сумма логдоходностей возвращается к открытию с шумом ~1e-17 (отступление-реализация, считается флагом)


def day_rho(r, vol) -> tuple:
    """-> (ρ1…ρ6, флаги крайних случаев) одного дня; RV дня > 0 обязателен (канонические дни)."""
    r, vol = np.asarray(r, float), np.asarray(vol, float)
    n, q = len(r), r * r
    rv = float(q.sum())
    k = int(np.argmin(r)) + 1
    p = np.r_[0.0, np.cumsum(r) / np.sqrt(rv)]
    rho1 = (n - k) / n
    den = -float(p.min())
    f2 = den <= DEN_TOL
    rho2 = 0.0 if f2 else float((p[n] - p[k]) / den)
    before, after = float(q[:k].sum()) / k, (float(q[k:].sum()) / (n - k) if k < n else 0.0)
    f3e, f3z = k == n, k < n and (before <= 0.0 or after <= 0.0)
    rho3 = 0.0 if (f3e or f3z) else float(np.log(after / before))
    tot = float(vol.sum())
    f4 = tot <= 0.0
    rho4 = rho1 if f4 else float(vol[k:].sum() / tot)
    neg = np.where(r < 0, q, 0.0)
    first = 2 * np.arange(n) < n
    a, b = float(neg[first].sum()), float(neg[~first].sum())
    eps = EPS_REL * rv
    f5 = a <= 0.0 or b <= 0.0
    rho5 = float(np.log((b if b > 0 else eps) / (a if a > 0 else eps)))
    rho6 = float((np.abs(r) > 3.0 * np.sqrt(rv / n)).sum())
    return np.array([rho1, rho2, rho3, rho4, rho5, rho6]), (f2, f3e, f3z, f4, f5)


def day_arrays(bars: pd.DataFrame) -> dict:
    """Бары -> {дата: (r, vol)} основной сессии в порядке баров (те же r и объём, что у S и Σ)."""
    main = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = M.bar_returns(bars)
    x, v = r["r"].to_numpy(float), main["volume"].to_numpy(float)
    return {d: (x[ix], v[ix]) for d, ix in r.groupby("date").indices.items()}


def day_table(arrays: dict) -> pd.DataFrame:
    """{дата: (r, vol)} -> по сессиям ρ1…ρ6 и флаги. Только бары самого дня."""
    rows = {}
    for d, (x, v) in arrays.items():
        rho, fl = day_rho(x, v)
        rows[d] = np.r_[rho, np.array(fl, float)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=RHO_NAMES + FLAG_NAMES).rename_axis("date").sort_index()


def features(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """-> ρ1…ρ6 и флаги на днях панели `daily.index`."""
    return day_table(day_arrays(bars)).reindex(daily.index)
