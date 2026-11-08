"""V4-NTP-диагностика, `NTP_STATES.csv` и поведение π: доказательство «добавляет ли HAR-J информацию в каких-то состояниях» (Ж-225 п. 4; 0 конфигураций). Группы — по PIT-доверию NTP3 на дату прогноза (π_t > 0,5 и ≤ 0,5;
π известно до дня цели, не определяется по факту). Внутри группы — D_C8 = QLIKE(C8) − QLIKE(M) и D_K1 = QLIKE(K1) − QLIKE(M) (плюс — ML лучше; M = √(K2·K3)) вне выборки, M16, общие даты; DM (НУ лаг 5) и блочный
бутстрап — по ряду «среднее по бумагам группы на дату» (даты без бумаг группы не входят). Модуль читает RV задним числом — вне путей прогноза (`v4_ntpd_*`, тест изоляции)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST

WINDOWS = ("2017_2021", "2022_2025", "all")
GROUPS = {"pi_gt_half": lambda p: p > 0.5, "pi_le_half": lambda p: p <= 0.5, "all": lambda p: np.ones(len(p), bool)}
ALPHA = 0.05


def frame(r: dict) -> pd.DataFrame:
    """Общие даты (M16): D_C8, D_K1 и PIT-доверие NTP3 строки."""
    c = r["c"][r["c"]["group"] == "M16"]
    c = c.merge(r["fc"]["NTP3"][["ticker", "date", "pi"]], on=["ticker", "date"], how="left")
    qm = _q(c, np.sqrt(c["k2"].to_numpy(float) * c["k3"].to_numpy(float)))
    return c.assign(d_c8=_q(c, c["C8"]) - qm, d_k1=_q(c, c["k1"]) - qm)


def _q(c: pd.DataFrame, f) -> np.ndarray:
    x = c["rv_tplus"].to_numpy(float) / np.asarray(f, float)
    return x - np.log(x) - 1.0


def states(r: dict) -> pd.DataFrame:
    """Окно × группа × сравнение: число строк и дат, доля строк, средний π, Δ, DM p, бутстрап ДИ 95 %, вывод (`adds_info`: Δ < 0, DM p < 0,05 и верхняя граница ДИ < 0)."""
    f = frame(r)
    rows = []
    for w in WINDOWS:
        fw = f[ST.win_mask(f, w)]
        for g, sel in GROUPS.items():
            s = fw[sel(fw["pi"].to_numpy())]
            for name, col in (("C8-M", "d_c8"), ("K1-M", "d_k1")):
                dl = s.groupby("date")[col].mean().sort_index().to_numpy()
                ps = TST.pair_stats(dl)
                rows.append({"window": w, "group": g, "comparison": name, "n_rows": len(s), "n_dates": ps["n_dates"], "share_rows": len(s) / max(len(fw), 1), "mean_pi": float(s["pi"].mean()), **ps,
                             "adds_info": bool(ps["delta_mean"] < 0 and ps["dm_p"] < ALPHA and ps["boot_hi"] < 0) if np.isfinite(ps["dm_p"]) else False})
    return pd.DataFrame(rows)


def answer(st: pd.DataFrame) -> str:
    """Ответ по NTP_STATES: в каких группах и окнах HAR-J (гибрид C8 / чистый K1) значимо лучше ML; если нигде — «≈ 0»."""
    hit = st[st["adds_info"] & st["group"].isin(["pi_gt_half", "pi_le_half"])]
    if hit.empty:
        sg = st[st["group"].isin(["pi_gt_half", "pi_le_half"]) & (st["comparison"] == "C8-M") & st["window"].isin(["2017_2021", "2022_2025"])]
        sgn = "; ".join("%s: Δ(C8 − M) %s в %s" % (w, "< 0" if (g["delta_mean"] < 0).all() else ("> 0" if (g["delta_mean"] > 0).all() else "разных знаков"), "обеих группах") for w, g in sg.groupby("window"))
        return "HAR-J значимого вклада по состояниям не добавляет: ни в группе π > 0,5, ни в группе π ≤ 0,5 разность QLIKE к ML (C8 − M, K1 − M) не отрицательна значимо (DM p < 0,05 и ДИ < 0) ни в одном окне (%s)." % sgn
    return "HAR-J добавляет информацию в состояниях: " + "; ".join("%s, %s, %s (Δ %.4f, DM p %.3f)" % (x.group, x.window, x.comparison, x.delta_mean, x.dm_p) for x in hit.itertuples()) + "."


def pi_behavior(r: dict) -> pd.DataFrame:
    """Средний π (M16) по годам цели и в шоках: 06.04.2018 (±14 дней), 24.02.2022 (28 дней после) — по конфигурациям."""
    rows = []
    for cfg, f in r["fc"].items():
        f = f[f["ticker"].isin(r["c"][r["c"]["group"] == "M16"]["ticker"].unique())]
        for y, g in f.groupby(f["t_plus"].dt.year):
            rows.append({"config": cfg, "period": str(y), "mean_pi": float(g["pi"].mean()), "std_pi": float(g["pi"].std()), "n": len(g)})
        for name, lo, hi in (("shock_2018-04-06_pm14d", "2018-03-23", "2018-04-20"), ("shock_2022-02-24_p28d", "2022-02-24", "2022-03-24")):
            g = f[(f["date"] >= lo) & (f["date"] <= hi)]
            rows.append({"config": cfg, "period": name, "mean_pi": float(g["pi"].mean()), "std_pi": float(g["pi"].std()), "n": len(g)})
    return pd.DataFrame(rows)
