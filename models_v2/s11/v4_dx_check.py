"""V4-DX: проверки гейта G26 для `run_v4 --stage DX --check`. Возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_decl as DECL, v4_dx_feat as DF, v4_dx_iso as DXI, v4_dx_report as RP, v4_io as IO, v4_isolation as ISO


def g26(r: dict) -> list:
    bad, Y = [], IO.read_csv("DX_YEARS.csv")
    m16 = Y[Y["group"] == "M16"]
    if len(m16) != 13 * 3 or "SBER" not in set(Y["group"]) or set(Y["m"]) != {"KG", "K2", "K3"} or Y["w_har"].isna().any():
        bad.append("G26: DX_YEARS неполон (%d строк M16)" % len(m16))
    e = 1e-12
    if ((Y["o3"] > Y["o2"] + e) | (Y["o2"] > Y["o1"] + e) | (Y["o1"] > Y["q_best"] + e)).any() or ((Y["share_in_range"] + Y["share_outside"] - 1).abs() > e).any() or (Y["w_har_lo90"] > Y["w_har_hi90"]).any():
        bad.append("G26: нарушен порядок O3 ≤ O2 ≤ O1 ≤ лучшая одиночная / доли / интервал")
    a, o = Y[(Y["group"] == "M16") & (Y["slice"] == "all")].set_index("m"), IO.read_csv("A_ORACLE.csv")
    o = o[(o["ticker"] == "M16") & (o["window"] == "all")].set_index("m")
    if any(abs(a.loc[m, k] - o.loc[m, k]) > 1e-9 for m in ("KG", "K2", "K3") for k in ("o1", "o2", "o3")):
        bad.append("G26: O1–O3 (M16, all) не совпали с A_ORACLE.csv (oracle G7)")
    F = IO.read_csv("DX_FEATURES.csv")
    if F.empty or not set(DF.FEATS) <= set(F["feature"]) or ((F["slice"] == "Y2017") & F["feature"].isin(DF.FEATS)).any():
        bad.append("G26: DX_FEATURES пуст / нет признаков / у 2017 есть квинтили (прошлого окна нет)")
    T = IO.read_csv("DX_2018.csv")
    for s in ("C8", "C1"):
        t = T[T["series"] == s]
        if abs(t[t["kind"] == "month"]["contrib_to_year"].sum() - float(t[t["kind"] == "year"]["contrib_to_year"].iloc[0])) > e or len(t[t["kind"] == "month"]) != 12 or t[t["kind"] == "ticker"].shape[0] != 17:
            bad.append("G26: DX_2018 — вклады месяцев не суммируются в год / не 12 месяцев / не 17 бумаг (%s)" % s)
    txt = (IO.REP / "DX_REPORT.md").read_text()
    if txt.splitlines()[0] != RP.first_line(Y) or len(txt.split()) > 1500 or not all(x in txt for x in ("2022–2025", "Отступления", "2018", "SBER", "Вопрос 6")):
        bad.append("G26: DX_REPORT — первая строка / ≤ 1500 слов (%d) / разделы" % len(txt.split()))
    t0 = dt.datetime.strptime(json.loads(DECL.CP.read_text())["DX_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    if any((IO.REP / n).stat().st_mtime <= t0 for n in ("DX_YEARS.csv", "DX_FEATURES.csv", "DX_2018.csv", "DX_REPORT.md")):
        bad.append("G26: файлы DX созданы не позже объявления")
    return bad + ["G26: диагностика импортирована путём прогноза: %s" % x for x in DXI.violators() + ISO.violators()]


def all_checks(r: dict) -> list:
    return g26(r)
