"""S-T --check (только чтение): `problems(t, inp, rep)` — список расхождений; пустой — всё сошлось.
Проверки: (1) QLIKE и Δ таблицы = пересчёт по потерям (знак и база Δ; среднее дневных потерь, не годовых — № 160) и = прежним S_A / S_C / S_R_TABLE до 1e-12;
(2) годы: Σ вкладов = Δ, QLIKE по годам взвешено по n даёт QLIKE окна, совпадение с прежними S_*_YEARS; (3) срезы: основная строка = Δ таблицы (без обрезки), знаменатель = QLIKE базы на оставшихся датах,
пересчёт обрезки через `r_eval_tab.trim_delta`, совпадение с S_R_CONC; (4) обрезки: по 44 даты, Жаккар = общие / объединение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import r_eval_tab as RT, t_slices as TS, t_table as TT

EPS = 1e-12
REL = 1e-11   # прежние CSV записаны с 12 значащими цифрами (%.12g): относительная ошибка печати до 5e-12


def _old(name: str, rep) -> pd.DataFrame:
    return pd.read_csv(rep / name, comment="#")


def _cmp(new: pd.DataFrame, old: pd.DataFrame, keys: list, cols: list, tag: str) -> list:
    m = new.merge(old, on=keys, suffixes=("", "_old"))
    if m.empty:
        return ["%s: нет общих строк с прежней таблицей" % tag]
    bad = []
    for c in cols:
        a, b = m[c].to_numpy(float), m[c + "_old"].to_numpy(float)
        ok = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= EPS + REL * np.abs(b))
        if not ok.all():
            bad.append("%s: %s расходится с прежней таблицей (%d строк)" % (tag, c, int((~ok).sum())))
    return bad


def _table(t, ls, rep) -> list:
    tb, bad = t["table"], []
    ok = tb[tb["status"] == "рассчитан"].rename(columns=lambda c: c.replace("-", "_"))
    for r in ok.itertuples():
        if abs(r.qlike - float(ls[r.series].mean())) > EPS:
            bad.append("QLIKE %s ≠ среднему дневных потерь" % r.series)
        for b in TT.BASES:
            if abs(getattr(r, "d_%s_pct" % b.replace("-", "_")) - 100.0 * (r.qlike - float(ls[b].mean())) / float(ls[b].mean())) > 1e-9:
                bad.append("Δ %s к %s: знак или база" % (r.series, b))
    if set(tb[tb["status"] != "рассчитан"]["series"]) != set(TT.MISSING):
        bad.append("строки «не рассчитывался» ≠ S-RA0, S-RA1, S-RCD")
    new = tb.rename(columns=lambda c: c.replace("-", "_"))
    for f in ("S_A_TABLE.csv", "S_C_TABLE.csv", "S_R_TABLE.csv"):
        o = _old(f, rep)
        o = o[o["role"].isin(["main", "reference"])].rename(columns=lambda c: c.replace("-", "_"))
        bad += _cmp(new, o, ["series"], ["qlike", "d_X1_pct", "d_W5_G_pct", "d_K2_pct", "d_K3_pct"], "TABLE vs " + f)
    return bad


def _years(t, rep) -> list:
    y, bad = t["years"], []
    p = y[y["base"] != "-"]
    gap = max(abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) for _, g in p.groupby(["series", "base", "kind"]))
    if gap > EPS:
        bad.append("Σ вкладов ≠ Δ (%.3g)" % gap)
    s = y[(y["base"] == "-") & (y["kind"] == "period")]
    q = t["table"].set_index("series")["qlike"]
    for k, g in s.groupby("series"):
        if abs(float((g["n"] * g["q_series"]).sum() / g["n"].sum()) - q[k]) > EPS:
            bad.append("QLIKE окна %s ≠ среднему дневных потерь по годам (взвешено по n)" % k)
    for f in ("S_A_YEARS.csv", "S_C_YEARS.csv", "S_R_YEARS.csv"):
        bad += _cmp(y, _old(f, rep), ["series", "base", "period"], ["n", "q_series", "q_base", "delta_pct", "contrib_pct", "delta_total_pct"], "YEARS vs " + f)
    return bad


def _slices(t, inp, rep) -> list:
    s, ls, tp, bad = t["slices"], inp["ls"], pd.DatetimeIndex(inp["tp"]), []
    tb = t["table"].set_index("series")
    for (h, b), g in s.groupby(["candidate", "base"], sort=False):
        d = ls[h] - ls[b]
        for r, m in zip(g.itertuples(), TS.masks(tp, d)):
            if r.n != int(m.sum()) or abs(r.q_base - float(ls[b][m].mean())) > EPS:
                bad.append("%s|%s|%s: n или знаменатель" % (h, b, r.slice))
        r0 = g.iloc[0]
        want = 100.0 * (float(ls[h].mean()) - float(ls[b].mean())) / float(ls[b].mean())
        if abs(r0["delta_pct"] - want) > 1e-9 or r0["kind"] != "основной" or r0["n"] != len(tp):
            bad.append("%s|%s: основная строка не равна Δ без обрезки на всех датах" % (h, b))
        if b in TT.BASES and abs(r0["delta_pct"] - tb.at[h, "d_%s_pct" % b]) > 1e-9:
            bad.append("%s|%s: основная строка ≠ таблице" % (h, b))
        if abs(g.iloc[3]["delta_pct"] - RT.trim_delta(ls[h], ls[b])) > 1e-9:
            bad.append("%s|%s: обрезка ≠ trim_delta" % (h, b))
    c = _old("S_R_CONC.csv", rep)
    for item, key, i in (("no_day", "2022-02-24", 1), ("no_year", "2022", 2), ("trim", "K=22", 3)):
        o = c[(c["series"] == "S-RCK") & (c["base"] == "W5-G") & (c["item"] == item) & (c["key"] == key)]["value"].iloc[0]
        v = s[(s["candidate"] == "S-RCK") & (s["base"] == "W5-G")].iloc[i]["delta_pct"]
        if abs(v - o) > 1e-9:
            bad.append("срез %s S-RCK|W5-G ≠ S_R_CONC" % item)
    return bad


def _trimsets(t) -> list:
    ts, bad = t["trimsets"], []
    pr = ts[ts["kind"] == "пара сравнений"]
    if len(pr) != 28 or (pr["n_each"] != 2 * TS.KTRIM).any():
        bad.append("обрезки: не 28 пар или не по 44 даты")
    if (abs(pr["jaccard"] - pr["n_common"] / (2 * 44 - pr["n_common"])) > EPS).any():
        bad.append("Жаккар ≠ общие / объединение")
    ls_ = ts[ts["kind"] == "перечень дат"]
    if len(ls_) != 4 or any(len(d.split("; ")) != TS.KTRIM for d in ls_["dates"]):
        bad.append("перечень дат: не 22 + 22 на сравнение")
    return bad


def problems(t: dict, inp: dict, rep=TT.REP) -> list:
    return _table(t, inp["ls"], rep) + _years(t, rep) + _slices(t, inp, rep) + _trimsets(t)
