"""Л6: оценка X1 и ZB на новом периоде (протокол HARJ_HYBRID_SBER_FWD, раздел 5).

Семейство из 7 односторонних гипотез «ряд A лучше ряда B» (H1…H7), всегда 7 строк в порядке H1…H7.
Δ_AB — в процентах среднего QLIKE базы B (одно определение, Д7); p — DM Ньюи–Уэст лаг 5 односторонний (`v6_1_eval.dm_one`);
интервал — блочный бутстрап средних разности (`stats_d.block_idx`, `boot_means`), блок, повторы, seed — константы ниже, 90 % двусторонний;
Холм — `s3.dm.holm` по 7 p без дополнения до N программы (мутация № 103: N = числу строк). «Подтверждено» — только при p Холма < α.
Описание (5.4, без вывода): ряды, MCS, месяцы, состояния E/N, ZB против ZB0, M и cfg8 как компоненты X1.
Модуль только считает по переданной рамке общих пар; данных не читает и ничего не пишет.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import dm as dmod
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_1_eval as EV
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION

HYP = (("H1", "X1", "K2"), ("H2", "X1", "K3"), ("H3", "ZB", "K2"), ("H4", "ZB", "K3"), ("H5", "ZB", "X1"), ("H6", "X1", "K1"), ("H7", "ZB", "K1"))
SERIES = ("K1", "K2", "K3", "X1", "ZB")
M_FAMILY = len(HYP)
B, BLOCK, SEED, LAG, CI = 2000, 10, 0, 5, 90
ALPHA, MCS_ALPHA = 0.05, 0.10
CONF, NOT_CONF = "Подтверждено на новом периоде", "Не подтверждено на новом периоде"
FAM_COLS = ["hyp", "A", "B", "n", "d_first", "d_last", "tp_first", "tp_last", "qlike_A", "qlike_B", "delta_pct", "lo90_pct", "hi90_pct", "dm_stat", "p_one", "p_holm", "verdict"]


def _ix(x) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(x).as_unit("ns")


def common_frame(rv: pd.DataFrame, f: dict, extra: dict | None = None, start=ARCHIVE_END, end=LAST_SESSION) -> pd.DataFrame:
    """Общие пары всех пяти рядов: t⁺ ∈ (start; end], rv и все σ̂² конечны и > 0. rv: d, t_plus, rv; f: {ряд: Series σ̂² с индексом d};
    extra: {имя: Series} (e, ZB0, M, cfg8) — выравниваются по d без условия на конечность."""
    fr = pd.DataFrame({"d": _ix(rv["d"]), "t_plus": _ix(rv["t_plus"]), "rv": np.asarray(rv["rv"], float)})
    ok = (fr["t_plus"] > start) & (fr["t_plus"] <= end) & np.isfinite(fr["rv"]) & (fr["rv"] > 0)
    for name in SERIES:
        s = f[name]
        fr[name] = pd.Series(np.asarray(s, float), index=_ix(s.index)).reindex(_ix(fr["d"])).to_numpy(float)
        ok &= np.isfinite(fr[name]) & (fr[name] > 0)
    for name, s in (extra or {}).items():
        fr[name] = pd.Series(np.asarray(s, float), index=_ix(s.index)).reindex(_ix(fr["d"])).to_numpy(float)
    return fr[ok].sort_values("d").reset_index(drop=True)


def _q(fr: pd.DataFrame, name: str) -> np.ndarray:
    return metrics.qlike(fr["rv"].to_numpy(float), fr[name].to_numpy(float))


def _dates(fr: pd.DataFrame) -> dict:
    iso = lambda s, i: "" if len(s) == 0 else str(s.iloc[i].date())  # noqa: E731
    return dict(d_first=iso(fr["d"], 0), d_last=iso(fr["d"], -1), tp_first=iso(fr["t_plus"], 0), tp_last=iso(fr["t_plus"], -1))


def pair_stats(fr: pd.DataFrame, a: str, b: str, idx: np.ndarray | None = None) -> dict:
    """A против B на строках, где оба ряда конечны: n, даты, средние QLIKE, Δ_AB % базы B, 90 % интервал, DM (A лучше), p_one."""
    fr = fr[np.isfinite(fr[a]) & np.isfinite(fr[b])].reset_index(drop=True)
    n = len(fr)
    row = dict(A=a, B=b, n=n, **_dates(fr), **{k: float("nan") for k in ("qlike_A", "qlike_B", "delta_pct", "lo90_pct", "hi90_pct", "dm_stat", "p_one")})
    if n < 2:
        return row
    qa, qb = _q(fr, a), _q(fr, b)
    d, base = qa - qb, float(qb.mean())
    idx = SD.block_idx(n, B_=B, block=BLOCK, seed=SEED) if idx is None or idx.shape[1] != n else idx
    bm = SD.boot_means(d, idx)
    lo, hi = (100 * float(np.percentile(bm, q)) / base for q in ((100 - CI) / 2, 100 - (100 - CI) / 2))
    r = EV.dm_one(fr["rv"], fr[b], fr[a])
    row.update(qlike_A=float(qa.mean()), qlike_B=base, delta_pct=100 * float(d.mean()) / base, lo90_pct=lo, hi90_pct=hi, dm_stat=r["stat"], p_one=r["p_one"])
    return row


def verdict(p_holm: float, delta: float, n: int) -> str:
    """5.3: «подтверждено» только при p Холма < α (и Δ < 0); иначе «не подтверждено (n = …)» — не «опровергнуто»."""
    return CONF if np.isfinite(p_holm) and np.isfinite(delta) and p_holm < ALPHA and delta < 0 else f"{NOT_CONF} (n = {n})"


def holm7(p) -> np.ndarray:
    """Холм по семи p (порядок H1…H7); неопределённое p считается 1. Длина — всегда число гипотез семейства."""
    p = np.asarray(p, float)
    if len(p) != M_FAMILY:
        raise ValueError(f"Холм: {len(p)} p вместо {M_FAMILY} (N = числу строк запрещено)")
    return dmod.holm(np.where(np.isfinite(p), p, 1.0))


def family_table(fr: pd.DataFrame) -> pd.DataFrame:
    """Всегда 7 строк H1…H7. Одна случайная индексация бутстрапа на все гипотезы (зависимость между гипотезами сохраняется)."""
    n = len(fr)
    idx = SD.block_idx(n, B_=B, block=BLOCK, seed=SEED) if n >= 2 else None
    rows = [dict(hyp=h, **pair_stats(fr, a, b, idx)) for h, a, b in HYP]
    ph = holm7([r["p_one"] for r in rows])
    for r, p in zip(rows, ph):
        r["p_holm"] = float(p)
        r["verdict"] = verdict(r["p_holm"], r["delta_pct"], r["n"])
    return pd.DataFrame(rows)[FAM_COLS]


def check_family(fam: pd.DataFrame) -> list:
    """Согласованность таблицы семейства: 7 строк по порядку, Δ из QLIKE, Холм по 7, вывод из p Холма. Пустой список — согласована."""
    bad = []
    if list(fam["hyp"]) != [h for h, _, _ in HYP]:
        return ["строки не H1…H7 по порядку"]
    for _, r in fam.iterrows():
        if np.isfinite(r["qlike_B"]) and abs(r["delta_pct"] - 100 * (r["qlike_A"] - r["qlike_B"]) / r["qlike_B"]) > 1e-9:
            bad.append(f"{r['hyp']}: Δ не из QLIKE (знак или база)")
        if r["verdict"] != verdict(r["p_holm"], r["delta_pct"], int(r["n"])):
            bad.append(f"{r['hyp']}: вывод не по p Холма")
    if not np.allclose(holm7(fam["p_one"]), fam["p_holm"].to_numpy(float), rtol=0, atol=1e-12):
        bad.append("p Холма не по семи p (N ≠ 7)")
    return bad


def series_table(fr: pd.DataFrame) -> pd.DataFrame:
    """Абсолютный QLIKE каждого ряда (пять рядов семейства), n и точные даты."""
    rows = [dict(series=s, role="кандидат" if s in ("X1", "ZB") else "контроль", n=len(fr), **_dates(fr), qlike=float(_q(fr, s).mean()) if len(fr) else float("nan")) for s in SERIES]
    return pd.DataFrame(rows)


def mcs_table(fr: pd.DataFrame) -> pd.DataFrame:
    """MCS 90 % по пяти рядам (`stats_d.mcs`, блок, повторы, seed как у семейства). Описание, не вывод."""
    cols = ["series", "n", "tp_first", "tp_last", "qlike", "p_mcs", "in_mcs"]
    if len(fr) < 2:
        return pd.DataFrame(columns=cols)
    L = np.column_stack([_q(fr, s) for s in SERIES])
    r = SD.mcs(L, SD.block_idx(len(fr), B_=B, block=BLOCK, seed=SEED), alpha=MCS_ALPHA)
    dt = _dates(fr)
    return pd.DataFrame([dict(series=s, n=len(fr), tp_first=dt["tp_first"], tp_last=dt["tp_last"], qlike=float(L[:, j].mean()), p_mcs=float(r["p"][j]), in_mcs=bool(r["in"][j])) for j, s in enumerate(SERIES)])[cols]


def group_table(fr: pd.DataFrame, key: pd.Series, name: str) -> pd.DataFrame:
    """QLIKE пяти рядов и Δ_AB (% базы группы) по группам key (месяц t⁺ или состояние e_d): n, даты. Без p и вывода."""
    rows = []
    for g in pd.unique(key.dropna()):
        sub = fr[(key == g).to_numpy()].reset_index(drop=True)
        qs = {s: float(_q(sub, s).mean()) for s in SERIES}
        dl = {f"delta_{h}": 100 * (qs[a] - qs[b]) / qs[b] for h, a, b in HYP}
        rows.append(dict(**{name: g}, n=len(sub), **_dates(sub), **{f"qlike_{s}": v for s, v in qs.items()}, **dl))
    return pd.DataFrame(rows)


def descr_table(fr: pd.DataFrame) -> pd.DataFrame:
    """ZB против ZB0, X1 против M и X1 против cfg8 (компоненты X1): те же Δ, интервал и DM, но описание без поправки и вывода."""
    pairs = [("ZB", "ZB0"), ("X1", "M"), ("X1", "cfg8")]
    rows = [pair_stats(fr, a, b) for a, b in pairs if a in fr and b in fr]
    return pd.DataFrame(rows)


def all_tables(fr: pd.DataFrame) -> dict:
    """Все таблицы раздела 5: семейство и описание. Требует в рамке столбцы e (состояние e_d), ZB0, M, cfg8."""
    month = fr["t_plus"].dt.strftime("%Y-%m")
    state = fr["e"].map({1.0: "E", 0.0: "N"}) if "e" in fr else pd.Series([None] * len(fr), dtype=object)
    return dict(family=family_table(fr), series=series_table(fr), mcs=mcs_table(fr), months=group_table(fr, month, "month"),
                states=group_table(fr, state, "state"), descr=descr_table(fr), pairs=fr)
