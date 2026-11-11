"""W5-A: `W5_TABLE0.csv` и блок E `W5_DIAG.csv` (перенос `w5/table0.py`, логика без изменений; `days.pkl` не используется — окно M передаётся из `x1_w5_diag.build()`).
Общая таблица окна M (Ж-251 п. 2): абсолютный QLIKE сохранённых рядов по окну и половинам; шум Δ — блоковый бутстрап (`models_v2.s5.d.stats_d`, блок 20, B 2000, seed 0) средних разностей
дневных потерь, SE в % QLIKE базы, пары X1−K2, X1−K3, ZB−X1, ZB0−X1. Только сохранённые ряды; обучения нет.
Публичное API: `series(x) -> (x, {имя: np.ndarray})` (стоп при NaN, расхождении дат/RV, cfg8 ≠ колонке C); `windows(x)`; `table(x, s) -> DataFrame` (W5_TABLE0);
`noise(x, s) -> DataFrame` (блок `E_noise`); `head_lines()` — шапка файла W5_TABLE0.csv; `sha(rel)`; `KIND`, `SRC`. Ничего не пишет."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s5.d import stats_d as SD

ROOT = C.ROOT
SRC = {"zb0_a": "data/regen/harj_x1/Z/zb0.parquet", "zb0_n": "data/fwd/models/new/zb0.parquet",
       "cfg8_a": "data/fwd/models/cfg8.parquet", "cfg8_n": "data/fwd/models/new/cfg8.parquet",
       "mplus_S1": "data/regen/harj_x1/W4/mplus_S1.parquet", "mplus_S2": "data/regen/harj_x1/W4/mplus_S2.parquet",
       "ka": "data/regen/harj_x1/W4/ka.parquet"}
KIND = {"K1": "архивная самостоятельная HAR-J (T0)", "K2": "архивная самостоятельная XGBoost (T0)", "K3": "архивная самостоятельная LightGBM (T0)",
        "M": "M = √(K2·K3), фиксированная функция K2, K3", "X1": "гибрид, контрольная точка (DE E1)", "ZB": "гибрид: X1·e^g в днях E (лист Z)",
        "ZB0": "контроль: X1·e^b в днях E (лист Z)", "cfg8": "компонента C гибрида X1 (V7 лист 3)",
        "M+S1": "компонент, не гибрид: M·e^g на S (W4 ступень 1; ступень 1 не пройдена по зарегистрированному порогу 2018–2021; полный гибрид W4 не рассчитывался)",
        "M+S2": "компонент, не гибрид: M·e^g на S + Σ (W4 ступень 1; не пройдена; полный гибрид W4 не рассчитывался)",
        "Ka": "компонент, не гибрид: контроль (а) W4, M·e^g, g = ρ·ln mean(RV/M)"}


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def head_lines():
    return ["# W5_TABLE0 — общая таблица окна M (Ж-251 п. 2): абсолютный QLIKE = среднее дневных потерь l = RV/F − ln(RV/F) − 1 на одних датах; Δ — в % QLIKE базы, минус — лучше базы",
            "# DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; только сохранённые ряды; генератор models_v2.s16.x1_w5_diag_t",
            "# M = √(K2·K3); окно M — общий период K1, K2, K3, X1: t+ 2018-01-04…2026-11-10, n = 2198; половины по году t+",
            "# sha256[:16]: " + "; ".join("%s=%s" % (k, sha(v)) for k, v in {**C.SRC, **SRC}.items())]


def series(x):
    d = x["d"]
    out = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "M": x["M"], "X1": x["F"], "ZB": x["ZB"]}
    for k, a, n in (("ZB0", "zb0_a", "zb0_n"), ("cfg8", "cfg8_a", "cfg8_n")):
        z = pd.concat([pd.read_parquet(ROOT / SRC[a]), pd.read_parquet(ROOT / SRC[n])], ignore_index=True)
        z["d"] = pd.to_datetime(z["d"])
        out[k] = d.map(dict(zip(z["d"], z["F"])))
    for k, s in (("M+S1", "mplus_S1"), ("M+S2", "mplus_S2"), ("Ka", "ka")):
        z = pd.read_parquet(ROOT / SRC[s])
        z["d"] = pd.to_datetime(z["d"])
        if not np.allclose(z["rv"].to_numpy(), x["rv"].to_numpy(), rtol=1e-12) or not (z["d"].to_numpy() == d.to_numpy()).all():
            raise SystemExit("%s: даты или RV не совпадают с окном M" % k)
        out[k] = z["F"].reset_index(drop=True)
    for k, v in out.items():
        if v.isna().any():
            raise SystemExit("%s: нет прогноза на %d датах окна M" % (k, v.isna().sum()))
    if not np.allclose(out["cfg8"].to_numpy(), x["C"].to_numpy(), rtol=1e-12, atol=0):
        raise SystemExit("cfg8 ≠ колонке C x1")
    return x, {k: np.asarray(v, float) for k, v in out.items()}


def windows(x):
    tp, yr = x["t_plus"], x["t_plus"].dt.year
    w = {"M": np.ones(len(x), bool), "h1_2018_2021": (yr <= 2021).to_numpy(), "h2_2022_2026": (yr >= 2022).to_numpy()}
    for y in range(2018, 2027):
        w[str(y)] = (yr == y).to_numpy()
    return w, tp


def table(x, s):
    rv = x["rv"].to_numpy()
    w, tp = windows(x)
    rows = []
    for k, f in s.items():
        l = C.qlike(rv, f)
        for wn in ("M", "h1_2018_2021", "h2_2022_2026"):
            m = w[wn]
            rows.append(dict(series=k, kind=KIND[k], window=wn, t_plus_from=tp[m].min().date(), t_plus_to=tp[m].max().date(), n=int(m.sum()),
                             qlike=l[m].mean(), d_K2_pct=100 * (l[m].mean() / C.qlike(rv[m], s["K2"][m]).mean() - 1),
                             d_K3_pct=100 * (l[m].mean() / C.qlike(rv[m], s["K3"][m]).mean() - 1), d_X1_pct=100 * (l[m].mean() / C.qlike(rv[m], s["X1"][m]).mean() - 1)))
    return pd.DataFrame(rows)


def noise(x, s):
    rv = x["rv"].to_numpy()
    w, _ = windows(x)
    rows = []
    for a, b in (("X1", "K2"), ("X1", "K3"), ("ZB", "X1"), ("ZB0", "X1")):
        la, lb = C.qlike(rv, s[a]), C.qlike(rv, s[b])
        for wn, m in w.items():
            dd = la[m] - lb[m]
            bm = SD.boot_means(dd, SD.block_idx(len(dd), B_=2000, block=20, seed=0))
            q = lb[m].mean()
            rows.append(dict(block="E_noise", state="%s-%s" % (a, b), group="boot block20 B2000 seed0", window=wn, n=int(m.sum()),
                             q_base=q, d_pct=100 * dd.mean() / q, se_pct=100 * bm.std(ddof=1) / q,
                             lo90_pct=100 * np.percentile(bm, 5) / q, hi90_pct=100 * np.percentile(bm, 95) / q,
                             share_days_changed=float((np.abs(dd) > 0).mean())))
    return pd.DataFrame(rows)
