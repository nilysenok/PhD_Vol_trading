"""V6-3, G31: диагностика обращения времени (направление B, без бюджета; план V6, п. 5). Диагностика задним числом: во вход прогноза НЕ попадает (модуль импортирует только оркестратор `run_v6`, тест AST).
Траектория дня воспроизводится обращённой и признаки S, Σ, R пересчитываются ТЕМИ ЖЕ функциями (`v6_1_feat.raw_features/assemble`, `v6_1b_sig.increments/areas/five_mean`, `v6_3_r.features`) на баре-эквиваленте обращённого дня. Два режима:
`put` — стандартное обращение пути X̃(t) = X(1−t) − X(1): порядок баров обратный, доходности с обратным знаком (цена идёт задом наперёд), объём баров в обратном порядке; `poryadok` — только порядок баров (знак и объём баров прежние: так обращал путь тест 1b).
s1–s3 (ln RV − ln H) и s6 (J/RV) берутся из дневного ряда, где RV = Σr² и BV = Σ|r_k||r_{k−1}| при обращении не меняются: инвариантны по построению (RV проверяется на барах). Дни 2015-07-07 … 2021-12-30 (цели до 2022 не используются)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_m10 as M
from models_v2.s13 import v6_1_feat as FE, v6_1b_sig as SG, v6_3_r as RF

MODES = ("put", "poryadok")
CUT = "2021-12-31"
TOL = 1e-7


def reverse_bars(bars: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Бары основной сессии обращённых дней на той же сетке времени (begin прежний). mode: put | poryadok."""
    main = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    br = M.bar_returns(bars)
    r = br["r"].to_numpy(float)
    o, c, v = main["open"].to_numpy(float).copy(), main["close"].to_numpy(float).copy(), main["volume"].to_numpy(float).copy()
    for ix in br.groupby("date").indices.values():
        x = r[ix][::-1]
        x = -x if mode == "put" else x
        cl = o[ix[0]] * np.exp(np.cumsum(x))
        c[ix], o[ix], v[ix] = cl, np.r_[o[ix[0]], cl[:-1]], v[ix][::-1]
    return pd.DataFrame({"begin": main["begin"], "open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": v})


def sigma_areas(arrays: dict, index) -> pd.DataFrame:
    """Σ (12 признаков: шесть площадей и их средние за 5 сессий) по {дата: (r, vol)} на днях `index`: те же функции, что `v6_1b_sig.day_values/features`, без Σ³."""
    rows = {}
    for d, (x, v) in arrays.items():
        A = SG.areas(SG.increments(x, v)[0])
        rows[d] = [A[i, j] for i, j in SG.PAIRS]
    a = pd.DataFrame.from_dict(rows, orient="index", columns=SG.AREA_NAMES).rename_axis("date").sort_index().reindex(index)
    return pd.concat([a, SG.five_mean(a).set_axis(SG.M5_NAMES, axis=1)], axis=1)


def feature_sets(bars: pd.DataFrame, daily: pd.DataFrame, h: pd.Series) -> dict:
    """-> {группа: датафрейм признаков на днях h.index}: S (s1…s9), Σ (12), R (ρ1…ρ6) из баров `bars` (прямых или обращённых)."""
    S = FE.assemble(FE.raw_features(bars, daily), h)
    ar = RF.day_arrays(bars)
    return {"S": S, "Sigma": sigma_areas(ar, h.index), "R": RF.features(bars, daily)[RF.RHO_NAMES].reindex(h.index)}


def verdict(x: np.ndarray, y: np.ndarray) -> tuple:
    """-> (корреляция, доля дней, где значение меняется, вывод). Меняется: |y − x| > TOL·(1 + |x|); меняет знак: y = −x на всех днях."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ch = float((np.abs(y - x) > TOL * (1 + np.abs(x))).mean())
    flip = bool((np.abs(y + x) <= TOL * (1 + np.abs(x))).all())
    c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return c, ch, ("инвариантен" if ch == 0.0 else "меняет знак" if flip else "меняется"), int(ok.sum())


def diag(bars: pd.DataFrame, daily: pd.DataFrame, h: pd.Series) -> pd.DataFrame:
    """Таблица TR_DIAG: для каждого признака S, Σ, R и режима — корреляция значения на реальной и обращённой траектории, доля изменившихся дней, вывод. Дни ≤ 2021-12-30."""
    bars = bars[bars["begin"] < pd.Timestamp(CUT) + pd.Timedelta(days=1)]
    daily, h = daily.loc[:CUT], h.loc[:CUT]
    real = feature_sets(bars, daily, h)
    rows = []
    for mode in MODES:
        rev = feature_sets(reverse_bars(bars, mode), daily, h)
        for g, df in real.items():
            for f in df.columns:
                c, ch, v, n = verdict(df[f].to_numpy(float), rev[g][f].to_numpy(float))
                rows.append(dict(mode=mode, group=g, feature=f, corr_real_reversed=c, share_days_changed=ch, verdict=v, n_days=n))
    return pd.DataFrame(rows)
