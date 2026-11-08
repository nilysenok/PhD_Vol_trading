"""V5-0, G7: предварительная диагностика спектра (условие направления 2). Доли энергии полос дня d (DCT-II, low 3k<n, mid n≤3k<2n, high 3k≥2n; полосы объявлены до счёта) и частная корреляция задним числом
y = ln(RV_{t⁺}/F) (F = H или M) с долей полосы дня d при контроле признаков HAR-J дня d (ln RV, ln RV_w, ln RV_m, ln(1+J)) и Intraday Shape V4 (доля последнего часа, Херфиндаль). Бумага — фиксированный эффект. Окна по году цели.
Результат не выбирает гиперпараметров; «направление имеет смысл / нет» — только как решение листа."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_0_dx_ols as O, v5_0_dx_score as SC

BANDS = ("low", "mid", "high")
CONTROLS = ["l_rv", "l_rv_w", "l_rv_m", "l_j", "lasth", "herf"]
BASES = {"H": "k1", "M": "M"}


def shares(b: pd.DataFrame) -> pd.DataFrame:
    """Дневные доли полос и признаки дня d (всё известно к концу d)."""
    d = b.dropna(subset=["rv"]).copy()
    tot = d[["e_low", "e_mid", "e_high"]].sum(axis=1)
    for x in BANDS:
        d["sh_" + x] = d["e_" + x] / tot
    d["l_rv"], d["l_rv_w"], d["l_rv_m"] = np.log(d["rv"]), np.log(d["rv_w"]), np.log(d["rv_m"])
    d["l_j"] = np.log1p(d["j"])
    d["lasth"] = d["rv_close"] / d["rv"]
    return d


def band_stats(d: pd.DataFrame) -> pd.DataFrame:
    """Среднее, разброс, автокорреляция (лаги 1, 5) долей: по бумаге и по годам (пул)."""
    rows = []
    for t, g in d.groupby("ticker"):
        for x in BANDS:
            s = g["sh_" + x]
            rows.append(dict(section="band", item=t, band=x, mean=float(s.mean()), sd=float(s.std()), ac1=float(s.autocorr(1)), ac5=float(s.autocorr(5)), n=int(s.notna().sum())))
    m = d[d["date"].dt.year >= 2017].assign(y=lambda z: z["date"].dt.year).melt(id_vars="y", value_vars=["sh_" + x for x in BANDS])
    for (y, var), s in m.groupby(["y", "variable"])["value"]:
        rows.append(dict(section="band_year", item=str(y), band=var[3:], mean=float(s.mean()), sd=float(s.std()), n=int(s.notna().sum())))
    return pd.DataFrame(rows)


def partial_rows(k: pd.DataFrame, d: pd.DataFrame, lead: int = 0) -> pd.DataFrame:
    """Частные корреляции y с долями mid и high (low = 1 − mid − high) при контролях; lead — только для мутации (доли дня d + lead)."""
    c = S.common(k, ["k1", "k2", "k3", "d2", "M", "K8"])
    c = c[c["group"] == "M16"]
    f = d[["ticker", "date", "sh_mid", "sh_high"] + CONTROLS].sort_values(["ticker", "date"])
    if lead:
        f[["sh_mid", "sh_high"]] = f.groupby("ticker")[["sh_mid", "sh_high"]].shift(-lead)
    c = c.merge(f, on=["ticker", "date"], how="left")
    rows = []
    for base, col in BASES.items():
        c["y"] = np.log(c["rv_tplus"] / c[col])
        for win, (lo, hi) in SC.WINDOWS.items():
            w = c[c["t_plus"].dt.year.between(lo, hi)]
            for x in ("sh_mid", "sh_high"):
                rows.append(dict(section="partial", base=base, window=win, band=x[3:], **O.partial(w, "y", x, CONTROLS)))
    return pd.DataFrame(rows)


def spec_table(k: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    d = shares(b)
    return pd.concat([band_stats(d), partial_rows(k, d)], ignore_index=True)
