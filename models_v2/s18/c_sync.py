"""S-C-DATA: отчёт о синхронизации (покрытие бумаг и IMOEX на барах SBER, расписания, покрытие разложения). Только чтение данных."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18.c_data import POOL, Inputs


def paper_coverage(inp: Inputs) -> pd.DataFrame:
    """Строка на (год, инструмент): доля баров SBER, где доходность инструмента определена; число сессий без единого такого бара."""
    year = inp.days.year.to_numpy()[inp.day]
    rows = []
    for j, t in enumerate(list(POOL) + ["IMOEX"]):
        x = inp.ri if t == "IMOEX" else inp.R[:, j]
        ok = np.isfinite(x)
        per_day = np.add.reduceat(ok.astype(int), inp.off[:-1])
        for y in np.unique(year):
            m = year == y
            ds = inp.days.year == y
            rows.append(dict(year=int(y), ticker=t, share_bars=ok[m].mean(), n_sessions=int(ds.sum()),
                             sessions_zero=int((per_day[ds] == 0).sum()), first_session=str(inp.days[np.argmax(per_day > 0)].date())))
    return pd.DataFrame(rows)


def schedule(inp: Inputs) -> pd.DataFrame:
    """По годам: сессии, баров SBER (медиана/мин), сессии с < 53 барами, мода первого и последнего слота, исключённые дни по причинам."""
    ex = inp.stats["excluded"]
    first = inp.slot[inp.off[:-1]]
    last = inp.slot[inp.off[1:] - 1]
    rows = []
    for y in np.unique(inp.days.year):
        m = inp.days.year == y
        e = ex[ex["date"].dt.year == y]["reason"].value_counts().to_dict()
        rows.append(dict(year=int(y), sessions=int(m.sum()), bars_median=float(np.median(inp.nbars[m])), bars_min=int(inp.nbars[m].min()),
                         sessions_lt53=int((inp.nbars[m] < 53).sum()), first_slot_mode=int(pd.Series(first[m]).mode()[0]),
                         last_slot_mode=int(pd.Series(last[m]).mode()[0]), first_slot_gt0=int((first[m] > 0).sum()),
                         last_slot_lt52=int((last[m] < 52).sum()), excluded=str(e)))
    return pd.DataFrame(rows)


def short_days(inp: Inputs, nmax: int = 48) -> pd.DataFrame:
    """Сокращённые сессии (баров SBER ≤ nmax): дата, число баров, первый и последний слот."""
    k = np.where(inp.nbars <= nmax)[0]
    return pd.DataFrame(dict(d=inp.days[k], n_bars=inp.nbars[k], first_slot=inp.slot[inp.off[k]], last_slot=inp.slot[inp.off[k + 1] - 1]))


def decomposition_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """По годам и k: покрытие баров и RV, доля common/specific/uncovered от RV, максимумы невязок, перекрёстный член PIT-варианта."""
    d = df.copy()
    d["year"] = d["d"].dt.year
    rows = []
    for (y, k), g in d.groupby(["year", "k_factors"]):
        rv = g["rv"].sum()
        pit = g[~g["flag_nopit"]]
        rows.append(dict(year=int(y), k=int(k), sessions=g["d"].nunique(), cov_bars=g["n_cov"].sum() / g["n_sber_bars"].sum(),
                         common_share=g["rv_common"].sum() / rv, specific_share=g["rv_specific"].sum() / rv, uncovered_share=g["rv_uncovered"].sum() / rv,
                         max_abs_ident=float(g["ident_err"].abs().max()), max_rel_ident=float((g["ident_err"].abs() / g["rv"]).max()),
                         max_rel_ident_pit=float((g["ident_err_pit"].abs() / g["rv"]).max()), max_abs_cross=float(g["cross"].abs().max()),
                         pit_common_share=pit["rv_common_pit"].sum() / max(pit["rv"].sum(), 1e-300),
                         pit_cross_share=pit["cross_pit"].sum() / max(pit["rv"].sum(), 1e-300),
                         pit_abs_cross_share=pit["cross_pit"].abs().sum() / max(pit["rv"].sum(), 1e-300),
                         max_abs_ident_pit=float(g["ident_err_pit"].abs().max())))
    return pd.DataFrame(rows)
