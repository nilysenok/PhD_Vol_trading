"""Конвейер Ж-043, общие функции: целевая min(mSR 2018, mSR 2019) после 0,04 %. Только train 2018–2019.

Оценщик, сверка с s9 и вспомогательные функции Ж-042 берутся из `zh042_lib` без изменений; движок `fastbt/*` не правится.
Запуск из корня: FASTBT_MAIN=<своя копия> /usr/bin/python3 scripts/zh043_konveyer.py S3.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "scripts"]
import zh042_lib as L0                                  # noqa: E402
from fastbt import metriki_e as me                      # noqa: E402
from fastbt import shag0                                # noqa: E402
from fastbt.sweep_val_baza import s9_engine             # noqa: E402

OUT = L0.Path("docs/tables/zh043")
SHAPKA = "# статус: разработка, train 2018–2019; validation не открыта; Ж-043\n"
TIE, WIN = 0.05, 126
open_sweep, coerce, point_from_row, ref_rows = L0.open_sweep, L0.coerce, L0.point_from_row, L0.ref_rows


def write_csv(df: pd.DataFrame, name: str, extra: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(SHAPKA + extra + df.to_csv(index=False), "utf-8")


def fin(x) -> bool:
    return x is not None and math.isfinite(x)


def obj_msr(m18, m19) -> float:
    """Целевая min(mSR 2018, mSR 2019); любой nan (mSR не определён) -> nan: точка недопустима (толкование Ж-043 п. 1)."""
    return min(m18, m19) if fin(m18) and fin(m19) else float("nan")


def sigma126(x18, x19) -> float:
    """σ скользящего 126-дневного Шарпа на train (2018 + 2019 подряд): шаг 1 день, годовая шкала, ddof=1."""
    s = pd.Series(np.concatenate([x18, x19]), dtype=float)
    r = (s.rolling(WIN).mean() / s.rolling(WIN).std(ddof=1) * math.sqrt(252)).dropna()
    return float(r.std(ddof=1)) if len(r) > 1 else float("nan")


def pick_best(rows: list, score: str = "score", tol: float = TIE) -> dict | None:
    """Лучшая по оценке; точки в пределах tol от лучшей — ничья, выигрывает меньшее σ (Ж-043 п. 2). nan-оценки не участвуют."""
    ok = [r for r in rows if fin(r[score])]
    if not ok:
        return None
    top = max(r[score] for r in ok)
    return min((r for r in ok if r[score] >= top - tol), key=lambda r: (r["sigma126"] if fin(r["sigma126"]) else math.inf))


def year_stats(x) -> dict:
    """Метрики года по ряду избыточной: Шарп, mSR, Сортино, Кальмар, MDD (`metriki_e`)."""
    return dict(sharpe=me.sharpe(x), msr=me.msr(x), sortino=me.sortino_e(x), kalmar=me.kalmar(x), mdd=me.mdd(x))


def evalpt(sw, p: dict) -> dict:
    """Одна проба: 2018 (setup) и 2019 (control) при 0,04 % плюс Шарп при 0,035 % (колонки реестра). Всё из оценщика."""
    a, b = sw.evaluate(p, "setup", 0.04), sw.evaluate(p, "control", 0.04)
    a5, b5 = sw.evaluate(p, "setup", 0.035), sw.evaluate(p, "control", 0.035)
    x18, x19 = np.asarray(a["excess_d"]), np.asarray(b["excess_d"])
    ys = {2018: year_stats(x18), 2019: year_stats(x19)}
    d = dict(obj=obj_msr(ys[2018]["msr"], ys[2019]["msr"]), sigma126=sigma126(x18, x19),
             s2018_0035=a5["by_year"][2018], s2019_0035=b5["by_year"][2019],
             storon_god_2018=a["sdelok_y"], storon_god_2019=b["sdelok_y"],
             sr_d=a["sr_d"], T=a["T"], skew=a["skew"], kurt=a["kurt"])
    for g, s in ys.items():
        d.update({f"{k}_{g}": v for k, v in s.items()})
    d["s2018"], d["s2019"] = ys[2018]["sharpe"], ys[2019]["sharpe"]
    d["cel_ge1"] = int(fin(d["obj"]) and d["obj"] >= 1.0)
    return d


def main_rows(sid: str) -> list:
    """«A только Sk»: полный движок s9, как `sweep_val_baza.S9Baza` (результат режется по году, без обрезки окна по end).

    Окно по end (`ekviv_s9.WINDOWS`) платит выход на последнем дне окна и даёт расхождение 0,001–0,004 с `bazy_sverka`
    (причина названа в отчёте; тест `tests/test_zh043_okno.py`). Печатаются только 2018 и 2019."""
    eng = s9_engine()
    sidr = shag0.sid_row(eng)
    pos = shag0.keep_sids(np.asarray(eng.A_row, float), sidr, [sid])
    ex = {c: eng.port(pos, comm=c)["excess"] for c in L0.COMMS}
    out = []
    for god in (2018, 2019):
        sl = lambda s: s[s.index.year == god]                                  # noqa: E731
        m = (sidr == sid) & (np.asarray(eng.r["year"]) == god)
        tr, gr = L0.sides_gross(pos, eng.lr_row, eng.first, m)
        x = sl(ex[0.0004]).values
        st = L0._stat({c: me.sharpe(sl(v).values) for c, v in ex.items()}, tr, gr)
        out.append(dict(stroka="A_tolko_Sk_s9", god=god, **st, **{f"{k}_posle_004": v for k, v in year_stats(x).items() if k != "sharpe"},
                        msr_ge1=int(fin(me.msr(x)) and me.msr(x) >= 1.0)))
    return out
