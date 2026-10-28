"""Реверс, раунд 1, лист R-б: лонг/шорт, дивиденды, комиссия, вклад ФДР для якоря A и A_strogaya.
Запуск: FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_b.py --period {train4,test1}  (нужны data/fastbt/s9/cache_latest_*.txt).
Расчёт — то же, что scripts/analysis/razlozhenie_a.py (#476): тот же оценщик и те же определения (fastbt/metriki_e.py, Д7).
train4 пишет в --out (по умолчанию scratch-имя, образец не затирается) и сверяет с docs/tables/a_strogaya/razlozhenie.csv;
test1 — только после снятия замка зоны Z2 и слияния листа R-P (период test1 в s9/a_strogaya), иначе отказ.
Таблицы: docs/tables/revers_r1/b_razlozhenie.csv (шапка-комментарий «реверс раунд 1, 2022–2024, справочно, без выбора»)."""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import razlozhenie_a as RA  # noqa: E402  (подмена run_e1 для записи сделок — только в этом процессе)
from fastbt import a_strogaya as AS, etap1, metriki_e as me, s9  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402

OUT = ROOT / "docs" / "tables" / "revers_r1"
SHAPKA = "# реверс раунд 1, 2022–2024, справочно, без выбора"
FDR = ("t0", "t1")
STROKI = [("baza", "база: комиссия 0,04 %, дивиденды, обе стороны"), ("long", "только лонги"), ("short", "только шорты"),
          ("comm0", "без комиссии"), ("div0", "без дивидендов"), ("fdr_t0", "ФДР t0 (занятость на 18:50)"),
          ("fdr_t1", "ФДР t1 (занятость на 18:50, T+1)")]
VARIANTY = {"baza": {}, "long": {"storona": 1}, "short": {"storona": -1}, "comm0": {"comm": 0.0}, "div0": {"div0": True},
            "fdr_t0": {"fdr": "t0"}, "fdr_t1": {"fdr": "t1"}}


def ustanovit_period(period: str) -> None:
    """Период в a_strogaya (PER/GODY); test1 без поддержки R-P — отказ, а не молчаливый train4."""
    if period not in s9.PERIODS_S9:
        raise SystemExit(f"период {period!r} не поддержан в s9.PERIODS_S9 ({sorted(s9.PERIODS_S9)}): нужен лист R-P")
    AS.PER, AS.GODY = period, s9.PERIODS_S9[period][0]


def schitat(arg: tuple) -> dict:
    """Все варианты для одного Sk (в процессе пула период ставится заново: spawn не наследует глобали)."""
    sid, period = arg
    ustanovit_period(period)
    inp, base, out = etap1.build_inputs(sid, period=period), etap1.base_point(sid), {"yakor": {}, "strogaya": {}}
    for imya, cls, strogo in (("yakor", EvaluatorV2, False), ("strogaya", AS.AStrogaya, True)):
        for v, kw in VARIANTY.items():
            i2 = RA.replace(inp, div={t: np.zeros_like(np.asarray(d, float)) for t, d in inp.div.items()}) if kw.get("div0") else inp
            ev = RA.klass(cls)(sid, i2, base, kw.get("comm", AS.COMM), fdr=kw.get("fdr"))
            ev.storona, ev.sdelki = kw.get("storona", 0), {}
            RA.TEK["ev"] = ev
            out[imya][v] = AS.schitat(ev, strogo)
        print(sid, imya, "готово", flush=True)
    return out


def metriki(x: np.ndarray, baza: np.ndarray, plata: float) -> dict:
    return RA.metriki(x, baza, plata)


def tablica(per: dict) -> pd.DataFrame:
    """Строки: портфель × вариант × плата за шорт (0 и 5 %), метрики mSR / Шарп ddof=1 √252 / Σ PnL %; вклад = строка − база."""
    rows = []
    for imya in ("yakor", "strogaya"):
        res = {s: per[s][imya] for s in AS.SIDS}
        port = {v: RA.portfel(res, v) for v in VARIANTY}
        for plata in (0.0, AS.PLATA):
            m0 = metriki(port["baza"]["x"], port["baza"]["baza"], plata)
            for v, nazv in STROKI:
                m = metriki(port[v]["x"], port[v]["baza"], plata)
                rows.append(dict(portfel=imya, stroka=nazv, kod=v, plata_zaem=plata, **m,
                                 d_msr=m["msr"] - m0["msr"], d_sharpe=m["sharpe"] - m0["sharpe"], d_sum_pnl_pct=m["sum_pnl_pct"] - m0["sum_pnl_pct"]))
    return pd.DataFrame(rows)


def sverka(df: pd.DataFrame, tol: float) -> float:
    """train4: общие строки (kod, плата) против образца #476; возвращает макс |Δ| по mSR/Шарпу/Σ PnL."""
    obr = pd.read_csv(ROOT / "docs" / "tables" / "a_strogaya" / "razlozhenie.csv")
    obr = obr.drop_duplicates(["portfel", "kod", "plata_zaem"])
    m = df.merge(obr, on=["portfel", "kod", "plata_zaem"], suffixes=("", "_obr"))
    if m.empty:
        raise SystemExit("сверка: нет общих строк с образцом")
    d = max(float((m[c] - m[c + "_obr"]).abs().max()) for c in ("msr", "sharpe", "sum_pnl_pct"))
    if d > tol:
        raise SystemExit(f"сверка с образцом: макс |Δ| = {d:.2e} > допуск {tol:.0e}")
    return d


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=("train4", "test1"), required=True)
    ap.add_argument("--out", type=Path, default=None, help="файл таблицы (по умолчанию docs/tables/revers_r1/b_razlozhenie.csv для test1)")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    ustanovit_period(a.period)
    with Pool(2) as p:
        per = dict(zip(AS.SIDS, p.map(schitat, [(s, a.period) for s in AS.SIDS])))
    df = tablica(per)
    if a.period == "train4":
        print(f"сверка с образцом #476: макс |Δ| = {sverka(df, a.tol):.2e} (допуск {a.tol:.0e})")
    out = a.out or OUT / "b_razlozhenie.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(SHAPKA + f", период {a.period}\n" + df.to_csv(index=False), encoding="utf-8")
    print(df.to_string())


if __name__ == "__main__":
    main()
