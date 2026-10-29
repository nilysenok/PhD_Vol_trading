"""Реверс, раунд 1, лист R-б: лонг/шорт, дивиденды, комиссия, вклад ФДР для A_zashch (A (s9), ежегодный top-10 ансамбль; Ж-124 п. 1) и A_strogaya.
Запуск: FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_b.py --period {train4,test1} [--yakor A_zashch|A_fix] [--procs N]
--yakor A_zashch (умолчание, содержательный счёт): ряд — scripts/revers/r1_a_zashch.py (кэш s9 → s9.Engine.port(A_row)); строки варианта
  считаются из позиций A_row (длинные/короткие — срез знака итоговых позиций; div0 — те же позиции, цены без дивиденда; ФДР — fdr.noch_iz_pos;
  в модели s9 позиция держится ночью целиком, поэтому «ФДР t0» ≡ база — это свойство модели, не пропуск). test1: база сверяется с a_cd_test1.csv ≤ 1e-12.
--yakor A_fix — фиксированная точка этапа 1 (EvaluatorV2), только якорь кода: сверка с docs/tables/a_strogaya/razlozhenie.csv (#476) на train4
  и с опубликованным b_razlozhenie.csv на test1; в содержательные таблицы не входит.
A_strogaya (строка `strogaya`) — как было. Расчёт и определения — scripts/analysis/razlozhenie_a.py (#476), fastbt/metriki_e.py (Д7).
train4 пишет в --out (по умолчанию scratch-имя, образец не затирается); test1 — docs/tables/revers_r1/b_razlozhenie.csv.
Колонки *_ned (недельные метрики, Ж-124 п. 3) — в конце, старые не меняются."""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(_p))
import razlozhenie_a as RA  # noqa: E402  (подмена run_e1 для записи сделок — только внутри RA.zapis_sdelok())
import r1_a_zashch as Z  # noqa: E402
import shapka_lib as SL  # noqa: E402
from fastbt import a_strogaya as AS, etap1, metriki_e as me, s9, zona  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402

OUT = ROOT / "docs" / "tables" / "revers_r1"
SHAPKA = "# реверс раунд 1, 2022–2024, справочно, без выбора"
YAKORI = ("A_zashch", "A_fix")
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
    """Все варианты для одного Sk (в процессе пула период ставится заново: spawn не наследует глобали).
    A_fix считается только при yakor == 'A_fix'; A_strogaya — всегда."""
    sid, period, *rest = arg
    yakor = rest[0] if rest else "A_fix"              # старый вызов (sid, period) — оба оценщика, как раньше; main передаёт якорь явно
    ustanovit_period(period)
    inp, base, out = etap1.build_inputs(sid, period=period), etap1.base_point(sid), {"A_fix": {}, "strogaya": {}}
    with RA.zapis_sdelok():
        for imya, cls, strogo in (("A_fix", EvaluatorV2, False), ("strogaya", AS.AStrogaya, True)):
            if imya == "A_fix" and yakor != "A_fix":
                continue
            for v, kw in VARIANTY.items():
                i2 = RA.replace(inp, div={t: np.zeros_like(np.asarray(d, float)) for t, d in inp.div.items()}) if kw.get("div0") else inp
                ev = RA.klass(cls)(sid, i2, base, kw.get("comm", AS.COMM), fdr=kw.get("fdr"))
                ev.storona, ev.sdelki = kw.get("storona", 0), {}
                RA.TEK["ev"] = ev
                out[imya][v] = AS.schitat(ev, strogo)
            print(sid, imya, "готово", flush=True)
    return out


def portfeli_zashch(period: str, div: str | None = None, kesh: dict | None = None) -> dict:
    """{вариант: {x, baza, dates}} для A_zashch; все семь вариантов считаются (div0 и ФДР — см. docstring r1_a_zashch)."""
    out = {}
    for v in Z.VARIANTY:
        p, _ = Z.ryad_i_pozicii(period, div, v, kesh=kesh)
        out[v] = dict(x=p["excess"].to_numpy(float), baza=p["baza"].to_numpy(float), dates=p.index.to_numpy().astype("datetime64[D]"))
    return out


def portfeli_pary(per: dict, imya: str) -> dict:
    res = {s: per[s][imya] for s in AS.SIDS}
    return {v: RA.portfel(res, v) for v in VARIANTY}


def metriki(x: np.ndarray, baza: np.ndarray, plata: float) -> dict:
    return RA.metriki(x, baza, plata)


def tablica(ports: dict) -> pd.DataFrame:
    """ports = {портфель: {вариант: {x, baza, dates}}}. Строки: портфель × вариант × плата за шорт (0 и 5 %), метрики mSR / Шарп ddof=1 √252 /
    Σ PnL %; вклад = строка − база. В конце — недельные метрики (me.stroka_ned)."""
    rows = []
    for imya, port in ports.items():
        for plata in (0.0, AS.PLATA):
            m0 = metriki(port["baza"]["x"], port["baza"]["baza"], plata)
            for v, nazv in STROKI:
                p = port[v]
                m = metriki(p["x"], p["baza"], plata)
                rows.append(dict(portfel=imya, stroka=nazv, kod=v, plata_zaem=plata, **m,
                                 d_msr=m["msr"] - m0["msr"], d_sharpe=m["sharpe"] - m0["sharpe"], d_sum_pnl_pct=m["sum_pnl_pct"] - m0["sum_pnl_pct"],
                                 **me.stroka_ned(p["x"], p["dates"], None, plata, p["baza"])))
    return pd.DataFrame(rows)


def sverka(df: pd.DataFrame, tol: float) -> float:
    """train4: общие строки A_fix / strogaya (kod, плата) против образца #476; возвращает макс |Δ| по mSR/Шарпу/Σ PnL."""
    obr = pd.read_csv(ROOT / "docs" / "tables" / "a_strogaya" / "razlozhenie.csv")
    obr = obr.drop_duplicates(["portfel", "kod", "plata_zaem"])
    m = df.assign(portfel=df.portfel.replace({"A_fix": "yakor"})).merge(obr, on=["portfel", "kod", "plata_zaem"], suffixes=("", "_obr"))
    if m.empty:
        raise SystemExit("сверка: нет общих строк с образцом")
    d = max(float((m[c] - m[c + "_obr"]).abs().max()) for c in ("msr", "sharpe", "sum_pnl_pct"))
    if d > tol:
        raise SystemExit(f"сверка с образцом: макс |Δ| = {d:.2e} > допуск {tol:.0e}")
    return d


def sverka_a_cd(df: pd.DataFrame, a_cd: Path, tol: float = 1e-12) -> float:
    """test1: строка A_zashch/база/плата 0 против строки «A (s9) (а) полные» из a_cd (Шарп и mSR); иначе отказ (A_zashch молча ≠ A (s9))."""
    o = pd.read_csv(a_cd, comment="#")
    o = o[o.portfel.str.startswith("A (s9") & (o.stroka == "(а) полные")].iloc[0]
    r = df[(df.portfel == "A_zashch") & (df.kod == "baza") & (df.plata_zaem == 0)].iloc[0]
    d = 0.0
    for c in ("sharpe", "msr"):
        a, b = float(r[c]), float(o[c])
        if np.isnan(a) != np.isnan(b):
            raise SystemExit(f"сверка с {a_cd.name}: NaN в {c} не совпал")
        d = d if np.isnan(a) else max(d, abs(a - b))
    if d > tol:
        raise SystemExit(f"сверка A_zashch с {a_cd.name}: макс |Δ| = {d:.2e} > допуск {tol:.0e}")
    return d


def zapisat(out: Path, df: pd.DataFrame, period: str, yakor: str) -> None:
    """Шапка: статус, коммит кода, входы с путями и sha256 (proverka.md п. 4)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    sh = SHAPKA + f", период {period}; якорь {yakor}\n" + SL.blok(period) + zona.shapka_dividendy()
    out.write_text(sh + df.to_csv(index=False), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=("train4", "test1"), required=True)
    ap.add_argument("--yakor", choices=YAKORI, default="A_zashch")
    ap.add_argument("--out", type=Path, default=None, help="файл таблицы (по умолчанию docs/tables/revers_r1/b_razlozhenie.csv для test1)")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--procs", type=int, default=2, help="процессов Pool для A_strogaya / A_fix (по Sk), ≤ 6")
    ap.add_argument("--sobrat-kesh", action="store_true", help="собрать кэш s9 периода, если с текущим ключом его нет (2 процесса)")
    a = ap.parse_args(argv)
    ustanovit_period(a.period)
    if a.yakor == "A_zashch":
        Z.kesh_s9(a.period, sobrat=a.sobrat_kesh)           # кэш s9 до Pool: сборка внутри воркера спаунила бы детей из daemonic-процесса
    with Pool(max(1, min(a.procs, len(AS.SIDS)))) as p:
        per = dict(zip(AS.SIDS, p.map(schitat, [(s, a.period, a.yakor) for s in AS.SIDS])))
    ports = {a.yakor: portfeli_zashch(a.period) if a.yakor == "A_zashch" else portfeli_pary(per, "A_fix"), "strogaya": portfeli_pary(per, "strogaya")}
    df = tablica(ports)
    if a.period == "train4" and a.yakor == "A_fix":
        print(f"сверка с образцом #476: макс |Δ| = {sverka(df, a.tol):.2e} (допуск {a.tol:.0e})")
    if a.period == "test1" and a.yakor == "A_zashch":
        net = OUT / "_net" / "a_cd_test1.csv" if zona.div_net() else OUT / "a_cd_test1.csv"
        print(f"сверка A_zashch с {net.name}: макс |Δ| = {sverka_a_cd(df, net):.2e}")
    out = a.out or OUT / "b_razlozhenie.csv"
    zapisat(out, df, a.period, a.yakor)
    print(df.to_string())


if __name__ == "__main__":
    main()
