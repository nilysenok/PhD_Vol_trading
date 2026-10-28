"""Реверс, раунд 1, лист R-а (Ж-117): две метрики (Sharpe ddof=1 √252 и mSR), полный ряд / без 10 худших / без 10 лучших дней /
стоп 2σ / размер 1/σ, для якоря A и A_strogaya. Справочно, без выбора.
Запуск: FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_a.py --period {train4,test1} [--out DIR]
train4 — воспроизводит docs/tables/dve_metriki/a_strogaya_yakor.csv (сверка в конце, допуск 1e-9). test1 = 2022-01-01…2024-12-31,
прогрев 2021; зависит от листа R-P (period="test1" в etap1.build_inputs/evaluate_v2), до его слияния test1 отказывает.
TEST_C и D (замороженные в s9) поверх A: их σ̂ из файлов зоны Z2 даёт R-P; на train4 s9 σ-источники не читает, поэтому здесь
они не считаются (отступление, см. PR)."""
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
from fastbt import a_strogaya as AS, etap1, zona  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402
import dve_metriki_a as D  # noqa: E402

SHAPKA = "реверс раунд 1, 2022–2024, справочно, без выбора"
PERIODY = ("train4", "test1")


def ryad(ev, pt: dict, per: str) -> dict:
    """Как D.ryad, но период параметром."""
    row, x, tik = ev.evaluate_v2(pt, per)
    if x is None:
        raise ValueError(f"{ev.sid} {pt}: {row.get('prichina')}")
    df, inv, eng, _, rs = ev._last
    _, _, dts, _, _, rdates = ev._series(per, ev.comm)
    cat = lambda k: np.concatenate([getattr(r, k) for r in rs])                       # noqa: E731
    keep = np.isin(df.index.year, etap1.years_of(per))
    return dict(x=x, tik=tik, baza=np.asarray(ev.zaem), dates=dts.to_numpy().astype("datetime64[D]"),
                expo=cat("expo"), cnt=cat("cnt"), god=rdates.year.to_numpy(), expo_d=np.bincount(inv, cat("expo"), len(eng.dates))[keep])


def schitat(args: tuple) -> dict:
    sid, per = args
    inp, base, out = etap1.build_inputs(sid, period=per), etap1.base_point(sid), {}
    for imya, strogo in (("yakor", False), ("strogaya", True)):
        cls = D.Strogaya if strogo else D.Yakor
        ev0 = AS.AStrogaya(sid, inp, base, AS.COMM) if strogo else EvaluatorV2(sid, inp, base, AS.COMM)
        ev = cls(sid, inp, base, AS.COMM)
        out[imya] = {"baza": ryad(ev0, {"t_vyhod": "open2"} if strogo else {}, per)}
        for k, pt in D.TOCHKI.items():
            if k == "baza" or (k == "kontrol" and not strogo):
                continue
            out[imya][k] = ryad(ev, {**pt, **(D.STRICT if strogo or k == "stop" else {})}, per)
        print(sid, imya, "готово", flush=True)
    return out


def tablica(per: str) -> pd.DataFrame:
    with Pool(2) as p:
        res_s = dict(zip(AS.SIDS, p.map(schitat, [(s, per) for s in AS.SIDS])))
    rows = []
    for imya in ("yakor", "strogaya"):
        res = {s: res_s[s][imya] for s in AS.SIDS}
        port = {k: D.portfel(res, k) for k in res["S1"]}
        for k, nazv, hv in D.STROKI:
            for plata in (AS.PLATA, 0.0):
                a = port[k]
                rows.append(dict(portfel=imya, stroka=nazv, plata_zaem=plata, **D.stroka(a["x"], a["delo"], hv, plata, a["baza"])))
    return pd.DataFrame(rows)


def sverka(df: pd.DataFrame, obrazec: pd.DataFrame) -> float:
    """Макс. |Δ| по числовым колонкам против образца (train4)."""
    cols = [c for c in obrazec.columns if obrazec[c].dtype.kind == "f"]
    return float(np.abs(df[cols].to_numpy() - obrazec[cols].to_numpy()).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=PERIODY, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "tables" / "revers_r1")
    a = ap.parse_args()
    df = tablica(a.period)
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"a_dve_metriki_{a.period}.csv"
    f.write_text(f"# {SHAPKA}; период {a.period}\n" + zona.shapka_dividendy() + df.to_csv(index=False), encoding="utf-8")
    print(df.round(3).to_string())
    if a.period == "train4":
        obr = pd.read_csv(ROOT / "docs" / "tables" / "dve_metriki" / "a_strogaya_yakor.csv")
        d = sverka(df, obr)
        print(f"сверка с образцом dve_metriki: max|Δ| = {d:.2e}")
        if not d < 1e-9 or list(df.stroka) != list(obr.stroka):
            raise SystemExit("train4 не воспроизводит образец")


if __name__ == "__main__":
    main()
