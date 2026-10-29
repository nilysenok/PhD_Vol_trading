"""Реверс, раунд 1, лист R-а: строки «по годам» таблицы (а) предрегистрации — Sharpe (ddof=1, √252) и mSR по каждому году периода
для якоря A и A_strogaya, плюс базовая строка «(а) полные», с платой за заём и без. Справочно, без выбора.
Те же дневные ряды, что в r1_a.py: r1_a.schitat((sid, период)) и D.portfel импортом, метрики D.stroka (Д7, metriki_e) — копии логики нет.
Запуск: FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_a_gody.py --period {train4,test1} [--out FILE.csv]
NET (Ж-121): FASTBT_Z2_DIV=NET … --period test1 --out docs/tables/revers_r1/_net/a_po_godam_test1.csv
Контроль: сумма PnL по годам = PnL полного периода (точно, 1e-9); train4 — годы против docs/tables/a_strogaya/pasport.csv (ряд A)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
from fastbt import a_strogaya as AS, zona  # noqa: E402
import dve_metriki_a as D  # noqa: E402
import r1_a  # noqa: E402

GODY = {"train4": (2016, 2017, 2018, 2019), "test1": (2022, 2023, 2024)}
TOL_SUMMA, TOL_PASPORT = 1e-9, 1e-6


def zadaniya(per: str) -> list:
    """Аргументы воркеров: период едёт в каждый (sid, per) — spawn не наследует глобали."""
    return [(s, per) for s in AS.SIDS]


def gody_ryada(dates) -> np.ndarray:
    return np.asarray(dates).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970


def kontrol_summy(x, dates, per: str) -> None:
    """Годы ряда ровно период; сумма PnL по годам = PnL полного периода."""
    god = gody_ryada(dates)
    if tuple(sorted(set(god.tolist()))) != GODY[per]:
        raise ValueError(f"годы ряда {sorted(set(god.tolist()))} ≠ периоду {per} {GODY[per]}")
    s = sum(float(x[god == y].sum()) for y in GODY[per])
    if abs(s - float(np.sum(x))) > TOL_SUMMA:
        raise ValueError(f"сумма по годам {s} ≠ полный период {float(np.sum(x))}")


def stroki_portfelya(imya: str, a: dict, per: str) -> list:
    """(а) полные + по годам; по каждой плате за заём. x с платой — тот же, что в D.stroka."""
    rows, god = [], gody_ryada(a["dates"])
    for plata in (AS.PLATA, 0.0):
        xp = a["x"] - plata * a["baza"]
        kontrol_summy(xp, a["dates"], per)
        rows.append(dict(portfel=imya, stroka="(а) полные", plata_zaem=plata,
                         **D.stroka(a["x"], a["delo"], None, plata, a["baza"])))
        for y in GODY[per]:
            m = god == y
            rows.append(dict(portfel=imya, stroka=f"по году {y}", plata_zaem=plata,
                             **D.stroka(a["x"][m], a["delo"][m], None, plata, a["baza"][m])))
    return rows


def tablica_iz(port: dict, per: str) -> pd.DataFrame:
    """port = {imya: портфельный ряд 'baza' (x, delo, baza, dates)}."""
    return pd.DataFrame([r for imya in ("yakor", "strogaya") for r in stroki_portfelya(imya, port[imya], per)])


def tablica(per: str) -> pd.DataFrame:
    with Pool(2) as p:
        res_s = dict(zip(AS.SIDS, p.map(r1_a.schitat, zadaniya(per))))
    port = {imya: D.portfel({s: res_s[s][imya] for s in AS.SIDS}, "baza") for imya in ("yakor", "strogaya")}
    return tablica_iz(port, per)


def sverka_pasport(df: pd.DataFrame, pas: pd.DataFrame) -> float:
    """train4: макс. |Δ| Sharpe и mSR по годам против паспорта (ряд A, те же плата и год)."""
    d = 0.0
    for _, r in df[df.stroka.str.startswith("по году")].iterrows():
        p = pas[(pas.ryad == "A") & (pas.variant == r.portfel) & (pas.plata_zaem == r.plata_zaem)
                & (pas.god.astype(str) == r.stroka[-4:])].iloc[0]
        d = max(d, abs(p.sharpe - r.sharpe), abs(p.msr - r.msr))
    return float(d)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=tuple(GODY), required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    df = tablica(a.period)
    out = a.out or ROOT / "docs" / "tables" / "revers_r1" / f"a_po_godam_{a.period}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out.write_text(f"# {r1_a.SHAPKA}; период {a.period}, по годам\n# коммит: {kom}\n" + zona.shapka_dividendy()
                   + df.to_csv(index=False), encoding="utf-8")
    print(df.round(3).to_string())
    print("контроль: сумма PnL по годам = полный период (1e-9) — выполнен в stroki_portfelya")
    if a.period == "train4":
        d = sverka_pasport(df, pd.read_csv(ROOT / "docs" / "tables" / "a_strogaya" / "pasport.csv"))
        print(f"сверка с паспортом A по годам: max|Δ| = {d:.2e}")
        if not d < TOL_PASPORT:
            raise SystemExit("train4 по годам не совпадает с паспортом")


if __name__ == "__main__":
    main()
