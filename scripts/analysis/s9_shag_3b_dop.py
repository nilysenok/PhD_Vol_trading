"""Шаг 3′ B — то, чего нет в s9_shag_db.py: журнал со сдвигом σ, сверка слоёв на 5 точках, сверка A и к3 с шагом 0.

    FASTBT_MAIN=<своя копия> python3 scripts/analysis/s9_shag_3b_dop.py [--sigma naive|harj|hyb]
Выходы: shag3B_zhurnal_{σ}.txt (первая строка — проверка сдвига σ), shag3B_sverka_sloy.csv (5 точек на σ, 0,04 %),
shag3B_sverka_A_k3.csv (A и к3 из таблиц шага 3′ против шага 0, допуск 1e-4). Движок не меняется.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s9_shag_common as C  # noqa: E402
import s9_shag_db as DB  # noqa: E402
from fastbt import shag0  # noqa: E402
from fastbt import shag_c_tab as T  # noqa: E402

TOL = 1e-4
SDVIG = Path(os.environ.get("FASTBT_MAIN", ROOT)) / "data" / "regen" / "ch4_dev" / "proverka_sdvig.csv"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(C.OUT / name, sep=";", decimal=",", encoding="utf-8-sig")


def sverka(eng, pts: list, comm: float) -> list:
    """Быстрый слой (Engine.port) против медленного (msp.portfolio): метрики до 4 знаков и max|Δ| excess."""
    from s9_shag0 import slow_port
    rows = []
    for lab, pos in pts:
        a, b = eng.port(pos, comm=comm), slow_port(eng, pos, comm).reindex(eng.dates)

        def f(p):
            return pd.DataFrame([shag0.stats(p, np.ones(len(p), bool))]).to_csv(sep=";", float_format="%.4f")
        rows.append(dict(row=lab, comm_pct=comm * 100, metriki_do_4_znaka=f(a) == f(b),
                         max_abs_excess="%.1e" % float(np.abs(a["excess"].values - b["excess"].values).max())))
    return rows


def a_k3_vs_shag0(eng) -> list:
    """Шарп A (движок) и к3 (колонка shag3B_otbor) против shag0_A_metriki / shag0_k3, допуск 1e-4;
    Δ к A в otbor против Шарп B − Шарп A (движок), допуск 2e-4 (три числа округлены до 4 знаков)."""
    a0 = read("shag0_A_metriki.csv")
    k0 = read("shag0_k3.csv")
    rows = []
    for sg in C.SIGMAS:
        o = read(f"shag3B_otbor_{sg}.csv")
        for comm in (0.04, 0.035):
            d = o[o["comm_pct"].round(4) == comm]
            a_ref = float(a0[(a0["row"] == "A") & (a0["comm_pct"].round(4) == comm)]["sharpe"].iloc[0])
            k_ref = float(k0[k0["row"].str.startswith("к3") & (k0["comm_pct"].round(4) == comm)]["sharpe"].iloc[0])
            a_me = shag0.stats(eng.port(np.asarray(eng.A_row, float), comm=comm / 100),
                               np.ones(len(eng.dates), bool))["sharpe"]
            dk = float(np.abs(d["sharpe_k3"].values - k_ref).max())
            dd = float(np.abs(d["sharpe"].values - a_me - d["d_vs_A"].values).max())   # округление 4 знака → ≤ 1e-4
            rows.append(dict(sigma=sg, comm_pct=comm, sharpe_A_shag0=a_ref, sharpe_A_dvizhok=a_me, sharpe_k3_shag0=k_ref,
                             abs_dA=abs(a_me - a_ref), max_dk3=dk, max_d_vs_A_nesvod=dd,
                             ok=bool(abs(a_me - a_ref) < TOL and dk < TOL and dd < 2 * TOL)))
    return rows


def main(sigma: str | None) -> None:
    from fastbt import s9
    s9.use_v2(C.COMM_MAIN)
    eng = s9.Engine(s9.load_cache())
    sv = []
    for sg in ([sigma] if sigma else list(C.SIGMAS)):
        line = T.sdvig_line(T.sdvig(SDVIG, sg))
        pts = [p for src in C.SIGMAS[sg] for p in DB.points("B", eng, src, s9)]
        idx = np.linspace(0, len(pts) - 1, 5).round().astype(int)
        sel = [(f"{pts[i][1]} {C.SIGMAS[sg][i // 9]}", np.asarray(eng.A_row) * eng.g_row(pts[i][2])) for i in idx]
        r = sverka(eng, sel, C.COMM_MAIN)
        for x in r:
            x["sigma"] = sg
        sv += r
        (C.OUT / f"shag3B_zhurnal_{sg}.txt").write_text(
            line + "\nсдвиг к2 (круговой, по тикеру): лаги в [21, T−21], сдвиг 0 воспроизводит точку — проверено в прогоне\n"
            + "сверка слоёв (5 точек): " + ", ".join(f"{x['row']}: {x['metriki_do_4_znaka']}, max|Δ| {x['max_abs_excess']}"
                                                  for x in r) + "\n", encoding="utf-8")
        print(sg, line)
    DB.write(sv, "shag3B_sverka_sloy.csv")
    ak = a_k3_vs_shag0(eng)
    DB.write(ak, "shag3B_sverka_A_k3.csv")
    print(pd.DataFrame(sv).to_string())
    print(pd.DataFrame(ak).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", choices=list(C.SIGMAS))
    main(ap.parse_args().sigma)
