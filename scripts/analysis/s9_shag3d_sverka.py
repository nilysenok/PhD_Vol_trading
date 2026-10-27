"""Шаг 3′ D: сверка быстрого слоя (Engine.port) с медленным (msp.portfolio) на 5 точках сетки D + A и к3 из шага 0.

    FASTBT_MAIN=<своя копия> python3 scripts/analysis/s9_shag3d_sverka.py
→ docs/tables/strat_dev_v3/shag3D_sverka_sloy.csv. Строки метрик (CSV, 4 знака) обязаны совпасть; A и к3 при 0,04 и 0,035
сверяются с shag0_A_metriki.csv / shag0_k3.csv до 1e-4. Без аргументов самопроверки нет — скрипт и есть проверка.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import s9_shag0 as S0  # noqa: E402
import s9_shag_common as C  # noqa: E402
import s9_shag_db as DB  # noqa: E402
from fastbt import fast_kontrol as fk  # noqa: E402
from fastbt import s9, shag0  # noqa: E402

PICK = [("D6lag1", "W252_p70"), ("D5", "x30"), ("D6lag1", "W252_p80")]   # + первая точка масштаба и первая D1-подобная


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(C.OUT / name, sep=";", decimal=",", encoding="utf-8-sig")


def pick_points(eng) -> list:
    pts = DB.points("D", eng, "hyb_h5", s9)
    by = {(f, lab): (f, lab, G) for f, lab, G, _, _, _ in pts}
    out = [by[k] for k in PICK if k in by]
    for f, lab, G, scale, _, _ in pts:
        if (scale or f == "D1") and len(out) < 5 and all(lab != o[1] or f != o[0] for o in out):
            out.append((f, lab, G))
    return out[:5]


def check_vs_shag0(eng, comm: float) -> list:
    """A и к3 (позиции движка) против shag0_*: Шарп до 1e-4."""
    A = np.asarray(eng.A_row, float)
    k3 = fk.k3_positions(A, eng.r["pair"], eng.pairs)
    rows = []
    for lab, pos, f in (("A", A, "shag0_A_metriki.csv"), ("к3", k3, "shag0_k3.csv")):
        ref = rd(f)
        ref = ref[(ref["row"].str.startswith(lab)) & (ref["comm_pct"].round(4) == round(comm * 100, 4))]
        mine = shag0.stats(eng.port(pos, comm=comm), np.ones(len(eng.port(pos, comm=comm)), bool))["sharpe"]
        d = abs(mine - float(ref["sharpe"].iloc[0]))
        rows.append(dict(row=f"{lab} против shag0", comm_pct=comm * 100,
                         metriki_do_4_znaka=d < 1e-4, max_abs_excess="%.1e" % d))
        assert d < 1e-4, (lab, comm, d)
    return rows


def main() -> None:
    s9.use_v2(C.COMM_MAIN)
    eng = s9.Engine(s9.load_cache())
    rows = []
    for comm in (C.COMM_MAIN, C.COMM_ALT):
        rows += check_vs_shag0(eng, comm)
        for f, lab, G in pick_points(eng):
            pos = eng.A_row * eng.g_row(G)
            a, b = eng.port(pos, comm=comm), S0.slow_port(eng, pos, comm).reindex(eng.port(pos, comm=comm).index)
            sa = pd.DataFrame([shag0.stats(a, np.ones(len(a), bool))]).to_csv(sep=";", float_format="%.4f", index=False)
            sb = pd.DataFrame([shag0.stats(b, np.ones(len(b), bool))]).to_csv(sep=";", float_format="%.4f", index=False)
            rows.append(dict(row=f"{f}:{lab} (hyb_h5)", comm_pct=comm * 100,
                             metriki_do_4_znaka=sa == sb,
                             max_abs_excess="%.1e" % float(np.abs(a["excess"].values - b["excess"].values).max())))
    p = DB.write(rows, "shag3D_sverka_sloy.csv")
    print(pd.DataFrame(rows).to_string())
    assert all(r["metriki_do_4_znaka"] for r in rows), "быстрый и медленный слои расходятся"
    print("OK", p)


if __name__ == "__main__":
    main()
