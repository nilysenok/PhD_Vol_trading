"""Таблица «было/стало» после замены ряда ставки (Ъ 27.10, 12:00) → docs/tables/strat_dev_v3/stavka_bylo_stalo.csv.

    python3 scripts/analysis/stavka_bylo_stalo.py
«Было» — архив docs/otchety/arhiv_do_audita/posle_stavki/ (код тот же, ставка — parquet ЦБ),
«стало» — текущие docs/tables/strat_dev_v3 и strat_dev_v2. Шаг 0 — комиссии 0,035 и 0,04 %;
выбор точек C/D/B (#243, К) — 0,04 % (отбор при 0,04, Ъ 27.10, 12:05) и 0,05 % (k0050, справочно).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARH = ROOT / "docs" / "otchety" / "arhiv_do_audita" / "posle_stavki"
V3, V2 = ROOT / "docs" / "tables" / "strat_dev_v3", ROOT / "docs" / "tables" / "strat_dev_v2"


K3 = "к3: A без S5/S6"


def rd(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig")


def cell(df: pd.DataFrame, row: str, comm: float, col: str) -> float:
    m = df[(df["row"] == row) & ((df["comm_pct"] - comm).abs() < 1e-9)]
    return float(m[col].iloc[0])


def shag0_rows() -> list:
    out = []
    k_old, k_new = rd(ARH / "shag0_bylo" / "shag0_k3.csv"), rd(V3 / "shag0_k3.csv")
    l_old, l_new = rd(ARH / "shag0_bylo" / "shag0_signaly_LOO.csv"), rd(V3 / "shag0_signaly_LOO.csv")
    for c in (0.035, 0.04):
        for lab, o, n, row, col in [("Шарп A", k_old, k_new, "A", "sharpe"), ("Шарп к3 (A без S5/S6)", k_old, k_new, K3, "sharpe"),
                                    ("Δ к3 − A", k_old, k_new, K3, "d_vs_A")]:
            out.append(dict(blok="шаг 0", pokazatel=lab, komissiya_pct=c, bylo=cell(o, row, c, col), stalo=cell(n, row, c, col)))
        for s in range(1, 7):
            row = f"A без S{s}"
            out.append(dict(blok="шаг 0", pokazatel=f"вклад S{s} (Шарп A − Шарп A без S{s})", komissiya_pct=c,
                            bylo=cell(l_old, row, c, "vklad_sharpe"), stalo=cell(l_new, row, c, "vklad_sharpe")))
    return out


def vybor_rows() -> list:
    out = []
    for suf, c in (("", 0.04), ("_k0050", 0.05)):
        f = f"s9_vybor_C8_v2{suf}.csv"
        o, n = rd(ARH / "v2_bylo" / f), rd(V2 / f)
        for ap in ("A", "D", "C", "B"):
            ro, rn = o[o["approach"] == ap].iloc[0], n[n["approach"] == ap].iloc[0]
            same = (ro["design"], ro["sigma"]) == (rn["design"], rn["sigma"])
            out.append(dict(blok="отбор #243/К (v2)", pokazatel=f"{ap}: {rn['design']} на {rn['sigma']}", komissiya_pct=c,
                            bylo=float(ro["sharpe"]), stalo=float(rn["sharpe"]), tochka_bylo=f"{ro['design']} на {ro['sigma']}",
                            vybor_smenilsya="нет" if same else "да"))
    return out


def main() -> None:
    df = pd.DataFrame(shag0_rows() + vybor_rows())
    df["delta"] = df["stalo"] - df["bylo"]
    p = V3 / "stavka_bylo_stalo.csv"
    df.to_csv(p, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    print(df.to_string())


if __name__ == "__main__":
    main()
