"""Л5: техническая репетиция (протокол HARJ_HYBRID_SBER_FWD, раздел 7; Ж-248 п. 6) — 38 дней d = 2025-12-30 … 2026-02-26, только архив (cut <= 2026-02-27).

Каждый день: бегун (модули Л1–Л4) запускается на данных, обрезанных по моменту выпуска (бары всех 18 инструментов с begin <= d 18:40), и сверяется с архивом по узлам:
календарь, доступность баров, 17 рыночных признаков, K1, K2, K3, И2, M, z, e_d, s_d, cfg8 (+λ по seeds), X1 (+π₀, π₁, η, α), ZB (+λ, b, a), ZB0, даты переоценок.
Допуск 0 (побитно); объявленный 1e-12 отн. — только узлы torch / xgboost / lightgbm (колонка `tol`). Первый и последний день — режим `skvoznaya` (K2/K3 и cfg8 строк < d с нуля).
Команды: `--run through|obrezka` (части в data/fwd/models/rehearsal_part_*.csv, продолжение с места остановки), `--merge` (REHEARSAL.csv), `--check` (только читает).
"""
from __future__ import annotations

import torch  # noqa: F401  (первым: общий OpenMP с lightgbm/xgboost, класс № 152)

import sys
from pathlib import Path

import pandas as pd

from models_v2.s17 import fwd_reh_cut as RC, fwd_reh_ref as RR
from models_v2.s17.fwd_src import ROOT, Src

OUT = ROOT / "docs/otchety/harj_fwd/REHEARSAL.csv"
PARTS = ROOT / "data/fwd/models"
FIRST = pd.Timestamp("2025-12-30")
NODES = ["cal_t_plus", "cal_days", "avail", "mkt17", "dz0_full", "K1", "prefix_K1", "K2", "K3", "I2", "M", "z", "e_d", "s_d", "prefix_es", "refit_cfg8", "cfg8", "cfg8_lam",
         "X1", "X1_state", "refit_x1_alpha", "prefix_X1", "ZB", "ZB_state", "refit_zb", "prefix_ZB", "ZB0", "refit_zb0", "prefix_ZB0", "info"]
THROUGH_ONLY = ["prefix_K2", "prefix_K3", "prefix_cfg8"]
HEAD = ("# DEVELOPMENT/REHEARSAL Л5 (Ж-248 п. 6); 38 дней d × узлы; обрезка по данным <= d 18:40 против архива; допуск 0, для torch/xgboost/lightgbm 1e-12 отн. (колонка tol); "
        "mode skvoznaya — K2/K3 и cfg8 строк < d с нуля; архивные строки, цен новых файлов нет\n")


def days() -> list:
    h = pd.read_parquet(RR.PAN / "har_SBER.parquet", columns=["d"])
    return [pd.Timestamp(d) for d in h["d"] if d >= FIRST]


def part_path(kind: str) -> Path:
    return PARTS / f"rehearsal_part_{kind}.csv"


def run(kind: str) -> int:
    ds = days()
    todo = [ds[0], ds[-1]] if kind == "through" else ds[1:-1]
    path = part_path(kind)
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {str(r["d"]) for r in rows}
    src = Src()
    R = RR.Ref(src)
    for c in todo:
        if str(c.date()) in done:
            continue
        rows += RC.run_cut(c, src, R, through=(kind == "through"))
        tmp = path.with_name("tmp_" + path.name)
        pd.DataFrame(rows, columns=RR.COLS).to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(path)
        bad = [r["node"] for r in rows if str(r["d"]) == str(c.date()) and not r["ok"]]
        print(c.date(), "расхождения:", bad or "нет", flush=True)
    return 0


def merge() -> int:
    df = pd.concat([pd.read_csv(part_path(k)) for k in ("through", "obrezka")], ignore_index=True)
    df = df.sort_values(["d", "node"], kind="stable").reset_index(drop=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HEAD)
        df.to_csv(f, index=False, lineterminator="\n")
    return check()


def problems(df: pd.DataFrame) -> list:
    """Список расхождений и нарушений состава таблицы; пустой — «0 расхождений». Ничего не пишет."""
    out, want = [], [str(d.date()) for d in days()]
    if sorted(df["d"].astype(str).unique()) != sorted(want) or len(want) != 38:
        out.append(f"дни: {len(df['d'].unique())} ≠ 38 ожидаемых")
    for d, g in df.groupby("d"):
        thr = g["mode"].iloc[0] == "skvoznaya"
        miss = [n for n in NODES + (THROUGH_ONLY if thr else []) if n not in set(g["node"])]
        out += [f"{d}: нет узла {n}" for n in miss]
        extra = [n for n in g["node"] if n in THROUGH_ONLY and not thr]
        out += [f"{d}: лишний узел {n}" for n in extra]
    through = sorted(df[df["mode"] == "skvoznaya"]["d"].astype(str).unique())
    if through != [want[0], want[-1]]:
        out.append(f"сквозные обрезки {through} ≠ первая и последняя")
    bad = df[~df["ok"].astype(bool)]
    return out + [f"{r.d} {r.node}: got={r.got[:80]} ref={r.ref[:80]} relmax={r.relmax} tol={r.tol}" for r in bad.itertuples()]


def check() -> int:
    df = pd.read_csv(OUT, comment="#")
    p = problems(df)
    n_bit = int(df["bitwise"].astype(bool).sum())
    if p:
        print(f"РАСХОЖДЕНИЙ: {len(p)}", *p, sep="\n")
        return 1
    print(f"0 расхождений: {df['d'].nunique()} дней × {df['node'].nunique()} узлов, строк {len(df)}; побитно {n_bit} из {len(df)}; допуск 1e-12 — только torch/xgboost/lightgbm")
    return 0


def main(argv: list) -> int:
    if "--run" in argv:
        return run(argv[argv.index("--run") + 1])
    if "--merge" in argv:
        return merge()
    if "--check" in argv:
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
