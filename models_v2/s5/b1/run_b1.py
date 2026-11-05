"""B1 (Ж-204, волна B): python3 models_v2/s5/b1/run_b1.py --stage controls|all|audit|readme|check-readme. Процессов <= 4."""
import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics  # noqa: E402
from models_v2.s4 import runner, select  # noqa: E402
from models_v2.s5 import common  # noqa: E402
from models_v2.s5.b1 import configs, designs, jobs, report, tables, world  # noqa: E402

DATES = ("2018-06-01", "2022-06-01", "2025-06-02")
HERE = Path(__file__).resolve().parent


def stage_controls() -> int:
    w = world.load()
    specs = designs.build_specs(w.F, w.DZ, w.lb)
    jobs.init(w.F, specs, w.i2, w.eps)
    c = common.controls()
    qb, qh = float(metrics.qlike(c.rv, c.s2_b2).mean()), float(metrics.qlike(c.rv, c.s2_harj).mean())
    print("QLIKE Б-2 %.12f HAR-J(б) %.12f n=%d" % (qb, qh, len(c)))
    pc = pd.read_parquet(common.OUT / "fix_forecasts_c.parquet")
    ok = round(qb, 4) == 0.2205 and round(qh, 4) == 0.2337 and len(c) == common.N_GRID
    for ds in DATES:
        d = int(w.grid[np.searchsorted(w.F.dates[w.grid], np.datetime64(ds))])
        for nm, model in (("ctrl_b2", "fx_b2_g132_b"), ("ctrl_b1", "fx_b1_g132_b")):
            _, r = jobs.block_job((nm, d))
            s2 = float(designs.sigma2(specs[nm], [d], [r["f"]], [r["s2"]], w.eps)[0][0])
            ref = float(pc[(pc.model == model) & (pc.d == w.F.dates[d])]["sigma2"].iloc[0])
            print(ds, nm, s2, ref, "побитно" if s2 == ref else "rel=%.2e" % abs(s2 / ref - 1))
            ok &= s2 == ref
    for Y, y in ((2019, 2017), (2023, 2020)):
        _, q = jobs.fold_job(("ctrl_b2", Y, y))
        ref = runner.fold_score(w.F, w.DZ["m_mkx"], w.i2[Y], select.cut_row(w.F, Y), y, False)
        print("фолд", Y, y, q[0], ref, q[0] == ref)
        ok &= q[0] == ref
    print("controls:", "OK" if ok else "ОШИБКА")
    return 0 if ok else 1


def stage_all() -> int:
    w = world.load()
    specs = designs.build_specs(w.F, w.DZ, w.lb)
    ctrl, rows, folds, info = tables.all_rows(w, specs)
    df = tables.table(ctrl, rows, folds)
    tables.write_tables(df)
    n = tables.append_registry(df)
    print(info, "новых trials в реестре:", n, "по направлениям:", configs.counts())
    print(df[["row_id", "q_int_kind", "q_int", "q_int_b2", "qlike_all", "qlike_no_240222", "d_harj_no_240222", "perspective"]].to_string())
    return 0


def stage_audit() -> int:
    bad = []
    for f in sorted(HERE.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        if len(src.splitlines()) > 200:
            bad.append("%s: %d строк" % (f.name, len(src.splitlines())))
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno - n.lineno + 1 > 40:
                bad.append("%s:%s: %d строк" % (f.name, n.name, n.end_lineno - n.lineno + 1))
    print("нарушений: %d" % len(bad), *bad, sep="\n")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("controls", "all", "audit", "readme", "check-readme"))
    s = ap.parse_args().stage
    if s == "readme":
        return report.write_auto()
    if s == "check-readme":
        return report.check_auto()
    return {"controls": stage_controls, "all": stage_all, "audit": stage_audit}[s]()


if __name__ == "__main__":
    sys.exit(main())
