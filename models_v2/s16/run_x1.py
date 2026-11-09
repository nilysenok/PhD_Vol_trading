"""X1-пилот, бегун: `python3 -W ignore -m models_v2.s16.run_x1 --stage P0 [--check]`. Лист P0 (Ж-241, Ж-242): таблицы P0_SCALE / P0_INIT / P0_DYN / P0_MIX, отчёт P0_REPORT.md (генератор), ключ `P0` в CHECKPOINT.json.
Прогнозы не пишутся, конфигураций нет; повтор cfg6 V7 на трёх датах переоценки — только диагностика (без записи прогнозов). `--check` ничего не пишет: пересчитывает дешёвые таблицы и сверяет с файлами, пересобирает отчёт,
сверяет хэши выходов V6, V7, DE и собственных файлов до и после."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch раньше s14)
from models_v2.s16 import x1_p0_dyn as DY, x1_p0_doc as DOC, x1_p0_init as IN, x1_p0_mix as MX, x1_p0_scale as SC, x1_p0_syn as SY

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_x1"
WATCH = [ROOT / p for p in ("data/regen/harj_hybrid_v6", "data/regen/harj_hybrid_v7", "data/regen/harj_deep", "docs/otchety/harj_hybrid_v6", "docs/otchety/harj_hybrid_v7", "docs/otchety/harj_deep",
                            "state/harj_hybrid_v6.md", "state/harj_hybrid_v7.md", "state/harj_deep.md")]
CSVS = ("P0_SCALE.csv", "P0_INIT.csv", "P0_DYN.csv", "P0_MIX.csv")
KEYS = ("P0_AUDIT.md", "P0_REPORT.md") + CSVS


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            if f.is_file():
                h.update(str(f).encode() + sha(f).encode())
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(list(Path(__file__).parent.glob("x1_p0_*.py")) + [Path(__file__), Path(__file__).parent / "__init__.py"]):      # только файлы листа P0 (рядом лежат чужие x1_t0_*)
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / name, comment="#")


def write_csv(df: pd.DataFrame, name: str) -> None:
    (REP / name).write_text("# %s\n" % DOC.STATUS + df.to_csv(index=False), encoding="utf-8")


def same(a: pd.DataFrame, b: pd.DataFrame, cols=None) -> bool:
    cols = [c for c in (cols or a.columns) if c in b.columns]
    if len(a) != len(b):
        return False
    for c in cols:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        ok = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True) if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) else bool((x.astype(str) == y.astype(str)).all())
        if not ok:
            return False
    return True


def init_frame(trained: bool) -> pd.DataFrame:
    rows = IN.init_table(train_months=IN.FIT_MONTHS[1:] if trained else ()) + SY.rows()
    b = DY.synthetic_batch()
    for s in IN.FT.SEEDS:
        for g, zero in (("router_zero", True), ("router_random", False)):
            rows.append(dict(stage="symmetry", model="синтетика, 100 шагов Adam", group=g, seed=s, extra=DY.symmetry_dist(b, zero, s), note="расстояние параметров двух экспертов, стартовавших с одинаковых нулей"))
    return pd.DataFrame(rows)


def pi6_row() -> dict:
    return IN.pi_cfg6()


def build_report(scale, init, mix, dyn) -> str:
    return DOC.build(scale, init, mix, pi6_row(), (REP / "P0_AUDIT.md").read_text(encoding="utf-8"), dyn)


def write_stage() -> int:
    if shutil.disk_usage(ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    scale, mix, dyn = SC.scale_table(), MX.mix_table(), pd.DataFrame(DY.dyn_table())
    init = init_frame(True)
    for df, n in ((scale, "P0_SCALE.csv"), (init, "P0_INIT.csv"), (dyn, "P0_DYN.csv"), (mix, "P0_MIX.csv")):
        write_csv(df, n)
    (REP / "P0_REPORT.md").write_text(build_report(read_csv("P0_SCALE.csv"), read_csv("P0_INIT.csv"), read_csv("P0_MIX.csv"), read_csv("P0_DYN.csv")), encoding="utf-8")
    cp_path = REP / "CHECKPOINT.json"
    cp = json.loads(cp_path.read_text(encoding="utf-8")) if cp_path.exists() else {}
    cp["P0"] = dict(sha256={k: sha(REP / k) for k in KEYS}, code=code_hash(), configurations=0, forecasts_written=0, written_at=datetime.now().strftime("%Y-%m-%d %H:%M"), status=DOC.STATUS)
    cp_path.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print("P0: таблицы и отчёт записаны, CHECKPOINT ключ P0")
    return 0


def check_stage() -> int:
    before = (digest(WATCH), digest([REP]))
    bad, cp = [], json.loads((REP / "CHECKPOINT.json").read_text(encoding="utf-8")).get("P0")
    if cp is None:
        bad.append("CHECKPOINT: нет ключа P0")
    else:
        bad += ["sha256 %s ≠ CHECKPOINT" % k for k, v in cp["sha256"].items() if sha(REP / k) != v]
        bad += ["код s16 ≠ CHECKPOINT"] if cp["code"] != code_hash() else []
    scale, init, mix = read_csv("P0_SCALE.csv"), read_csv("P0_INIT.csv"), read_csv("P0_MIX.csv")
    if not same(SC.scale_table(), scale, [c for c in scale.columns if c != "note"]):
        bad.append("P0_SCALE.csv ≠ пересчёту")
    if not same(MX.mix_table(), mix, [c for c in mix.columns if c != "conclusion"]):
        bad.append("P0_MIX.csv ≠ пересчёту")
    keep = ["init", "synthetic_init", "symmetry"]
    now, old = init_frame(False), init[init["stage"].isin(keep)]
    cols = ["stage", "model", "group", "seed", "grad_data", "grad_pen", "grad_total", "extra"]
    if not same(now[now["stage"].isin(keep)][cols], old[cols]):
        bad.append("P0_INIT.csv (init, синтетика) ≠ пересчёту")
    if (REP / "P0_REPORT.md").read_text(encoding="utf-8") != build_report(scale, init, mix, read_csv("P0_DYN.csv")):
        bad.append("P0_REPORT.md ≠ генератору из таблиц")
    after = (digest(WATCH), digest([REP]))
    if before != after:
        bad.append("--check изменил выходы (хэш до ≠ после)")
    print("\n".join(bad) if bad else "OK: P0_SCALE/INIT/MIX пересчитаны и равны файлам, отчёт = генератор, CHECKPOINT P0 сходится; выходы V6/V7/DE и P0 не изменены (хэш V6/V7/DE %s)" % before[0][:16])
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["P0"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    return check_stage() if a.check else write_stage()


if __name__ == "__main__":
    raise SystemExit(main())
