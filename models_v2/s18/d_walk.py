"""S-D-IMPL-1/2 (Ж-256; `S_D_SPEC.md`, разд. 2–3): минимальные абляции S-RCK — S-RCKm (R при m := 1) и S-RCKf (потеря обучения, c и m на уровне C; валидация блоков — уровень F, как в R).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD; диагностика вклада элементов R, не кандидаты. Выходы — `data/regen/harj_struct/D/{rckm,rckf}.parquet` (R/ не трогается).
Публичное API: `VARS_D` = {"rckm": ("CK", "Rm1", "blocks"), "rckf": ("CK", "R", "blocks")}; `LEVEL_D` = {"rckm": "F", "rckf": "C"} (уровень потери обучения); `IDS_D`; `OUTS_D`; `D_DIR`;
`run(var, write=True, out=D_DIR) -> DataFrame`: `d_reg.require` + `d_reg.check()` пуст → `make_ctx("CK", LEVEL_D[var])` → `r_walk.walk_variant(..., spec=VARS_D[var])` → `r_walk.assemble`; существующий иной выход — отказ; запись через `.tmp` и `replace`.
Запуск: `python3 -W ignore -m models_v2.s18.d_walk --run rckm|rckf`. Проверки — `d_walk_chk.py`."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from models_v2.s18 import d_reg as DR
from models_v2.s18 import r_walk as W

VARS_D = {"rckm": ("CK", "Rm1", "blocks"), "rckf": ("CK", "R", "blocks")}
LEVEL_D = {"rckm": "F", "rckf": "C"}
IDS_D = {"rckm": "S-RCKm", "rckf": "S-RCKf"}
OUTS_D = {v: v + ".parquet" for v in IDS_D}
D_DIR = DR.D_DIR


def run(var: str, write: bool = True, out: Path = D_DIR) -> pd.DataFrame:
    DR.require(IDS_D[var])
    bad = DR.check()
    if bad:
        raise SystemExit("реестр S-D не готов: %s" % "; ".join(bad))
    ctx = W.make_ctx(VARS_D[var][0], LEVEL_D[var])
    df = W.assemble(var, ctx, W.walk_variant(var, ctx, work=out / "_part" / ("%s_%s" % (var, W.code_key())), spec=VARS_D[var]))
    p = out / OUTS_D[var]
    if write:
        if p.exists() and not W._same(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / (OUTS_D[var] + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
    return df


if __name__ == "__main__":
    if "--run" not in sys.argv:
        raise SystemExit("укажите --run rckm|rckf")
    v, t0 = sys.argv[sys.argv.index("--run") + 1], time.time()
    print("%s: записано %d строк, %.0f с" % (v, len(run(v)), time.time() - t0))
