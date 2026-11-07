"""P3-17, G17 (Ж-220, У-22): стадия `s17`. Запуск: `run_hp --stage s17 --part sel|loto|stats|e11 [--module xgb|lgbm] [--check]`.
Порядок: `sel` (выбор и объявление N, до любого p) → `loto --module xgb|lgbm` (фон) → `e11` (перенос) и `stats` (17 + LOTO; после LOTO). Результаты на 28 (`statistics/`, `hyb_*.parquet`, `loto*.parquet`) не трогаются."""
from __future__ import annotations

import json
import time

from models_v2.s10 import hp_paths as P, hp_s17_sel as SEL, hp_stats_sel as S


def _log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


def declared() -> dict:
    return json.loads((SEL.D17 / "SELECTION.json").read_text())


def stage(part: str, module: str | None, check: bool, nthread: int = 4) -> int:
    if check:
        from models_v2.s10 import hp_s17_check as CK
        bad = CK.check(part)
        print("\n".join(bad))
        print("ИТОГ:", "OK" if not bad else "FAIL")
        return 1 if bad else 0
    if part == "sel":
        SEL.run(_log)
    elif part in ("loto", "loto-rev") and module:
        from models_v2.s10 import hp_s17_loto as LO
        LO.run_module(module, [c.split("_", 1)[1] for c in declared()["loto"] if c.startswith(module + "_")], nthread, _log, reverse=part == "loto-rev")
    elif part in ("stats", "e11"):
        from models_v2.s10 import hp_s17_stats as ST
        ST.run_part(part, _log)
    else:
        print("часть %s не задана (sel|loto --module|stats|e11)" % part)
        return 1
    return 0
