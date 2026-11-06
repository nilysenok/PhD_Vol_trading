"""Волна A Ж-206. Запуск: python3 models_v2/s6/run_a.py --stage <inventory|origin|baseline|lgb_b2|diag|lgb_extra|check|audit|check-readme>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-sha", action="store_true")
    a = ap.parse_args()
    from models_v2.s6 import paths6
    if a.stage == "lgb_b2":
        from models_v2.s6 import lgb_b2
        lgb_b2.run_all(paths6.world(), control_only=a.control)
        return 0
    from models_v2.s6 import stages
    return stages.dispatch(a)


if __name__ == "__main__":
    raise SystemExit(main())
