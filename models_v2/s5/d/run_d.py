"""Волна D (Ж-204): python3 models_v2/s5/d/run_d.py --stage control|comp|d1|robust|d2|report|all   (пул <= 6; seed 42; расчёты только здесь)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s5.c import ctx_c  # noqa: E402
from models_v2.s5.d import d2_d, engine_d, robust_d, stages_d  # noqa: E402


def _report(C, p):
    from models_v2.s5.d import report_d   # отчётные модули тяжёлые (matplotlib) и не нужны счётным стадиям
    report_d.run(C)


POOLED = {"comp": stages_d.stage_comp, "d1": stages_d.stage_d1}
PLAIN = {"control": stages_d.stage_control, "robust": lambda C, p: robust_d.run(C), "d2": lambda C, p: d2_d.run(C), "report": _report}
ORDER = ("control", "comp", "d1", "robust", "d2", "report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=ORDER + ("all",), required=True)
    a = ap.parse_args()
    C = ctx_c.load()
    for s in (ORDER if a.stage == "all" else (a.stage,)):
        print("== stage", s, flush=True)
        if s in POOLED:
            with engine_d.pool(C["w"]) as p:
                POOLED[s](C, p)
        else:
            PLAIN[s](C, None)


if __name__ == "__main__":
    main()
