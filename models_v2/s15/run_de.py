"""Запуск листов Deep Structural Extension (SBER, h = 1). Стадия без `--check` ПИШЕТ выходы, `--check` только читает.
python3 -W ignore -m models_v2.s15.run_de --stage 0|reg|E1|E2|E3|E4|E5|FIN [--check]
`reg` — запись реестра 5 мест (`ALL_TRIALS.csv`) ДО любого прогноза; E1 = X1 … E5 = X5: незарегистрированной конфигурации — отказ, X5 без условия — отказ (`de_reg.guard`). E5 — условие X5 (механически по таблицам E2–E4; X5 при ложном условии не считается), FIN — сводные таблицы и итоговый документ."""
from __future__ import annotations

import argparse
import importlib
import sys

STAGES = {"E2": "de_e2_stage", "E3": "de_e3_stage", "E4": "de_e4_stage"}      # подключённые стадии листов E2+ (модуль с compute/write/check); тест подменяет таблицу


def _e5_fin(a) -> int:
    """E5 — условие X5 по таблицам E2–E4 (ничего не считается; X5 бегун отказывает, `de_reg.guard`); FIN — сводные таблицы и документ. Оба: без `--check` пишут, с `--check` только читают."""
    from models_v2.s15 import de_e5_stage, de_fin_stage
    S = de_e5_stage if a.stage == "E5" else de_fin_stage
    if a.check:
        bad = S.check()
        print("%s --check: %s" % (a.stage, "0 расхождений" if not bad else "расхождений %d: %s" % (len(bad), "; ".join(bad))))
        return 1 if bad else 0
    r = S.write()
    if a.stage == "E5":
        try:
            from models_v2.s15 import de_reg
            de_reg.guard("X5")
        except SystemExit as e:
            print("E5: условие X5 ложно (лучший из X2–X4 — %s, 6.2 против A0g — «%s»); X5 не считается; бегун отказывает: %s" % (r["best"], r["rule62"], e))
        return 0
    print("FIN: таблицы FIN_*.csv, документ и CHECKPOINT[FIN] записаны")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["0", "reg", "E1", "E2", "E3", "E4", "E5", "FIN"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "0":
        from models_v2.s15 import de0_stage as S
        if a.check:
            bad = S.check()
            print("DE0 --check: %s" % ("0 расхождений" if not bad else "расхождений %d: %s" % (len(bad), "; ".join(bad))))
            return 1 if bad else 0
        S.write()
        print("DE0: таблицы, отчёт и CHECKPOINT[0] записаны")
        return 0
    from models_v2.s15 import de_reg as R
    if a.stage == "reg":
        if a.check:
            raise SystemExit("reg: --check нет (реестр только дополняется)")
        print("реестр: добавлено строк %d" % R.register())
        return 0
    if a.stage in ("E5", "FIN"):
        return _e5_fin(a)
    cfg = R.STAGE_CFG[a.stage]
    R.guard(cfg)
    if a.stage in STAGES:
        S = importlib.import_module("models_v2.s15." + STAGES[a.stage])
        if a.check:
            bad = S.check()
            print("%s --check: %s" % (a.stage, "0 расхождений" if not bad else "расхождений %d: %s" % (len(bad), "; ".join(bad))))
            return 1 if bad else 0
        S.write()
        print("%s: %s, таблицы, отчёт и CHECKPOINT[%s] записаны" % (a.stage, cfg, a.stage))
        return 0
    if a.stage != "E1":
        raise SystemExit("%s (%s): счёт в своём листе, здесь не реализован" % (a.stage, cfg))
    from models_v2.s15 import de_e1_stage as S1
    if a.check:
        bad = S1.check()
        print("E1 --check: %s" % ("0 расхождений" if not bad else "расхождений %d: %s" % (len(bad), "; ".join(bad))))
        return 1 if bad else 0
    S1.write()
    print("E1: X1, таблицы, отчёт и CHECKPOINT[E1] записаны")
    return 0


if __name__ == "__main__":
    sys.exit(main())
