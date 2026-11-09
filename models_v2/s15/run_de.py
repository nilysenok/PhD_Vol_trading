"""Запуск листов Deep Structural Extension (SBER, h = 1). Стадия без `--check` ПИШЕТ выходы, `--check` только читает.
python3 -W ignore -m models_v2.s15.run_de --stage 0|reg|E1|E2|E3|E4|E5|FIN [--check]
`reg` — запись реестра 5 мест (`ALL_TRIALS.csv`) ДО любого прогноза; E1 = X1 … E5 = X5: незарегистрированной конфигурации — отказ, X5 без условия — отказ (`de_reg.guard`). E2…E5, FIN — каркас отказов (счёт в своих листах)."""
from __future__ import annotations

import argparse
import sys


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
    if a.stage == "FIN":
        raise SystemExit("FIN: каркас — итог собирается после E5")
    cfg = R.STAGE_CFG[a.stage]
    R.guard(cfg)
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
