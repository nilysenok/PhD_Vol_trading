"""HAR-J Hybrid V6, лист 0 (Ж-227), только SBER, h = 1. Запуск из корня репозитория, процесс один:
  python3 -W ignore -m models_v2.s13.run_v6 --stage 0          # пересчитать и записать выходы листа
  python3 -W ignore -m models_v2.s13.run_v6 --stage 0 --check  # ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, суммарный хэш выходов до и после
  python3 -W ignore -m models_v2.s13.run_v6 --digest-v5        # суммарный хэш выходов V5 (G9: до и после проверок V5)
  python3 -W ignore -m models_v2.s13.run_v6 --stage 1a [--check]   # лист 1a: A0 = S + LIN (записать / только проверить)
  python3 -W ignore -m models_v2.s13.run_v6 --stage 1b [--check]   # лист 1b: A1, A2, A3 по правилу 2.2, DX-A (записать / только проверить)
  python3 -W ignore -m models_v2.s13.run_v6 --stage 2 [--check]    # лист 2: C1, C2, C3, C4 по правилу 2.2 (записать / только проверить)
Стадия 0 моделей не обучает, конфигурации не расходует; стадия 1a расходует одну (A0); стадия 1b — A1, A2 и (по правилу) A3."""
from __future__ import annotations

import argparse
import sys

from models_v2.s13 import v6_0_io as IO, v6_0_stage as ST, v6_1_ev as EV1, v6_1_stage as S1, v6_1b_dxa as DXA, v6_1b_ev as EV1B, v6_1b_io as IO1B, v6_1b_stage as S1B, v6_2_ev as EV2, v6_2_io as IO2, v6_2_stage as S2


def evidence(r: dict) -> list:
    """Строки для EVIDENCE G1–G8 (числа из пересчёта)."""
    c, p, b, pa, rs, cr, cp = r["cres"], r["prior"], r["bres"], r["pres"], r["rres"], r["cr"], r["checkpoint"]
    cls = p["class"].value_counts().to_dict()
    best = b["best"]
    th = cp["thresholds"]
    pick = lambda s, w: best[(best["set"] == s) & (best["window"] == w)].iloc[0]
    out = ["G1: SHA ок %s; ряды ок %s; K8 побитно %s; сдвиг %s (мутация +1 не проходит); 0,2081 воспроизведён %s; QLIKE вся история %s; дат %d" % (c["files_ok"], c["series_ok"], c["k8_bitwise"], c["shift_ok"], c["ref_ok"], {k: round(v, 4) for k, v in c["qlike"].items()}, c["n_dates"]),
           "G2: рядов %d; классы %s; расхождений с отчётом > 1e-4: %d" % (len(p[p["class"].isin(["PIT-сопоставим", "потолок"])]), cls, int((p["diff_reported"] > 1e-4).sum()))]
    for w in ("all", "2017_2021"):
        for s in ("PIT-сопоставимые", "все ряды (потолки помечены)"):
            x = pick(s, w)
            out.append("G3: окно %s, %s: %s (%s) QLIKE %.6f на %d датах" % (w, s, x["series"], x["class_"], x["qlike"], x["n"]))
    out.append("G3: общие даты: %s" % b["common"][["set", "n_series", "n_dates", "first_d", "last_d"]].to_dict("records"))
    out.append("G5: max отн. разн. Σr² к RV канона %.2e, к цели K1 %.2e" % (pa["max_daily"], pa["max_target"]))
    out.append("G6: %s" % r["rres"]["warm"][["item", "value"]].to_dict("records"))
    out.append("G7: формула crit_wf ≤ %.2e; шум K8 − лучший из K2/K3: %s" % (cr["formula_max_diff"], {k: (round(v, 6) if isinstance(v, float) else v) for k, v in cr["noise"].items()}))
    out.append("G8: пороги %s; N = %d, N_wide = %d; go-рядов %d; чувствительность (все причинные Ж-210…214) %s" % ({k: round(v["threshold"], 6) for k, v in th.items() if isinstance(v, dict) and "threshold" in v}, cp["N"], cp["N_wide"], len(cp["go_series"]),
                                                                                                                {k: round(v["threshold"], 6) for k, v in r["alt"].items()}))
    return out


def main_1a(check: bool, before: str) -> int:
    """Стадия 1a (лист 1a): A0 = S + LIN. Без --check — пересчёт и запись; с --check — только чтение, хэш выходов до и после."""
    r = S1.compute()
    if not check:
        S1.write(r)
        print("стадия 1a записана; хэш выходов %s; счёт %.1f с" % (IO.outputs_digest(), r["seconds"]))
        return 0
    bad = S1.check(r)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш выходов до %s, после %s" % (before, after))
    print("\n".join(EV1.evidence(r)))
    print("счёт одного полного прогона: %.1f с" % r["seconds"])
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G10–G15: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


DXA_NOTE = "G19: диагностика задним числом, не выбор (никакой результат не используется для выбора признаков и параметров A1–A3); Wald для Σ с и без спектра V5"


def main_1b(check: bool, before: str) -> int:
    """Стадия 1b (лист 1b): A1, A2, A3 по правилу; DX-A (диагностика) вызывается здесь, оркестратором, а не путём прогноза."""
    r = S1B.compute()
    dx = DXA.compute(r["tab"], r["r0"]["S"].set_index("d"), r["sigfe"].set_index("d"))
    extra = {"DXA.csv": (dx, DXA_NOTE)}
    if not check:
        IO1B.write(r, extra)
        print("стадия 1b записана; хэш выходов %s; счёт %.1f с" % (IO.outputs_digest(), r["seconds"]))
        return 0
    bad = IO1B.check(r, extra)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш выходов до %s, после %s" % (before, after))
    print("\n".join(EV1B.evidence(r, dx)))
    print("счёт одного полного прогона: %.1f с" % r["seconds"])
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G17–G22: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main_2(check: bool, before: str) -> int:
    """Стадия 2 (лист 2): C1, C2, C3 и C4 по правилу 2.2 (Measurement Uncertainty)."""
    r = S2.compute()
    if not check:
        IO2.write(r)
        print("стадия 2 записана; хэш выходов %s; счёт %.1f с" % (IO.outputs_digest(), r["seconds"]))
        return 0
    bad = IO2.check(r)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш выходов до %s, после %s" % (before, after))
    print("\n".join(EV2.evidence(r)))
    print("счёт одного полного прогона: %.1f с" % r["seconds"])
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G24–G28: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "1a", "1b", "2"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--digest-v5", action="store_true")
    a = ap.parse_args(argv)
    if a.digest_v5:
        print("хэш выходов V5: %s" % IO.v5_digest())
        return 0
    if a.stage is None:
        ap.error("нужен --stage 0, --stage 1a или --digest-v5")
    before = IO.outputs_digest()
    if a.stage == "1a":
        return main_1a(a.check, before)
    if a.stage == "1b":
        return main_1b(a.check, before)
    if a.stage == "2":
        return main_2(a.check, before)
    r = ST.compute()
    if not a.check:
        ST.write(r)
        print("стадия 0 записана; хэш выходов %s" % IO.outputs_digest())
        return 0
    bad = ST.check(r)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш выходов до %s, после %s" % (before, after))
    print("\n".join(evidence(r)))
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G1–G8: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
