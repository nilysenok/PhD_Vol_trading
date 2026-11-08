"""HAR-J Hybrid V6, лист 0 (Ж-227), только SBER, h = 1. Запуск из корня репозитория, процесс один:
  python3 -W ignore -m models_v2.s13.run_v6 --stage 0          # пересчитать и записать выходы листа
  python3 -W ignore -m models_v2.s13.run_v6 --stage 0 --check  # ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, суммарный хэш выходов до и после
  python3 -W ignore -m models_v2.s13.run_v6 --digest-v5        # суммарный хэш выходов V5 (G9: до и после проверок V5)
Моделей не обучает; конфигурации не расходуются."""
from __future__ import annotations

import argparse
import sys

from models_v2.s13 import v6_0_io as IO, v6_0_stage as ST


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--digest-v5", action="store_true")
    a = ap.parse_args(argv)
    if a.digest_v5:
        print("хэш выходов V5: %s" % IO.v5_digest())
        return 0
    if a.stage is None:
        ap.error("нужен --stage 0 или --digest-v5")
    before = IO.outputs_digest()
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
