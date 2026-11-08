"""HAR-J Hybrid V5, лист 0 (Ж-226). Запуск из корня репозитория, процесс один:
  python3 -W ignore -m models_v2.s12.run_v5 --stage 0          # пересчитать и записать выходы листа
  python3 -W ignore -m models_v2.s12.run_v5 --stage 0 --check  # ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, суммарный хэш выходов до и после
Моделей не обучает; конфигурации не расходуются."""
from __future__ import annotations

import argparse
import sys

from models_v2.s12 import v5_0_io as IO, v5_0_stage as ST


def evidence(r: dict) -> list:
    """Строки для EVIDENCE G1–G7, G10 (числа из пересчёта)."""
    c, v, t = r["cres"], r["ver"], r["thr"]
    e = r["crit_eqy"].set_index("series")
    rho = e["rank_EqY"].corr(e["rank_crit"], method="spearman")
    rv = r["pass_rv"]
    rv = rv[rv["section"] == "identity"]
    ck = r["checkpoint"]
    return ["G1: файлов SHA ок %s из %d; бумаг %d; K8 побитно %s; crit_panel %s; SBER %s" % (c["files_ok"], c["n_files"], c["n_tickers"], c["k8_bitwise"], {k: round(float(c["crit"][k]), 6) for k in ("k1", "k2", "k3", "d2", "K8")}, c["sber"]),
            "G2: класс «%s»; max|crit_panel − формула| = %.2e; вес 2017 = %.6f, 2021 = %.1f; доля пропусков %.4f; бумаг %d; K1 на общих датах %.6f, на своих %.6f" % (v["kind"], v["max_abs_diff"], v["w2017"], v["w2021"], v["missing_share"], v["n_tickers"], v["k1_common"], v["k1_own_dates"]),
            "G3: CRIT_EQY строк %d; Спирмен(ранг EqY, ранг crit) = %.4f; топ-3 по crit %s; их ранг по EqY %s" % (len(e), rho, list(e.sort_values("rank_crit").index[:3]), [int(e.loc[s, "rank_EqY"]) for s in e.sort_values("rank_crit").index[:3]]),
            "G4: бумаг %d; max отн. разн. Σr² к RV канона %.2e, к цели %.2e, Парсеваль %.2e" % (len(rv), rv["max_rel_bars_vs_daily"].max(), rv["max_rel_bars_vs_target"].max(), rv["max_rel_parseval"].max()),
            "G5–G7: строк PASS_SCORE %d, PASS_JUMP %d, PASS_SPECTRAL %d" % (len(r["pass_score"]), len(r["pass_jump"]), len(r["pass_spec"])),
            "G10: N_prior %d, N_declared %d, N_wide %d; пороги go: %s" % (ck["N_prior"], ck["N_declared_before_stats"], ck["N_wide"], {k: round(float(t[k]), 6) for k in ("cond1_crit_panel", "cond2_crit_no2017", "cond3_crit_trim1")})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["0", "GS", "SP"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cfg", choices=["GS1", "GS2", "GS3", "GS4", "GS5"])
    ap.add_argument("--decide-form", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "SP":
        from models_v2.s12 import v5_sp_main
        return v5_sp_main.main(a)
    if a.stage == "GS":
        from models_v2.s12 import v5_gs_main
        return v5_gs_main.main(a)
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
    print("\n".join(bad) if bad else "G1–G7, G10: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
