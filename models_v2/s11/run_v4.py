"""HAR-J Hybrid V4, листы V4-0 (аудит) и V4-A (комплементарность, oracle). Запуск из корня репозитория, по одной стадии; без обучения моделей; процессов 1:
  python3 -W ignore -m models_v2.s11.run_v4 --stage audit [--check]
  python3 -W ignore -m models_v2.s11.run_v4 --stage A [--check]
Стадия пересчитывает свои выходы (идемпотентно, кроме времени CHECKPOINT: повторный запуск его не сдвигает); `--check` после записи проверяет гейты G1–G5 (audit) или G6–G7 (A) и печатает итог."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

import pandas as pd

from models_v2.s11 import v4_a, v4_audit as AU, v4_bench, v4_compl, v4_gaps, v4_io as IO, v4_isolation, v4_oofcode, v4_oracle, v4_trials

DONE_KEYS = ("Ж-211", "Ж-212", "Ж-214", "Ж-217")


def stage_audit() -> dict:
    inv = pd.DataFrame([r for t in IO.T17 for r in AU.inventory_rows(t)])
    IO.write_csv(inv, "INVENTORY.csv", "17 бумаг × {K1,K2,K3,LGBM-D2}: файл, SHA-256, даты, покрытие по годам цели")
    aud = AU.audit_all()
    IO.write_csv(aud, "AUDIT_ALIGN.csv", "цель RV, даты, источники, сдвиг (Д1), OOF по метаданным; код-проверка пар — v4_oofcode")
    bt, bo = v4_bench.run()
    IO.write_csv(bt, "SBER_BENCH.csv", "ориентир SBER 0,2081 (Ж-211) и ряды панели на тех же датах; итоги: " + json.dumps(bo, ensure_ascii=False))
    cp = v4_trials.write_all()
    IO.write_csv(v4_gaps.build(aud), "GAPS.csv", "пробелы для мета-обучения и оговорки уровня дизайна")
    return {"aud": aud, "bench": bo, "cp": cp, "oofcode": v4_oofcode.run()}


def check_audit(r: dict) -> list:
    bad, aud = [], r["aud"]
    inv = IO.read_csv("INVENTORY.csv")
    if len(inv) != 17 * 5 or inv[inv["model"] != "H∩K2∩K3"]["sha256"].str.len().ne(64).any() or (inv["n"] <= 0).any() or inv["by_year_target"].isna().any():
        bad.append("G1: INVENTORY неполон")
    if not (aud["align_ok"].all() and aud["oof_k2k3"].all()) or not all(r["oofcode"].values()):
        bad.append("G2: выравнивание/OOF не доказаны: %s" % r["oofcode"])
    for t in IO.T17:         # мутация «M сдвинут на +1 день» ловится проверкой сдвига
        a = IO.aligned(t)
        if AU.shift_ok(a["rv"].iloc[:-1], a["k2"].shift(-1).iloc[:-1]) or AU.shift_ok(a["rv"].iloc[:-1], a["k3"].shift(-1).iloc[:-1]):
            bad.append("G2: сдвиг M +1 не пойман на " + t)
    b = r["bench"]
    if not (b["repro_ok"] and b["rv_def_same"] and b["xj_vs_k2_maxdiff"] == 0.0 and b["lf_vs_k3_maxdiff"] == 0.0):
        bad.append("G3: ориентир SBER не воспроизведён / определение RV расходится: %s" % b)
    cp, regs = r["cp"], v4_trials.load_regs()
    bad += ["G4: " + x for x in v4_trials.verify()]
    done = (IO.REP / "DONE_EXPERIMENTS.md").read_text() if (IO.REP / "DONE_EXPERIMENTS.md").exists() else ""
    if cp["N_prior"] != v4_trials.n_prior(regs) or not all(k in done for k in DONE_KEYS):
        bad.append("G4: N_prior/DONE_EXPERIMENTS")
    g = IO.read_csv("GAPS.csv")
    if g.empty or g["blocks_meta_learning"].astype(str).eq("True").any():
        bad.append("G5: пробелы блокируют мета-обучение или реестр пуст")
    return bad


def stage_a() -> dict:
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text())      # N объявлено до расчёта (p в этом листе нет)
    comp, orc = v4_compl.compl_table(), v4_a.full_table()
    IO.write_csv(comp, "A_COMPLEMENT.csv", "комплементарность пар H-K2, H-K3, K2-K3; окна по году цели; кризис = RV IMOEX(d) > q0,9 прошлых 252 дней; p-значений нет")
    IO.write_csv(orc, "A_ORACLE.csv", "oracle bound, только диагностика (Ж-222 п. 2): O1…O4, части выигрыша; M = K2, K3, KG; p-значений нет; N=%d" % cp["N_declared_before_stats"])
    return {"comp": comp, "orc": orc}


def check_a(r: dict) -> list:
    bad, comp, orc = [], r["comp"], r["orc"]
    n_sl = 4 * 3 + 10
    if len(comp) != 17 * 3 * n_sl or set(comp["pair"]) != {"H-K2", "H-K3", "K2-K3"} or "SBER" not in set(comp["ticker"]):
        bad.append("G6: A_COMPLEMENT неполон (%d строк)" % len(comp))
    need = ["corr_err_rv", "corr_err_rel", "corr_err_log", "corr_q", "corr_dq_logratio", "share_opposite", "share_both_under", "share_both_over", "gain_top1", "gain_top5", "mean_logratio"]
    if comp[comp["n"] >= 30][need].isna().all().any():
        bad.append("G6: показатель пуст")
    if not v4_oracle.known_answer_ok():
        bad.append("G7: известный ответ oracle не сошёлся")
    bad += ["G7: " + x for x in v4_a.check_rows(orc)]
    if len(orc) != (17 + 2) * 4 * 3 or not {"M16", "SBER", "ALL17"} <= set(orc["ticker"]) or orc.duplicated(["ticker", "window", "m"]).any():
        bad.append("G7: A_ORACLE неполон (%d строк)" % len(orc))
    bad += ["G8: oracle импортирован путём прогноза: %s" % x for x in v4_isolation.violators()]
    cp_t = (IO.REP / "CHECKPOINT.json").stat().st_mtime
    if cp_t > (IO.REP / "A_ORACLE.csv").stat().st_mtime:
        bad.append("CHECKPOINT записан после A_ORACLE")
    return bad


def main_b(check: bool) -> int:
    """Этап B (G9–G14): модули v4_b*; `--check` после записи проверяет гейты и печатает итог."""
    from models_v2.s11 import v4_b_check, v4_b_stage       # локально: audit/A не зависят от модулей этапа B
    r = v4_b_stage.stage_b()
    if not check:
        print("стадия B записана; лучшая %s; конфигураций %d" % (r["best"], len(r["cfgs"])))
        return 0
    bad = v4_b_check.all_checks(r)
    print("\n".join(bad) if bad else "G9–G14: все проверки пройдены")
    print("лучшая B: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main_ca(check: bool) -> int:
    """Лист C-a (G15–G18): модули v4_z*/v4_c*; `--check` после записи проверяет гейты и печатает итог."""
    from models_v2.s11 import v4_c_check, v4_c_stage
    r = v4_c_stage.stage_ca()
    if not check:
        print("стадия Ca записана; лучшая %s" % r["best"])
        return 0
    bad = v4_c_check.all_checks(r)
    print("\n".join(bad) if bad else "G15–G18: все проверки пройдены")
    print("лучшая C-a: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main_cb(check: bool) -> int:
    """Лист C-b (G19–G21): модули v4_cb*; `--check` после записи проверяет гейты и печатает итог."""
    from models_v2.s11 import v4_cb_check, v4_cb_stage
    r = v4_cb_stage.stage_cb()
    if not check:
        print("стадия Cb записана; лучшая из C1–C10 %s" % r["best"])
        return 0
    bad = v4_cb_check.all_checks(r)
    print("\n".join(bad) if bad else "G19–G21: все проверки пройдены")
    print("лучшая из C1–C10: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main_e(check: bool) -> int:
    """Этап E (G22–G25): модули v4_e*; `--check` после записи проверяет гейты и печатает итог."""
    from models_v2.s11 import v4_e_check, v4_e_stage
    r = v4_e_stage.stage_e()
    if not check:
        print("стадия E записана; лучшая E %s" % r["best"])
        return 0
    bad = v4_e_check.all_checks(r)
    print("\n".join(bad) if bad else "G22–G25: все проверки пройдены")
    print("лучшая E: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main_cs(check: bool) -> int:
    """Лист DX-CS, Cold Start (G27–G28): прогнозы CS1, CS2 (`v4_cs*`), диагностика холодного старта и отчёт (`v4_dx_*`, вне путей прогноза)."""
    from models_v2.s11 import v4_cs_stage, v4_dx_cold, v4_dx_cs_check, v4_dx_cs_report
    r = v4_cs_stage.stage_cs()
    d = v4_dx_cold.cold_start(r)
    (v4_cs_stage.IO.REP / "CS_REPORT.md").write_text(v4_dx_cs_report.build(r, d))
    if not check:
        print("стадия CS записана; лучшая CS %s" % r["best"])
        return 0
    bad = v4_dx_cs_check.all_checks(r, d)
    print("\n".join(bad) if bad else "G27–G28: все проверки пройдены")
    print("лучшая CS: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main_dx(check: bool) -> int:
    """Лист DX (G26): диагностика недостатка информации по годам; модули `v4_dx_*` — задним числом, вне путей прогноза."""
    from models_v2.s11 import v4_dx_check, v4_dx_stage
    r = v4_dx_stage.stage_dx()
    if not check:
        print("стадия DX записана")
        return 0
    bad = v4_dx_check.all_checks(r)
    print("\n".join(bad) if bad else "G26: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


LATE = {"NTP": ("v4_ntpd_main", "main_ntp"), "CLD": ("v4_cldd_main", "main_cld"), "RQC": ("v4_rqc_main", "main_rqc")}        # листы NTP (G29–G31), CLD (G32–G34), K8-RQC (G35–G37):диагностика `v4_*d_*` вне путей прогноза, импорт — по требованию


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "A", "B", "Ca", "Cb", "E", "DX", "CS", "NTP", "CLD", "RQC"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    mains = {"CS": main_cs, "DX": main_dx, "E": main_e, "B": main_b, "Ca": main_ca, "Cb": main_cb}
    if a.stage in LATE:
        mod, fn = LATE[a.stage]
        return getattr(importlib.import_module("models_v2.s11." + mod), fn)(a.check)
    if a.stage in mains:
        return mains[a.stage](a.check)
    r = stage_audit() if a.stage == "audit" else stage_a()
    gates = "G1–G5" if a.stage == "audit" else "G6–G7"
    if not a.check:
        print("стадия %s записана" % a.stage)
        return 0
    bad = check_audit(r) if a.stage == "audit" else check_a(r)
    print("\n".join(bad) if bad else "%s: все проверки пройдены" % gates)
    if a.stage == "audit":
        print("N_prior=%d N_declared_before_stats=%d; ориентир SBER: %s; OOF по коду: %s" % (r["cp"]["N_prior"], r["cp"]["N_declared_before_stats"], r["bench"], r["oofcode"]))
        print("статусы бумаг:", r["aud"]["status"].value_counts().to_dict())
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
