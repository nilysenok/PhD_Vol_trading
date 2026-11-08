"""V4-I: запись выходов этапа (только `--stage I`, не `--check`): I_COMPARISON, I_PER_TICKER, I_TESTS, I_YEARS, I_SBER, I_LOTO, I_COMP (.csv), I_REPORT.md и строки реестра. Модуль вне путей прогноза."""
from __future__ import annotations

from models_v2.s11 import v4_b_decl as DECL, v4_i_out as OUT, v4_i_stage as ST, v4_io as IO


def add_derived(o: dict) -> dict:
    """Производные таблицы (в памяти): точность компонент; идентична при записи и при проверке."""
    o["acc"] = OUT.acc_table(o["cmp"], o["Fe"], o["c"])
    return o


def write_all(o: dict) -> None:
    add_derived(o)
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    grp = {**{x: "контроль" for x in DECL.CTRL}, "K8": "контроль K8", "C8": "ориентир", "CLD1": "ориентир", "H2": "ориентир", **{x: "I" for x in ST.CFGS}}
    IO.write_csv(o["comp"].assign(group=lambda d: d["series"].map(grp)), "I_COMPARISON.csv", "этап I (Ж-222 п. 10): crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021); qlike за окно 2017_2021 — простое среднее по M16; "
                 "ряды: K1, K2, K3, D2, K8, C8, CLD1, H2, I1–I3 на одинаковых датах; порог go = %.6f; %s" % (thr, note))
    IO.write_csv(o["pt"], "I_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "I_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(I) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × I1–I3 × (K1, K2, K3, D2, K8); %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["years"], "I_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели (2017 и 2018 — отдельные строки); 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "I_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "I_LOTO.csv", "LOTO лучшей I: оценка по 16 бумагам; исключённая бумага получает панельные параметры; " + note)
    IO.write_csv(o["acc"], "I_COMP.csv", "точность Ĉ (QLIKE к C), p̂ (доля верных при пороге 0,5, полнота, Брайер; константа = частота события в окне — ориентир), Ĵ и вклад Ĵ в RV; M16 — среднее по бумагам, SBER отдельно; " + note)
    (IO.REP / "I_REPORT.md").write_text(OUT.build(o))
    ST.write_trials(o["kc"]["key"])
