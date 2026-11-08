"""V4-H: запись выходов этапа (только в режиме `--stage H`, не `--check`): H_COMPARISON, H_PER_TICKER, H_TESTS, H_YEARS, H_SBER, H_LOTO, H_VALUE, H_COEF (.csv), H_REPORT.md и строки реестра. Модуль вне путей прогноза."""
from __future__ import annotations

import pandas as pd

from models_v2.s11 import v4_b_decl as DECL, v4_h_stage as ST, v4_hd_report as RP, v4_hd_value as V, v4_io as IO


def add_derived(o: dict) -> dict:
    """Производные таблицы (в памяти): ценность источников; идентична при записи и при проверке."""
    o["cfgs"] = ST.CFGS
    o["value"] = V.build(o["c"], o["comp"], o["tests"], ST.CFGS)
    o["coef"] = pd.concat([o["cf"][c] for c in ST.CFGS], ignore_index=True)
    return o


def write_all(o: dict) -> None:
    add_derived(o)
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    grp = {**{x: "контроль" for x in DECL.CTRL}, "K8": "контроль K8", "C8": "ориентир", "CLD1": "ориентир", **{x: "H" for x in ST.CFGS}}
    IO.write_csv(o["comp"].assign(group=lambda d: d["series"].map(grp)), "H_COMPARISON.csv", "этап H (Ж-225, Ж-222): crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021); qlike за окно 2017_2021 — простое среднее по M16; "
                 "ряды: K1, K2, K3, D2, K8, C8, CLD1, H1–H8 на одинаковых датах; порог go = %.6f; %s" % (thr, note))
    IO.write_csv(o["pt"], "H_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "H_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(H) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × H1–H8 × (K1, K2, K3, D2, K8); %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["years"], "H_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели (2017 и 2018 — отдельные строки); 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "H_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "H_LOTO.csv", "LOTO лучшей H: оценка по 16 бумагам, поперечные величины (медианы, секторные соседи, brk) по 16, исключённая получает панельные θ (η = 0); " + note)
    IO.write_csv(o["value"], "H_VALUE.csv", "вклад каждого источника: Δ QLIKE к K8 по окнам (M16), DM к K8; доля строк M16 с RV вне [min(H, M), max(H, M)] и снижение потерь на них (gain = QLIKE(K8) − QLIKE(H), плюс = лучше); " + note)
    IO.write_csv(o["coef"], "H_COEF.csv", "θ панели по месяцам (th_<признак>, при стандартизованном ортогонализованном z̃); n_train — строки обучения (до 60 — g = 0); converged — max|∇| < 1e-6; " + note)
    (IO.REP / "H_REPORT.md").write_text(RP.build(o))
    ST.write_trials(o["kc"]["key"])
