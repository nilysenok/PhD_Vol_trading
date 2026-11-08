"""V5 лист FIN, G23: сводные таблицы FIN_ALL, FIN_TICKERS, FIN_YEARS, FIN_SBER, FIN_TESTS, FIN_MCS. Определения прежние — функции листа 1a (`v5_gs_eval`), без правок: общие строки `S.common`, метрики `v4_b_stats.metr`,
`crit_panel` = `hp_stats_sel.crit_panel`, DM (НУ 5) и бутстрап дат `hp_stats_tst.pair_stats`, MCS 90 %, Холм `stats_d.holm_n`. Окна — по году цели. SBER в выбор (M16) не входит.
Числа GS/SP совпадают с таблицами листов 1a и 2 (сверка в `v5_fin_stage.check`, допуск 1e-12)."""
from __future__ import annotations

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_fin_data as D, v5_gs_eval as EV

COLS_CRIT = EV.full_cols(D.GS + D.SP) + ["M"]


def crit_inputs(d: dict) -> tuple:
    """-> (crit_panel по рядам, EqY по рядам) на общем кадре; M — отдельным рядом."""
    cs = S.crit_panel(d["kk"], COLS_CRIT).set_index("series")["crit_panel"]
    return cs, EV.eqy_series(d["kk"], COLS_CRIT)


def tables_main(d: dict) -> dict:
    """FIN_ALL (метрики по 8 окнам + три условия «go»), FIN_TICKERS, FIN_YEARS, FIN_SBER, FIN_TESTS, FIN_MCS."""
    cc, cp = d["cc"], d["cp"]
    cs, eqy = crit_inputs(d)
    pt = EV.metrics(cc, D.SER)
    go = EV.go_table(d["kk"], D.GS + D.SP, cp["go_thresholds"], eqy)
    allt = pd.concat([EV.comparison(cc, D.SER, pt, cs, eqy), go], ignore_index=True)
    return {"FIN_ALL": allt, "FIN_TICKERS": EV.per_ticker(pt, D.GS + D.SP + ["M"]), "FIN_YEARS": EV.years(cc, D.SER), "FIN_SBER": EV.ST.sber(cc, D.SER, EV.SBER_REF),
            "FIN_TESTS": EV.tests(cc, [d["best_gs"], d["best_sp"]], cp["N_declared_before_stats"], cp["N_wide"]), "FIN_MCS": EV.ST.mcs(cc, D.SER), "_pt": pt, "_cs": cs, "_eqy": eqy, "_go": go}
