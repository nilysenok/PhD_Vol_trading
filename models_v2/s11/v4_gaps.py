"""V4-0, G5: реестр пробелов для мета-обучения на 17 бумагах: вычисляемые по данным и оговорки уровня дизайна (без догадок: что нельзя доказать по метаданным — строка)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_io as IO

COLS = ["id", "kind", "tickers", "what", "evidence", "blocks_meta_learning", "minimal_restore"]


def _row(i, kind, tickers, what, evidence, blocks, restore) -> dict:
    return dict(zip(COLS, [i, kind, tickers, what, evidence, blocks, restore]))


def data_gaps(al: pd.DataFrame, aud: pd.DataFrame) -> list:
    """Вычисляемые пробелы: календарь бумаг, D2 на общих датах, неполный 2026, OOF-статус."""
    rows, cal = [], {t: set(al[t]["date"]) for t in al}
    union = set().union(*cal.values())
    short = {t: len(union - cal[t]) for t in cal if len(union - cal[t])}
    rows.append(_row("D1", "calendar", ";".join("%s:%d" % x for x in short.items()) or "—", "бумаги с меньшим числом общих дат H∩K2∩K3, чем объединённый календарь 17 (дыры собственного календаря бумаги)",
                     "число дат объединённого календаря, которых нет у бумаги: " + (str(short) if short else "0"), False,
                     "не требуется: прогнозы строятся на собственных торговых днях бумаги (R2); мета-слой обучается по бумаге, на общих датах H∩K2∩K3"))
    miss = {t: int((~np.isfinite(al[t]["d2"])).sum()) for t in al if (~np.isfinite(al[t]["d2"])).any()}
    rows.append(_row("D2", "coverage", ";".join("%s:%d" % x for x in miss.items()) or "—", "LGBM-D2 не определён на части общих дат H∩K2∩K3", "число дат без D2: " + (str(miss) if miss else "0"), False,
                     "не требуется: D2 — контроль, в мета-обучение не входит" if miss else "пробела нет"))
    n26 = {t: int((al[t]["t_plus"].dt.year == 2026).sum()) for t in al}
    rows.append(_row("D3", "partial_year", "все 17", "2026 неполный (цели до 27.02.2026)", "число дат 2026 по бумагам: %d…%d" % (min(n26.values()), max(n26.values())), False,
                     "не требуется; окно 2026 в выводах — только описательное (n мало)"))
    bad = aud[aud["status"] != "OOF-пригоден"]["ticker"].tolist()
    rows.append(_row("D4", "oof_status", ";".join(bad) or "—", "бумаги, у которых K1/K2/K3 не доказаны как OOF по метаданным и коду" if bad else "бумаг без доказанного OOF нет",
                     "AUDIT_ALIGN.csv, колонка status", bool(bad), "переобучение не требуется" if not bad else "разбор до причины; без массового переобучения"))
    return rows


def design_caveats() -> list:
    """Оговорки, которые метаданными не устранить (фиксируются, не угадываются)."""
    return [
        _row("C1", "design_selection", "SBER (строго), M16 (слабее)", "Дизайн K2/K3 (сдвиги `x2_best_j` в X2_DECISION.json, набор F4, `accepted.json`) выбран в Ж-206/210 на SBER (критерий crit_pen, 20 проб); охват периода выбора по метаданным не доказан — считать, что он захватывает и годы после 2021 (❓); годовые HP (И2) — PIT, дизайн — нет",
             "docs/otchety/xgb_lgbm_full_v1/xgboost/X2_DECISION.json; models_v2/s10/hp_k23_world.py, hp_k23_calc.py", False,
             "не устраняется без переобучения; в отчётах — потолок; для M16 дизайн выбран на другой бумаге, но на пересекающемся периоде; мета-слой на закрытых парах не усиливает эту оговорку"),
        _row("C2", "design_selection", "17", "LGBM-D2 выбран финалистом по crit_panel (M16 × 2017–2021) среди 20 гибридов — выбор по тем же данным, где он затем сравнивается", "HARJ_PANEL_HYBRID_FINAL.md §1", False,
             "использовать только как контроль; в окнах 2017–2021 сравнение с ним занижает отрыв от него (потолок для D2)"),
        _row("C3", "benchmark", "SBER", "Ориентир 0,2081 (A1_roll_126) — потолок: c, λ выбраны по crit_wf на всей истории, а не PIT", "TABLE17_SBER_H1.csv, пометка «потолок»", False,
             "не использовать как цель «go»; stop/go этапов — по crit_panel (Ж-222 п. 6)"),
        _row("C4", "meta_missing", "17", "У LGBM-D2 нет файла метаданных вида k23_meta; OOF доказан по коду движка (`train_sets`, `closed_mask`) и описанию ядра, не по записи обучения", "v4_oofcode.py::check_d2, check_closed_choice", False,
             "не требуется: D2 — контроль"),
        _row("C5", "registry", "17", "Прежние Optuna-поиски h = 1 диссертации (`data/regen/*h1*trials*.csv`, другая программа и модели) в реестрах гибридов отсутствуют; N_prior = 634 их не включает", "PRIOR_TRIALS.csv, строки other_h1_file", False,
             "решение дирижёра/владельца (класс B/A): включать ли их в N Холма; N_declared_before_stats при включении растёт — до первого p"),
    ]


def build(aud: pd.DataFrame) -> pd.DataFrame:
    al = {t: IO.aligned(t) for t in IO.T17}
    return pd.DataFrame(data_gaps(al, aud) + design_caveats(), columns=COLS)
