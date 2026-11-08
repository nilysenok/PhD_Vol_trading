"""V4-FIN (G45–G47): стадия `FIN`. Новых конфигураций и обучений нет (0 из 60): сохранённые прогнозы 46 конфигураций + контроли -> общие даты -> таблицы FIN_*.csv -> итоговый документ.
`stage_fin(write=False)` — режим `--check`: только чтение (пересчёт в памяти, сверка с диском); `write=True` пишет FIN_*.csv, документ, REPRODUCTION_V4.md и ключи FIN в CHECKPOINT.json."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from models_v2.s11 import v4_b_decl as DECL, v4_b_stage as BS, v4_b_stats as ST, v4_fin_data as FD, v4_fin_stats as FS, v4_io as IO
from models_v2.s10 import hp_stats_sel as S

NOTE = "итог программы HAR-J Hybrid V4 (FIN): 46 конфигураций + контроли на общих датах; crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021); qlike_simple_2017_2021 — простое среднее по M16; порог go = %.6f; без новых конфигураций"
OUTS = {"FIN_ALL.csv": "all", "FIN_TICKERS.csv": "tickers", "FIN_SBER.csv": "sber", "FIN_TESTS.csv": "tests", "FIN_MCS.csv": "mcs", "FIN_YEARS.csv": "years"}


def declare() -> dict:
    """Один раз до первого файла FIN: время объявления и правило отбора трёх лучших (после — не меняется)."""
    cp = json.loads(DECL.CP.read_text())
    if "FIN_declared_at" not in cp:
        DECL.update(FIN_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), FIN_rule="три лучшие архитектуры = три конфигурации из 46 (K8 включена) с наименьшим crit_panel (M16 × 2017–2021); окна 2022+ в выбор не входят; FIN конфигураций не расходует",
                    FIN_configs_spent=0, FIN_prior_parquet_sha=FD.parquet_sha(), FIN_prior_tables_sha=FD.tables_sha())
        cp = json.loads(DECL.CP.read_text())
    return cp


def compute(cp: dict) -> dict:
    P, k = FD.frame()
    cols = FD.CTRL + FD.CFGS
    c = S.common(k, cols)
    cs = FS.crit_series(k, cols)
    top = FS.best3(cs)
    g = FS.go_rules(k, cs, cp["go_threshold_B"])
    pt = FS.per_ticker(c, cols)
    cond = FS.conditions(g, cols)
    o = {"P": P, "k": k, "c": c, "cols": cols, "cs": cs, "top": top, "go": g, "cond": cond, "pt": pt, "cp": cp}
    o["all"] = FS.all_table(c, cols, cs, pt, cond)
    o["tickers"] = FS.tickers_table(pt, top)
    o["sber"] = ST.sber(c, cols, BS.sber_ref())
    o["tests"] = FS.tests_table(c, top, cp)
    o["mcs"] = FS.mcs_table(c, cols)
    o["years"] = FS.years_table(k, cols)
    return o


def write_tables(o: dict) -> None:
    n, thr = o["cp"], o["cp"]["go_threshold_B"]
    note = NOTE % thr
    top = ", ".join(o["top"])
    IO.write_csv(o["all"], "FIN_ALL.csv", note + "; окна 2017, 2018, 2017_2021, 2022_2025, 2026, all; Δ — в % уровня K1 того же окна; три лучшие: " + top)
    IO.write_csv(o["tickers"], "FIN_TICKERS.csv", "17 бумаг (SBER — диагностика, в выбор не входит); ряды K1, K2, K3, D2, K8, C1, C4 и три лучшие; " + note)
    IO.write_csv(o["sber"], "FIN_SBER.csv", "SBER на общих датах (2271 в окне all); ориентир 0,2081 (A1_roll_126, Ж-211) — потолок; " + note)
    IO.write_csv(o["tests"], "FIN_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(конфигурация) − QLIKE(контроль); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide); семейство M16 × {2022_2025, all}; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["mcs"], "FIN_MCS.csv", "MCS 90 %% (T_max, блочный бутстрап дат) по всем рядам; панельный QLIKE M16 по датам; %s" % note)
    IO.write_csv(o["years"], "FIN_YEARS.csv", "ann_qlike — средний QLIKE по M16 за год цели 2017…2026; pen, crit_part — только 2017–2021 (сумма crit_part = crit_panel); %s" % note)


def stage_fin(write: bool = True) -> dict:
    cp = declare() if write else json.loads(DECL.CP.read_text())
    if "FIN_declared_at" not in cp:
        raise RuntimeError("FIN не объявлен: сначала `run_v4 --stage FIN`")
    o = compute(cp)
    if write:
        from models_v2.s11 import v4_fin_write as W
        write_tables(o)
        W.write_docs(o)
        DECL.update(FIN_top3=o["top"], FIN_best=o["top"][0], FIN_go_any=bool(any(v["go"] for v in o["go"]["cfg"].values())), FIN_cache_note="производные таблицы; прогнозы и таблицы этапов не менялись")
    return o
