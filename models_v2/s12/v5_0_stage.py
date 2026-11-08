"""V5-0: стадия 0 = пересчёт всех таблиц листа в памяти (`compute`, чистая функция от входов), запись (`write`, только не в `--check`) и сверка с диском (`check`, только чтение).
Моделей не обучает, конфигурации не расходуются. Диагностические модули `v5_0_dx*` подключаются здесь (стадия вне путей прогноза)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_fin_check as FC, v4_fin_data as FD, v4_io as IO4
from models_v2.s12 import v5_0_bars as BA, v5_0_crit as CR, v5_0_ctrl as CT, v5_0_dx_jump as DJ, v5_0_dx_score as DS, v5_0_dx_spec as DX, v5_0_io as IO, v5_0_reg as RG

TABLES = {"CONTROLS.csv": "controls", "CRIT_AUDIT.csv": "crit_audit", "CRIT_EQY.csv": "crit_eqy", "PASS_RV.csv": "pass_rv", "PASS_SCORE.csv": "pass_score", "PASS_JUMP.csv": "pass_jump", "PASS_SPECTRAL.csv": "pass_spec", "SEARCH_HISTORY.csv": "search"}
NOTES = {"controls": "G1: контроли K1,K2,K3,M,D2,K8 побитно прежние (SHA, K8 пересчитан, crit_panel, SBER, сдвиг +1)", "crit_audit": "G2: crit_panel = аналитическая формула по годовым QLIKE; веса годов; покрытие дат",
         "crit_eqy": "G3: EqY (равновесная по годам) рядом с crit_panel — диагностика, не подтверждение и не выбор", "pass_rv": "G4: цель RV, разложение Парсеваля DCT-II, распределение n",
         "pass_score": "G5: score s = RV/F − 1, база H/M/K8; диагностика задним числом, гиперпараметры не выбирает", "pass_jump": "G6: дни скачков J>0 и доля потерь; диагностика задним числом",
         "pass_spec": "G7: доли энергии полос DCT-II и частные корреляции; диагностика задним числом", "search": "G10: реестры испытаний h=1; N по объединению ключей, не по сумме строк"}


def compute() -> dict:
    """Все таблицы листа в памяти + резюме для проверок."""
    k = IO.frame()
    ctrl, cres = CT.controls_table(k)
    cols = FD.CTRL + FD.CFGS + ["M"]
    audit, ver, ann = CR.audit_tables(k, cols)
    cp = S.crit_panel(k, FD.CTRL + FD.CFGS).set_index("series")["crit_panel"]
    eqy, _ = CR.eqy_table(ann, S.crit_panel(k, cols).set_index("series")["crit_panel"], FD.STAGE_OF)
    b = BA.cached_bands(write=False)
    rv = pd.concat([BA.definition_rows(), BA.identity_table(b, k), BA.n_distribution(b)], ignore_index=True)
    prior, v4, v5 = RG.load_regs()
    search, info = RG.history_table(prior, v4, v5)
    thr = RG.go_thresholds(k, cp)
    return dict(controls=ctrl, cres=cres, crit_audit=audit, ver=ver, ann=ann, crit_eqy=eqy, pass_rv=rv, bands=b, pass_score=DS.score_table(k), pass_jump=DJ.jump_table(k, b), pass_spec=DX.spec_table(k, b),
                search=search, info=info, thr=thr, cp=cp, checkpoint=RG.checkpoint(info, thr, cp), v5_trials=v5)


def write(r: dict) -> list:
    """Запись таблиц, кэша энергий полос, пустого ALL_TRIALS (заголовок) и CHECKPOINT; возвращает пути."""
    out = [IO.write_csv(r[v], f, NOTES[v]) for f, v in TABLES.items()]
    BA.cached_bands(write=True)
    if not RG.ALL_TRIALS.exists():
        RG.ALL_TRIALS.write_text(RG.HEADER, encoding="utf-8")
    RG.write_checkpoint(r["checkpoint"])
    return out


def g1(r: dict) -> list:
    c, bad = r["cres"], []
    if not (c["files_ok"] and c["n_files"] == 68 and c["n_tickers"] == 17 and c["audit_ok"]):
        bad.append("G1: SHA входов / 17 бумаг / AUDIT_ALIGN: %s" % {k: c[k] for k in ("files_ok", "n_files", "n_tickers", "audit_ok")})
    if not c["k8_bitwise"] or not c["crit_ok"] or c["crit_v4_diff"] > 1e-12 or not c["shift_ok"] or not c["sber_ok"]:
        bad.append("G1: K8 побитно %s, crit %s (diff V4 %.1e), сдвиг %s, SBER %s" % (c["k8_bitwise"], c["crit_ok"], c["crit_v4_diff"], c["shift_ok"], c["sber_ok"]))
    return bad


def g2g3(r: dict) -> list:
    v, bad = r["ver"], []
    if v["max_abs_diff"] > CR.TOL or len(r["crit_audit"][r["crit_audit"]["section"] == "series"]) != 51 or not v["years_all"] or abs(v["w2017"] - 25 / 48) > 1e-12 or v["w2021"] != 0.0:
        bad.append("G2: формула/веса: diff %.2e, веса 2017 %.6f 2021 %.1f" % (v["max_abs_diff"], v["w2017"], v["w2021"]))
    e, fy = r["crit_eqy"], IO4.read_csv("FIN_YEARS.csv")
    mine = r["ann"].groupby(["series", "year"])["ann"].mean()
    d = max(abs(mine[(s, int(y))] - a) for s, y, a in zip(fy["series"], fy["year"], fy["ann_qlike"]) if (s, int(y)) in mine.index)
    if len(e) != 51 or d > 1e-12:
        bad.append("G3: CRIT_EQY строк %d, расхождение с FIN_YEARS %.2e" % (len(e), d))
    return bad


def g4(r: dict) -> list:
    t = r["pass_rv"]
    t = t[t["section"] == "identity"]
    if len(t) != 17 or t["max_rel_bars_vs_daily"].max() > 1e-10 or t["max_rel_bars_vs_target"].max() > 1e-10 or t["max_rel_parseval"].max() > 1e-12:
        return ["G4: тождество/Парсеваль: %s" % t[["max_rel_bars_vs_daily", "max_rel_bars_vs_target", "max_rel_parseval"]].max().to_dict()]
    return []


def g10(r: dict) -> list:
    c, bad = r["checkpoint"], []
    if (c["N_prior"], c["N_declared_before_stats"], c["N_wide"]) != (680, 704, 3104) or len(r["v5_trials"]) != 0 or c["N_wide_optuna_trials"] != 2400:
        bad.append("G10: N %s" % ((c["N_prior"], c["N_declared_before_stats"], c["N_wide"]),))
    if not RG.CP.exists() or not RG.ALL_TRIALS.exists():
        return bad + ["G10: нет CHECKPOINT.json/ALL_TRIALS.csv на диске"]
    d = json.loads(RG.CP.read_text())
    if {k: v for k, v in d.items() if k not in ("written_at", "GS", "SP")} != json.loads(json.dumps({k: v for k, v in c.items() if k != "written_at"})):
        bad.append("G10: CHECKPOINT на диске ≠ пересчёту")
    return bad


def check(r: dict) -> list:
    """Сверка пересчёта в памяти с таблицами на диске (только чтение) + проверки гейтов G1–G4, G10."""
    bad = g1(r) + g2g3(r) + g4(r) + g10(r)
    for f, v in TABLES.items():
        if not (IO.REP / f).exists():
            bad.append("%s: нет файла" % f)
        elif not FC.frames_equal(IO.read_csv(f), r[v], 1e-9):
            bad.append("%s: диск ≠ пересчёт в памяти" % f)
    for v, need in (("pass_score", {"H", "M", "K8"}), ("pass_jump", {"freq", "conc"}), ("pass_spec", {"band", "partial"})):
        col = "base" if v == "pass_score" else "section"
        if not need <= set(r[v][col].dropna()):
            bad.append("G5–G7: %s неполна" % v)
    return bad
