"""V6-6: запись (`write`, не в `--check`) и проверка (`check`, только чтение) выходов стадии 6: прогноз A0g в `data/regen/harj_hybrid_v6/`, таблицы `A0g_*`, `G_INFO.csv`, `G_LEVELS.csv` и ключ `6` в `docs/otchety/harj_hybrid_v6/`, строка `Ж-227:A0g` в `ALL_TRIALS.csv`.
Проверка сверяет диск с пересчётом в памяти (допуск 1e-9), реестр, SHA прежних прогнозов и S (CHECKPOINT 1a…5), копию прежних строк против A0 с A_INFO/C_INFO/E_INFO, набор признаков A0g = S."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_1b_tab as TB, v6_6_stage as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 6 (SBER, h = 1; абляционная база A0g = S + GBM; ОТСТУПЛЕНИЕ от предрегистрации, вопрос 1)"


def header(note: str) -> str:
    return "# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, ST.CODE_VERSION, ST.code_hash(), IO.inputs_hash(), IO.CAVEAT, note)


def write_csv(df: pd.DataFrame, name: str, note: str):
    IO.REP.mkdir(parents=True, exist_ok=True)
    with open(IO.REP / name, "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")


def write(r: dict) -> None:
    IO.OUT.mkdir(parents=True, exist_ok=True)
    for f, df in ST.parquets(r).items():
        df.to_parquet(IO.OUT / f, index=False)
    for name, (df, note) in ST.tables(r).items():
        write_csv(df, name, note)
    cp, old = CP0.read(), CP0.read().get("6")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and S1.stable(old) == S1.stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "6": entry}, ensure_ascii=False, indent=1))
    CP0.ALL_TRIALS.write_text(RG.trial_text(r["ids"]), encoding="utf-8")


def old_copy_bad(r: dict) -> list:
    """Прежние строки против A0 — побитно строки A/C/E_INFO; независимый пересчёт пары (Δ QLIKE, p DM) совпадает с ними."""
    bad, info = [], r["info"]
    old = info[info["role"] == ST.ROLE_OLD].reset_index(drop=True)
    if not FC.frames_equal(old.drop(columns="role"), ST.old_rows().reset_index(drop=True)[list(old.columns.drop("role"))], 0.0):
        bad.append("G47: прежние строки против A0 ≠ A/C/E_INFO")
    fresh = TB.info_table(r["fs"], [(x, "A0") for x in ST.RULE_X], r["w"], r["cp0"])
    fresh = fresh[fresh["window"] == "2017_2021"].set_index("pair")
    for x in old[old["window"] == "2017_2021"].itertuples():
        if abs(fresh.loc[x.pair, "delta_qlike"] - x.delta_qlike) > 1e-9 or abs(fresh.loc[x.pair, "dm_p_one"] - x.dm_p_one) > 1e-9 or bool(fresh.loc[x.pair, "new_information"]) != bool(x.new_information):
            bad.append("G47: прежний вывод %s не воспроизводится из прогнозов" % x.pair)
    return bad


def gate_bad(r: dict) -> list:
    bad, w, cp, c = [], r["w"], r["cp"], r["cfg"][ST.CID]
    a = c["a"]
    if len(a) != 2271 or not (np.isfinite(a["F"]) & (a["F"] > 0)).all() or float(np.abs(a["rv"].to_numpy() - w["rv"].to_numpy()).max()) != 0.0:
        bad.append("G46: A0g не на 2271 общих датах, F ≤ 0 или цель изменена")
    z = a["g"] == 0
    if z.any() and not (a.loc[z, "F"] == a.loc[z, "H"]).all():
        bad.append("G46: A0g: g = 0, но F ≠ H побитно")
    if c["names"] != list(r["S"].columns) or "A0g" in r["prior_ids"] or len(r["prior_ids"]) != 17:
        bad.append("G46: A0g не на признаках S или прежних конфигураций не 17")
    if r["checkpoint"]["used"] != 18 or r["checkpoint"]["used"] != int(r["cp0"]["budget_V6"]):
        bad.append("G48: израсходовано не 18 из 18")
    if set(r["info"]["pair"]) != {"%s vs A0g" % x for x in ST.RULE_X} | {"%s vs A0" % x for x in ST.RULE_X} | {"A0g vs %s" % y for y in ST.DESC_Y} or r["info"]["window"].nunique() != 4:
        bad.append("G47: состав G_INFO")
    if len(r["levels"]) != 5:
        bad.append("G48: состав G_LEVELS")
    sha = ST.expected_sha(cp)
    bad += ["G46: прогноз %s на диске ≠ SHA в CHECKPOINT 1a…5" % k for k, f in r["prev"].items() if S1.sha_arr(f) != sha[k]]
    if cp["1a"]["s_features_sha"] != S1.sha_arr(r["S"].reset_index(drop=True)):
        bad.append("G46: S_features ≠ SHA в CHECKPOINT 1a")
    t = r["tab"].reindex(w.index)
    if float(np.abs(t["rv_next"] - w["rv"]).max()) != 0.0 or float(np.abs(t["h"] - w["K1"]).max()) != 0.0:
        bad.append("G46: таблица H ≠ общим датам листа 0")
    return bad + old_copy_bad(r)


def check(r: dict) -> list:
    bad = gate_bad(r)
    for f, df in ST.parquets(r).items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for name, (df, _) in ST.tables(r).items():
        if not (IO.REP / name).exists() or not FC.frames_equal(IO.read_csv(name), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёту" % name)
    cp = CP0.read().get("6")
    if cp is None or S1.stable(cp) != S1.stable(r["checkpoint"]):
        bad.append("G48: CHECKPOINT ключ 6 на диске ≠ пересчёту")
    if not RG.trials_ok(r["ids"], exact=True):
        bad.append("G46/G48: ALL_TRIALS.csv ≠ строки %s" % r["ids"])
    txt = CP0.ALL_TRIALS.read_text(encoding="utf-8") if CP0.ALL_TRIALS.exists() else ""
    if "ОТСТУПЛЕНИЕ" not in RG.CONFIGS[ST.CID]["source"] or "Ж-227:A0g," not in txt:
        bad.append("G46: строка Ж-227:A0g без пометки отступления")
    if not (IO.REP / "LEAF6_REPORT.md").exists():
        bad.append("G46: нет LEAF6_REPORT.md")
    return bad
