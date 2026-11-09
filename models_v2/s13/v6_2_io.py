"""V6-2: запись (`write`, не в `--check`) и проверка (`check`, только чтение) выходов стадии 2: прогнозы C1–C4, признаки Q и группы u в `data/regen/harj_hybrid_v6/`, таблицы и ключ `2` в `docs/otchety/harj_hybrid_v6/`, строки `ALL_TRIALS.csv`.
Проверка сверяет диск с пересчётом в памяти (допуск 1e-9), реестр, правило C4, прогнозы A1–A3 (SHA = CHECKPOINT 1b) и регрессию листа 1a (`v6_1_stage.check`)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_2_q as QF, v6_2_stage as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 2 (SBER, h = 1; Measurement Uncertainty: C1, C2, C3, C4 по правилу)"


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
    cp, old = CP0.read(), CP0.read().get("2")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and S1.stable(old) == S1.stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "2": entry}, ensure_ascii=False, indent=1))
    CP0.ALL_TRIALS.write_text(RG.trial_text(r["ids"]), encoding="utf-8")


def gate_bad(r: dict) -> list:
    bad, w = [], r["w"]
    q = r["Qd"].drop(columns="d")
    if q.isna().any().any() or len(q) != len(r["r0"]["S"]) or list(q.columns) != QF.Q_NAMES:
        bad.append("G24: NaN в Q или состав Q на днях панели")
    for c, v in r["cfg"].items():
        a = v["a"]
        if len(a) != 2271 or not (np.isfinite(a["F"]) & (a["F"] > 0)).all() or float(np.abs(a["rv"].to_numpy() - w["rv"].to_numpy()).max()) != 0.0:
            bad.append("G25: %s не на 2271 общих датах, F ≤ 0 или цель изменена" % c)
        z = a["g"] == 0
        if z.any() and not (a.loc[z, "F"] == a.loc[z, "H"]).all():
            bad.append("G25: %s: g = 0, но F ≠ H побитно" % c)
    if ("C4" in r["cfg"]) != r["dec"]["compute"] or r["checkpoint"]["used"] not in (7, 8):
        bad.append("G27/G28: C4 посчитана не по правилу 2.2 или израсходовано не 7–8 конфигураций")
    exp = {"C1 vs A0", "C2 vs A0", "C3 vs A0", "C2 vs C1", "C3 vs C1"} | ({"C4 vs C3", "C4 vs A0"} if "C4" in r["cfg"] else set())
    if r["info"]["window"].nunique() != 4 or set(r["info"]["pair"]) != exp:
        bad.append("G25: состав C_INFO")
    cp1b = CP0.read().get("1b", {}).get("f_sha", {})
    for c, f in r["prev"].items():
        if S1.sha_arr(f) != cp1b.get(c):
            bad.append("G29: прогноз %s на диске ≠ SHA в CHECKPOINT 1b" % c)
    if r["unstable"]["n_days"].min() < 50 or set(r["unstable"]["config"]) != set(r["cfg"]):
        bad.append("G26: C_UNSTABLE: группа менее 50 дней или не все конфигурации")
    return bad


def check(r: dict) -> list:
    bad = gate_bad(r) + S1.check(r["r0"])
    for f, df in ST.parquets(r).items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for name, (df, _) in ST.tables(r).items():
        if not (IO.REP / name).exists() or not FC.frames_equal(IO.read_csv(name), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % name)
    cp = CP0.read().get("2")
    if cp is None or S1.stable(cp) != S1.stable(r["checkpoint"]):
        bad.append("G28: CHECKPOINT ключ 2 на диске ≠ пересчёту")
    if not RG.trials_ok(r["ids"], exact=True):
        bad.append("G25/G28: ALL_TRIALS.csv ≠ строки %s" % r["ids"])
    return bad
