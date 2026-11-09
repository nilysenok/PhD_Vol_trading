"""V6-1b: запись (`write`, не в `--check`) и проверка (`check`, только чтение) выходов стадии 1b: прогнозы A1–A3 и признаки Σ в `data/regen/harj_hybrid_v6/`, таблицы и ключ `1b` в `docs/otchety/harj_hybrid_v6/`, строки `ALL_TRIALS.csv`.
Проверка сверяет диск с пересчётом в памяти (допуск 1e-9), реестр, правило A3 и регрессию листа 1a (`v6_1_stage.check`)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_feat as FE, v6_1_reg as RG, v6_1_stage as S1, v6_1b_stage as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 1b (SBER, h = 1; Path Signature: A1, A2, A3 по правилу)"


def header(note: str) -> str:
    return "# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, ST.CODE_VERSION, ST.code_hash(), IO.inputs_hash(), IO.CAVEAT, note)


def write_csv(df: pd.DataFrame, name: str, note: str):
    IO.REP.mkdir(parents=True, exist_ok=True)
    with open(IO.REP / name, "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")


def write(r: dict, extra: dict = None) -> None:
    """extra — дополнительные таблицы (DXA.csv) от оркестратора: имя -> (датафрейм, примечание)."""
    IO.OUT.mkdir(parents=True, exist_ok=True)
    for f, df in ST.parquets(r).items():
        df.to_parquet(IO.OUT / f, index=False)
    for name, (df, note) in {**ST.tables(r), **(extra or {})}.items():
        write_csv(df, name, note)
    cp, old = CP0.read(), CP0.read().get("1b")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and S1.stable(old) == S1.stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "1b": entry}, ensure_ascii=False, indent=1))
    RG.write_trials(r["ids"])


def gate_bad(r: dict) -> list:
    bad, w = [], r["w"]
    sf = r["sigfe"].drop(columns="d")
    if sf[ST.SG.SIGMA_NAMES + ST.SG.sigma3_names()].isna().any().any() or len(sf) != len(r["r0"]["S"]):
        bad.append("G17: NaN в Σ / Σ³ на днях панели")
    for c, v in r["cfg"].items():
        a = v["a"]
        if len(a) != 2271 or not (np.isfinite(a["F"]) & (a["F"] > 0)).all() or float(np.abs(a["rv"].to_numpy() - w["rv"].to_numpy()).max()) != 0.0:
            bad.append("G18: %s не на 2271 общих датах, F ≤ 0 или цель изменена" % c)
        z = a["g"] == 0
        if z.any() and not (a.loc[z, "F"] == a.loc[z, "H"]).all():
            bad.append("G18: %s: g = 0, но F ≠ H побитно" % c)
    if ("A3" in r["cfg"]) != r["dec"]["compute"] or r["checkpoint"]["used"] not in (3, 4):
        bad.append("G21/G22: A3 посчитана не по правилу 2.2 или израсходовано не 3–4 конфигурации")
    if r["info"]["window"].nunique() != 4 or set(r["info"]["pair"]) != ({"A1 vs A0", "A2 vs A0", "A2 vs A1"} | ({"A3 vs A1"} if "A3" in r["cfg"] else set())):
        bad.append("G18: состав A_INFO")
    return bad


def check(r: dict, extra: dict = None) -> list:
    bad = gate_bad(r) + S1.check(r["r0"])
    for f, df in ST.parquets(r).items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for name, (df, _) in {**ST.tables(r), **(extra or {})}.items():
        if not (IO.REP / name).exists() or not FC.frames_equal(IO.read_csv(name), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % name)
    cp = CP0.read().get("1b")
    if cp is None or S1.stable(cp) != S1.stable(r["checkpoint"]):
        bad.append("G22: CHECKPOINT ключ 1b на диске ≠ пересчёту")
    if not RG.trials_ok(r["ids"]):
        bad.append("G18/G22: ALL_TRIALS.csv ≠ строки %s" % r["ids"])
    return bad
