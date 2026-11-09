"""V6-3: запись (`write`, не в `--check`) и проверка (`check`, только чтение) выходов стадии 3: прогнозы E1–E3 и признаки R в `data/regen/harj_hybrid_v6/`, таблицы и ключ `3` в `docs/otchety/harj_hybrid_v6/`, строки `ALL_TRIALS.csv`.
Проверка сверяет диск с пересчётом в памяти (допуск 1e-9), реестр, правило E3, прогнозы A1–A3 и C1–C4 (SHA = CHECKPOINT 1b/2), Σ (SHA = ключ 1b), известные ответы диагностики обращения и регрессию листа 1a (`v6_1_stage.check`)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_1b_sig as SG, v6_3_r as RF, v6_3_stage as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 3 (SBER, h = 1; Time-Reversal / Shock-Recovery: E1, E2, E3 по правилу; диагностика B)"


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
    cp, old = CP0.read(), CP0.read().get("3")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and S1.stable(old) == S1.stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "3": entry}, ensure_ascii=False, indent=1))
    CP0.ALL_TRIALS.write_text(RG.trial_text(r["ids"]), encoding="utf-8")


def tr_bad(t: pd.DataFrame) -> list:
    """Известные ответы диагностики: режим `poryadok` — все 12 признаков Σ меняют знак, s1…s7 инвариантны; 27 признаков × 2 режима."""
    bad = [] if len(t) == 54 else ["G31: строк TR_DIAG %d, ожидалось 54" % len(t)]
    p = t[t["mode"] == "poryadok"].set_index(["group", "feature"])["verdict"]
    if not (p.loc["Sigma"] == "меняет знак").all() or not all(p.loc[("S", "s%d" % i)] == "инвариантен" for i in range(1, 8)):
        bad.append("G31: известный ответ (Σ меняет знак, s1–s7 инвариантны при обращении порядка баров) не выполнен")
    return bad


def gate_bad(r: dict) -> list:
    bad, w, Ra = [], r["w"], r["Rall"]
    if Ra[RF.RHO_NAMES].isna().any().any() or len(Ra) != len(r["r0"]["S"]) or len(r["merge"]) != 6 * len(SG.SIGMA_NAMES):
        bad.append("G30: NaN в R, число дней не равно панели S или состав R_MERGE")
    for c, v in r["cfg"].items():
        a = v["a"]
        if len(a) != 2271 or not (np.isfinite(a["F"]) & (a["F"] > 0)).all() or float(np.abs(a["rv"].to_numpy() - w["rv"].to_numpy()).max()) != 0.0:
            bad.append("G32: %s не на 2271 общих датах, F ≤ 0 или цель изменена" % c)
        z = a["g"] == 0
        if z.any() and not (a.loc[z, "F"] == a.loc[z, "H"]).all():
            bad.append("G32: %s: g = 0, но F ≠ H побитно" % c)
    if ("E3" in r["cfg"]) != r["dec"]["compute"] or r["checkpoint"]["used"] not in (10, 11):
        bad.append("G33/G34: E3 посчитана не по правилу или израсходовано не 10–11 конфигураций")
    if r["info"]["window"].nunique() != 4 or set(r["info"]["pair"]) != {"%s vs %s" % p for p in ST.pairs("E3" in r["cfg"])}:
        bad.append("G32: состав E_INFO")
    cp1b, cp2 = CP0.read().get("1b", {}), CP0.read().get("2", {})
    for c, f in r["prev"].items():
        if S1.sha_arr(f) != {**cp1b.get("f_sha", {}), **cp2.get("f_sha", {})}.get(c):
            bad.append("G35: прогноз %s на диске ≠ SHA в CHECKPOINT 1b/2" % c)
    if r["sigma_sha"] != cp1b.get("sigma_sha"):
        bad.append("G35: SIGMA_features на диске ≠ SHA в CHECKPOINT 1b")
    return bad + tr_bad(r["tr"])


def check(r: dict) -> list:
    bad = gate_bad(r) + S1.check(r["r0"])
    for f, df in ST.parquets(r).items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for name, (df, _) in ST.tables(r).items():
        if not (IO.REP / name).exists() or not FC.frames_equal(IO.read_csv(name), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % name)
    cp = CP0.read().get("3")
    if cp is None or S1.stable(cp) != S1.stable(r["checkpoint"]):
        bad.append("G34: CHECKPOINT ключ 3 на диске ≠ пересчёту")
    if not RG.trials_ok(r["ids"]):      # лист 4 дописывает D1–D3 после E1–E3: проверяется префикс и состав, не равенство (правка листа 4)
        bad.append("G32/G34: ALL_TRIALS.csv ≠ строки %s" % r["ids"])
    return bad
