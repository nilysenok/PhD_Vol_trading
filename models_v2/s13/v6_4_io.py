"""V6-4: запись (`write`, не в `--check`) и проверка (`check`, только чтение) выходов стадии 4: прогнозы D1–D3 и контроль K9 в `data/regen/harj_hybrid_v6/`, таблицы и ключ `4` в `docs/otchety/harj_hybrid_v6/`, строки `ALL_TRIALS.csv` (D1–D3; K9 — не строка).
Проверка сверяет диск с пересчётом в памяти (допуск 1e-9), реестр, SHA предыдущих прогнозов и S (CHECKPOINT 1a/1b/2/3), размер пула каждого дня (независимый пересчёт), сжатие D3 между D2 и K9."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_4_stage as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 4 (SBER, h = 1; Historical Analog: D1, D2, D3 против контроля K9)"


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
    cp, old = CP0.read(), CP0.read().get("4")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and S1.stable(old) == S1.stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "4": entry}, ensure_ascii=False, indent=1))
    CP0.ALL_TRIALS.write_text(RG.trial_text(r["ids"]), encoding="utf-8")


def pool_bad(r: dict) -> list:
    """G36: размер пула дня d = число пар tab с t⁺ ≤ d (независимо от `closed_rows`); пул не меньше 100 (k) и первая пара 2015-07-07."""
    tp, bad = r["tab"]["t_plus"].to_numpy("datetime64[ns]"), []
    n = np.array([(tp <= np.datetime64(d)).sum() for d in r["w"].index])
    if not (n == r["st"]["n_pool"].to_numpy()).all() or n.min() < 100 or str(r["tab"].index[0].date()) != "2015-07-07":
        bad.append("G36: размер пула дня d ≠ числу закрытых пар или пул мал")
    return bad


def gate_bad(r: dict) -> list:
    bad, w, st, cp = pool_bad(r), r["w"], r["st"], r["cp"]
    for c, v in r["cfg"].items():
        a = v["a"]
        if len(a) != 2271 or not (np.isfinite(a["F"]) & (a["F"] > 0)).all() or float(np.abs(a["rv"].to_numpy() - w["rv"].to_numpy()).max()) != 0.0:
            bad.append("G37: %s не на 2271 общих датах, F ≤ 0 или цель изменена" % c)
        z = a["g"] == 0
        if z.any() and not (a.loc[z, "F"] == a.loc[z, "H"]).all():
            bad.append("G37: %s: g = 0, но F ≠ H побитно" % c)
    if not (np.isfinite(r["f_k9"]) & (r["f_k9"] > 0)).all() or "K9" in r["ids"]:
        bad.append("G37: K9 не конечен/не положителен или попал в реестр")
    lo, hi = np.minimum(st["m2_100"], st["m_glob"]) - 1e-12, np.maximum(st["m2_100"], st["m_glob"]) + 1e-12
    if not ((st["m3_50"] >= lo) & (st["m3_50"] <= hi)).all():
        bad.append("G37: M_D3 вне отрезка [M_D2, M_glob]")
    if r["checkpoint"]["used"] != 14 or len(r["ids"]) != 14:
        bad.append("G39: израсходовано не 14 конфигураций")
    exp = {"%s vs K9" % c for c in ST.CIDS} | {"K9 vs K1"} | {"%s vs %s" % (c, y) for c in ST.CIDS for y in (*ST.DESC, r["lin"])}
    if r["info"]["window"].nunique() != 4 or set(r["info"]["pair"]) != exp:
        bad.append("G38: состав D_INFO")
    if len(r["reg"]) != 10 or not r["reg"]["verdict"].iloc[-1]:
        bad.append("G38: состав или вывод D_REGIME")
    sha = ST.expected_sha(cp)
    bad += ["G37: прогноз %s на диске ≠ SHA в CHECKPOINT 1a/1b/2/3" % c for c, f in r["prev"].items() if S1.sha_arr(f) != sha[c]]
    if S1.sha_arr(r["S"].reset_index(drop=True)) != cp["1a"]["s_features_sha"]:
        bad.append("G36: S_features на диске ≠ SHA в CHECKPOINT 1a")
    t = r["tab"].reindex(w.index)
    if float(np.abs(t["rv_next"] - w["rv"]).max()) != 0.0 or float(np.abs(t["h"] - w["K1"]).max()) != 0.0:
        bad.append("G36: таблица H ≠ общим датам листа 0")
    return bad


def check(r: dict) -> list:
    bad = gate_bad(r)
    for f, df in ST.parquets(r).items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for name, (df, _) in ST.tables(r).items():
        if not (IO.REP / name).exists() or not FC.frames_equal(IO.read_csv(name), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёту" % name)
    cp = CP0.read().get("4")
    if cp is None or S1.stable(cp) != S1.stable(r["checkpoint"]):
        bad.append("G39: CHECKPOINT ключ 4 на диске ≠ пересчёту")
    if not RG.trials_ok(r["ids"]):      # лист 5 дописывает CB1, F1, F2 после D1–D3: проверяется префикс и состав, не равенство (правка листа 5)
        bad.append("G37/G39: ALL_TRIALS.csv ≠ строки %s" % r["ids"])
    return bad
