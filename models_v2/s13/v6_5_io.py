"""V6-5: запись (`write`, не в `--check`) и проверка (`check`, только чтение) выходов стадии 5: прогнозы CB1, F1, F2 в `data/regen/harj_hybrid_v6/`, таблицы и ключ `5` в `docs/otchety/harj_hybrid_v6/`, строки `ALL_TRIALS.csv` (CB1, F1, F2).
Проверка сверяет диск с пересчётом в памяти (допуск 1e-9), реестр, SHA прежних прогнозов, Σ, Q и S (CHECKPOINT 1a…4), состав таблиц, число сред в каждой переоценке (независимый пересчёт по таблице H)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_5_stage as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 5 (SBER, h = 1; сочетание CB1 и Distributionally Robust F1, F2)"


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
    cp, old = CP0.read(), CP0.read().get("5")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and S1.stable(old) == S1.stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "5": entry}, ensure_ascii=False, indent=1))
    CP0.ALL_TRIALS.write_text(RG.trial_text(r["ids"]), encoding="utf-8")


def env_bad(r: dict) -> list:
    """Число сред F1/F2 в каждой переоценке: 1 ≤ n_env ≤ числа календарных лет t⁺ закрытых пар на d0 (независимо от `closed_rows`)."""
    tp = r["tab"]["t_plus"]
    bad = []
    for c in ("F1", "F2"):
        for f in r["cfg"][c]["run"]["fits"]:
            yrs = tp[tp.to_numpy("datetime64[ns]") <= np.datetime64(f["d0"])].dt.year.nunique()
            if not 1 <= f["model"].info["n_env"] <= yrs or f["n_train"] != int((tp.to_numpy("datetime64[ns]") <= np.datetime64(f["d0"])).sum()):
                bad.append("G41: %s: среды на %s вне [1; %d] или n_train не равно числу закрытых пар" % (c, f["d0"].date(), yrs))
    return bad


def gate_bad(r: dict) -> list:
    bad, w, cp = env_bad(r), r["w"], r["cp"]
    for c, v in r["cfg"].items():
        a = v["a"]
        if len(a) != 2271 or not (np.isfinite(a["F"]) & (a["F"] > 0)).all() or float(np.abs(a["rv"].to_numpy() - w["rv"].to_numpy()).max()) != 0.0:
            bad.append("G42: %s не на 2271 общих датах, F ≤ 0 или цель изменена" % c)
        z = a["g"] == 0
        if z.any() and not (a.loc[z, "F"] == a.loc[z, "H"]).all():
            bad.append("G42: %s: g = 0, но F ≠ H побитно" % c)
    for c, t in r["conv"].items():
        if not np.isfinite(t[["grad_norm", "L"]].to_numpy(float)).all() or len(t) != len(r["cfg"][c]["run"]["fits"]):
            bad.append("G42: сходимость %s: NaN или не по всем переоценкам" % c)
    if r["checkpoint"]["used"] != 17 or len(r["ids"]) != 17:
        bad.append("G44: израсходовано не 17 конфигураций")
    exp = {"%s vs %s" % p for p in ST.RULE + ST.DESC}
    if r["info"]["window"].nunique() != 4 or set(r["info"]["pair"]) != exp:
        bad.append("G43: состав F_INFO")
    ys = r["years"]
    if len(ys[ys["section"] == "итог"]) != 4 or set(ys["pair"]) != {"F1 vs CB1", "F2 vs CB1"}:
        bad.append("G43: состав F_YEARS")
    if len(r["levels"]) != 15:
        bad.append("G44: состав F_LEVELS")
    sha = ST.expected_sha(cp)
    bad += ["G42: прогноз %s на диске ≠ SHA в CHECKPOINT 1a…4" % c for c, f in r["prev"].items() if S1.sha_arr(f) != sha[c]]
    for key, got, nm in ((cp["1a"]["s_features_sha"], S1.sha_arr(r["S"].reset_index(drop=True)), "S_features"), (cp["1b"]["sigma_sha"], r["sigma_sha"], "SIGMA_features"), (cp["2"]["q_sha"], r["q_sha"], "Q")):
        if key != got:
            bad.append("G42: %s ≠ SHA в CHECKPOINT" % nm)
    t = r["tab"].reindex(w.index)
    if float(np.abs(t["rv_next"] - w["rv"]).max()) != 0.0 or float(np.abs(t["h"] - w["K1"]).max()) != 0.0:
        bad.append("G42: таблица H ≠ общим датам листа 0")
    return bad


def check(r: dict) -> list:
    bad = gate_bad(r)
    for f, df in ST.parquets(r).items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for name, (df, _) in ST.tables(r).items():
        if not (IO.REP / name).exists() or not FC.frames_equal(IO.read_csv(name), df, 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёту" % name)
    cp = CP0.read().get("5")
    if cp is None or S1.stable(cp) != S1.stable(r["checkpoint"]):
        bad.append("G44: CHECKPOINT ключ 5 на диске ≠ пересчёту")
    if not RG.trials_ok(r["ids"]):      # лист 6 дописывает A0g после CB1, F1, F2: проверяется префикс и состав, не равенство (правка листа 6)
        bad.append("G42/G44: ALL_TRIALS.csv ≠ строки %s" % r["ids"])
    return bad
