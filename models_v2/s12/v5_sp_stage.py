"""V5 лист 2 (Ж-226): стадия SP — сборка таблиц G20–G21 из кэша прогнозов, запись, сверка `--check` (только чтение) и строки EVIDENCE. Таблицы оценки — функции листа 1a (`v5_gs_eval*`, `v5_gs_enc`) без правок."""
from __future__ import annotations

import csv
import io
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_fin_check as FC, v4_trials as TR
from models_v2.s12 import v5_0_io as IO, v5_gs_enc as EN, v5_gs_eval as EV, v5_gs_eval2 as E2, v5_gs_panel as PA, v5_sp_eval as SE, v5_sp_run as RUN

SP = RUN.CFGS
TABLES = ["SP_COMPARISON", "SP_PER_TICKER", "SP_YEARS", "SP_TESTS", "SP_MCS", "SP_LOTO", "SP_SBER", "SP_STRESS", "SP_PARAMS", "SP_ENCOMPASS", "SP_ABLATION"]
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-226, лист 2 (Spectral Energy)"
VARIANT = {"SP1": "A: HAR по полосам, фиксированное разделение, R=clip(ΣÊ_b/Ê_tot, 2/3, 3/2)", "SP2": "B: общая поправка полос, θ 9, ридж κ=250", "SP3": "C: SP2 + межбумажный спектральный фактор, θ 12", "SP4": "SP2 + форма дня H3 V4 (lasth, herf, firsth), θ 18"}


def header(note: str) -> str:
    return "# %s; код v5.2 %s; входы %s; %s\n" % (STATUS, RUN.code_hash(), IO.inputs_hash(), note)


def write_csv(df: pd.DataFrame, name: str, note: str) -> None:
    with open(IO.REP / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")


def checkpoint() -> dict:
    return json.loads((IO.REP / "CHECKPOINT.json").read_text())


def write_checkpoint_sp(upd: dict) -> None:
    """Добавляет/обновляет ключ SP в CHECKPOINT.json; остальные ключи не трогает."""
    d = checkpoint()
    d["SP"] = {**d.get("SP", {}), **upd}
    (IO.REP / "CHECKPOINT.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))


def trial_rows(runs: dict) -> list:
    out = []
    for g, r in runs.items():
        cfg = {"cfg": g, "variant": VARIANT[g], "base": "K8=sqrt(H*sqrt(K2*K3))", "bands": "DCT-II low 3k<n, mid n<=3k<2n, high 3k>=2n", "kappa": 250 if g != "SP1" else None, "min_pairs": 250 if g == "SP1" else None, "min_days": 60 if g != "SP1" else None, "cache_key": r["key"]}
        b = io.StringIO()
        csv.writer(b, lineterminator="").writerow(["Ж-226:%s:%s" % (g, r["key"][:12]), "Ж-226", "harj_hybrid_v5", "spectral_energy_SP", json.dumps(cfg, ensure_ascii=False), "trial"])
        out.append(b.getvalue())
    return out


def tables(P, k: pd.DataFrame, runs: dict, thr: dict, cpd: dict) -> dict:
    kk = RUN.frame_with(k, runs)
    cols_full = EV.full_cols(SP)
    cc = S.common(kk, EV.BASE + SP).reset_index(drop=True)
    pt = EV.metrics(cc, EV.BASE + SP)
    cs = S.crit_panel(kk, cols_full).set_index("series")["crit_panel"]
    eqy = EV.eqy_series(kk, cols_full)
    go = EV.go_table(kk, SP, thr, eqy)
    best = min(SP, key=lambda g: cs[g])
    cmp_ = pd.concat([EV.comparison(cc, EV.BASE + SP, pt, cs, eqy), go], ignore_index=True)
    cmp_["decision_direction2"] = "go" if bool(go["go"].eq(True).any()) else "stop"
    cmp_["best_sp"] = best
    par = SE.params_table({g: runs[g]["params"] for g in SP})
    ab_cc = S.common(kk, ["K8", "H3", "SP2", "SP4"]).reset_index(drop=True)
    ca = SE.crit_all(kk, cols_full).set_index("series")
    ct = EV.crit_tk(kk, cols_full)
    return {"SP_COMPARISON": cmp_, "SP_PER_TICKER": EV.per_ticker(pt, SP), "SP_YEARS": EV.years(cc, EV.BASE + SP), "SP_TESTS": EV.tests(cc, SP, cpd["N_declared_before_stats"], cpd["N_wide"]), "SP_MCS": EV.ST.mcs(cc, EV.BASE + SP),
            "SP_LOTO": E2.loto_table(kk, {g: runs[g]["loto"] for g in SP}, runs, cols_full), "SP_SBER": EV.ST.sber(cc, EV.BASE + SP, EV.SBER_REF), "SP_STRESS": E2.stress(kk, cc, SP, best, cols_full), "SP_PARAMS": par,
            "SP_ENCOMPASS": EN.encompass(cc.assign(M=np.sqrt(cc["k2"] * cc["k3"])), best), "SP_ABLATION": SE.ablation(ab_cc, ca.reset_index(), eqy),
            "_best": best, "_contrib": E2.contributions(ct, best), "_crit": cs, "_go": go}


def compute(write: bool = False) -> dict:
    P = PA.build()
    k = IO.frame()
    cpd = checkpoint()
    F = RUN.inputs(write)
    runs = {g: RUN.get(P, g, write, F) for g in SP}
    return {"P": P, "runs": runs, "t": tables(P, k, runs, cpd["go_thresholds"], cpd), "cp": cpd}


def write(r: dict) -> None:
    for name in TABLES:
        write_csv(r["t"][name], name, "SP1–SP4, h = 1, 17 бумаг (M16 в выборе)")
    TR.replace_rows(("Ж-226:SP",), trial_rows(r["runs"]), IO.REP / "ALL_TRIALS.csv")
    go = r["t"]["_go"]
    write_checkpoint_sp({"used": 4, "best_sp": r["t"]["_best"], "direction2": "go" if bool(go["go"].eq(True).any()) else "stop", "cache_keys": {g: v["key"] for g, v in r["runs"].items()}, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def check(r: dict) -> list:
    """Сверка пересчёта в памяти с диском (только чтение)."""
    bad = []
    for name in TABLES:
        f = IO.REP / (name + ".csv")
        if not f.exists():
            bad.append("%s: нет файла" % name)
        elif not FC.frames_equal(IO.read_csv(name + ".csv"), r["t"][name], 1e-9):
            bad.append("%s: диск ≠ пересчёт" % name)
    d = checkpoint().get("SP", {})
    if d.get("used") != 4 or d.get("cache_keys") != {g: v["key"] for g, v in r["runs"].items()}:
        bad.append("CHECKPOINT[SP]: расход/ключи кэша не сходятся")
    tr = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    if sorted(tr[tr["key"].str.startswith("Ж-226:SP")]["config"].map(lambda s: json.loads(s)["cfg"])) != sorted(SP):
        bad.append("ALL_TRIALS: строки Ж-226:SP* не сходятся с SP1–SP4")
    if not all(v["cached"] for v in r["runs"].values()):
        bad.append("кэш прогнозов отсутствует или устарел (пересчёт в памяти)")
    return bad
