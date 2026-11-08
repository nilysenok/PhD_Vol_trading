"""V5 лист 1a (Ж-226): стадия GS — сборка таблиц G12–G14 из кэша прогнозов, запись, сверка `--check` (только чтение) и строки EVIDENCE."""
from __future__ import annotations

import csv
import io
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_fin_check as FC, v4_trials as TR
from models_v2.s12 import v5_0_io as IO, v5_gs_enc as EN, v5_gs_eval as EV, v5_gs_eval2 as E2, v5_gs_panel as PA, v5_gs_run as RUN

GS = EV.GS
TABLES = ["GS_COMPARISON", "GS_PER_TICKER", "GS_YEARS", "GS_TESTS", "GS_MCS", "GS_LOTO", "GS_SBER", "GS_STRESS", "GS_ROBUST", "GS_PARAMS", "GS_W", "GS_ENCOMPASS", "GS_CALIB"]
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-226, лист 1a (Graph-Score)"


def header(note: str) -> str:
    return "# %s; код v5.1a %s; входы %s; %s\n" % (STATUS, RUN.code_hash(), IO.inputs_hash(), note)


def write_csv(df: pd.DataFrame, name: str, note: str) -> None:
    with open(IO.REP / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")


def checkpoint() -> dict:
    return json.loads(RUN.CP.read_text())


def write_checkpoint_gs(upd: dict) -> None:
    """Добавляет/обновляет ключ GS в CHECKPOINT.json; остальные ключи не трогает; форма и её время не переписываются иным значением."""
    d = checkpoint()
    old = d.get("GS", {})
    if "form" in old and "form" in upd and old["form"] != upd["form"]:
        raise RuntimeError("форма score уже записана: %s" % old["form"])
    d["GS"] = {**old, **{k: v for k, v in upd.items() if not (k in ("form", "form_written_at") and k in old)}}
    RUN.CP.write_text(json.dumps(d, ensure_ascii=False, indent=1))


def decide(P, k: pd.DataFrame, write: bool) -> str:
    g1, g2 = RUN.get(P, "GS1", write), RUN.get(P, "GS2", write)
    form, cp = RUN.decide_form(k, g1["main"], g2["main"])
    if write:
        write_checkpoint_gs({"form": form, "form_crit_panel": cp, "form_rule": "меньший crit_panel из GS1 (сырой s) и GS2 (min(s,3)); записано до счёта GS3–GS5", "form_written_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return form


def load_runs(P, k: pd.DataFrame, write: bool) -> dict:
    decide(P, k, write)
    return {g: RUN.get(P, g, write) for g in GS}


def trial_rows(runs: dict) -> list:
    out = []
    for g, r in runs.items():
        cfg = {"cfg": g, "variant": {"GS1": "A raw", "GS2": "A min(s,3)", "GS3": "B market", "GS4": "C sparse top-3", "GS5": "D low-rank"}[g], "clip_s": bool(r["clip"]), "base": "K8=sqrt(H*sqrt(K2*K3))", "kappa": 250, "theta0": [0.9, 0, 0],
               "bounds": {"rho": [0, 0.99], "alpha": [0, 0.3], "beta": [0, 0.3]}, "cache_key": r["key"]}
        b = io.StringIO()
        csv.writer(b, lineterminator="").writerow(["Ж-226:%s:%s" % (g, r["key"][:12]), "Ж-226", "harj_hybrid_v5", "graph_score_GS", json.dumps(cfg, ensure_ascii=False), "trial"])
        out.append(b.getvalue())
    return out


def tables(P, k: pd.DataFrame, runs: dict, thr: dict, cpd: dict) -> dict:
    kk = RUN.frame_with(k, runs)
    cols_full = EV.full_cols(GS)
    cc = S.common(kk, EV.BASE + GS).reset_index(drop=True)
    pt = EV.metrics(cc, EV.BASE + GS)
    cs = S.crit_panel(kk, cols_full).set_index("series")["crit_panel"]
    eqy = EV.eqy_series(kk, cols_full)
    go = EV.go_table(kk, GS, thr, eqy)
    best = min(GS, key=lambda g: cs[g])
    cmp_ = pd.concat([EV.comparison(cc, EV.BASE + GS, pt, cs, eqy), go], ignore_index=True)
    cmp_["decision_direction1"] = "go" if bool(go["go"].eq(True).any()) else "stop"
    cmp_["best_gs"] = best
    loto = E2.loto_table(kk, {g: runs[g]["loto"] for g in GS}, runs, cols_full)
    par, w = E2.params_tables({g: runs[g]["params"] for g in GS})
    Fb = {"K8": P.k8, "M": P.m}
    ub = runs[best]["main"]
    mat = np.full(P.rv.shape, np.nan)
    pos = {d: i for i, d in enumerate(P.axis)}
    jx = {t: i for i, t in enumerate(P.tick)}
    mat[ub["t_plus"].map(pos).to_numpy(), ub["ticker"].map(jx).to_numpy()] = ub["F"].to_numpy()
    ct = EV.crit_tk(kk, cols_full)
    return {"GS_COMPARISON": cmp_, "GS_PER_TICKER": EV.per_ticker(pt, GS), "GS_YEARS": EV.years(cc, EV.BASE + GS), "GS_TESTS": EV.tests(cc, GS, cpd["N_declared_before_stats"], cpd["N_wide"]), "GS_MCS": EV.ST.mcs(cc, EV.BASE + GS),
            "GS_LOTO": loto, "GS_SBER": EV.ST.sber(cc, EV.BASE + GS, EV.SBER_REF), "GS_STRESS": E2.stress(kk, cc, GS, best, cols_full), "GS_ROBUST": E2.robust(cc), "GS_PARAMS": par, "GS_W": w,
            "GS_ENCOMPASS": EN.encompass(cc.assign(M=np.sqrt(cc["k2"] * cc["k3"])), best),
            "GS_CALIB": EN.calib(P, {"K8": Fb["K8"], best: mat}), "_best": best, "_contrib": E2.contributions(ct, best), "_crit": cs, "_go": go}


def compute(write: bool = False) -> dict:
    P = PA.build()
    k = IO.frame()
    cpd = checkpoint()
    runs = load_runs(P, k, write)
    t = tables(P, k, runs, cpd["go_thresholds"], cpd)
    return {"P": P, "runs": runs, "t": t, "cp": cpd}


def write(r: dict) -> None:
    for name in TABLES:
        write_csv(r["t"][name], name, "GS1–GS5, h = 1, 17 бумаг (M16 в выборе)")
    TR.replace_rows(("Ж-226:GS",), trial_rows(r["runs"]), IO.REP / "ALL_TRIALS.csv")
    best, go = r["t"]["_best"], r["t"]["_go"]
    write_checkpoint_gs({"used": 5, "best_gs": best, "direction1": "go" if bool(go["go"].eq(True).any()) else "stop", "cache_keys": {g: v["key"] for g, v in r["runs"].items()}, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def check(r: dict) -> list:
    """Сверка пересчёта в памяти с диском (только чтение)."""
    bad = []
    for name in TABLES:
        f = IO.REP / (name + ".csv")
        if not f.exists():
            bad.append("%s: нет файла" % name)
        elif not FC.frames_equal(IO.read_csv(name + ".csv"), r["t"][name], 1e-9):
            bad.append("%s: диск ≠ пересчёт" % name)
    d = checkpoint().get("GS", {})
    if d.get("form") is None or d.get("used") != 5 or d.get("cache_keys") != {g: v["key"] for g, v in r["runs"].items()}:
        bad.append("CHECKPOINT[GS]: форма/расход/ключи кэша не сходятся")
    tr = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    if sorted(tr[tr["key"].str.startswith("Ж-226:GS")]["config"].map(lambda s: json.loads(s)["cfg"])) != sorted(GS):
        bad.append("ALL_TRIALS: строки Ж-226:GS* не сходятся с GS1–GS5")
    if not all(v["cached"] for v in r["runs"].values()):
        bad.append("кэш прогнозов отсутствует или устарел (пересчёт в памяти)")
    return bad
