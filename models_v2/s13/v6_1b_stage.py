"""V6-1b: стадия `1b` = A1 (S + Σ, LIN), A2 (S + Σ, GBM) и, только по правилу плана п. 2.2, A3 (S + Σ + Σ³, LIN). Движок листа 1a (признаки S, обучатели, переоценка, оценка) не менялся: добавлены наборы признаков Σ, Σ³ (`v6_1b_sig`) и строки конфигураций.
`compute` — чистая функция от входов (A0 берётся из листа 1a), `write` — запись (не в `--check`), `check` — только чтение. Решение по A3 — автоматически из A_INFO (A1 или A2 против A0)."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from models_v2.s13 import (v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_feat as FE, v6_1_fit as FT, v6_1_reg as RG, v6_1_stage as S1, v6_1_wf as WF,
                           v6_1b_lev as LV, v6_1b_sig as SG, v6_1b_tab as TB)

CODE_VERSION = "v6.1b"
LEARNER = {"A1": "LIN", "A2": "GBM", "A3": "LIN"}
EXTRA = {"A1": ("sigma",), "A2": ("sigma",), "A3": ("sigma", "sig3")}


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листа 1b (Д5): все `v6_1b_*.py` (диагностика DX-A включена: её таблица входит в выходы листа)."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_1b_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def runner(cid: str, shift: float = 0):
    """Функция (tab, X) -> результат переоценки для конфигурации; shift — соседняя точка: LIN — шаг сетки λ (±1), GBM — множитель числа деревьев (0,5 / 2)."""
    if LEARNER[cid] == "LIN":
        return lambda tab, X: WF.walk_forward(tab, X, lambda A, e: FT.fit_lin(A, e, lam_shift=int(shift)))
    return lambda tab, X: WF.walk_forward(tab, X, lambda A, e: FT.fit_gbm(A, e, trees_mult=shift or 1.0))


def variants(cid: str) -> dict:
    if LEARNER[cid] == "LIN":
        return {"λ − 1 шаг сетки": runner(cid, -1), "%s (λ выбран)" % cid: runner(cid, 0), "λ + 1 шаг сетки": runner(cid, 1)}
    return {"число деревьев ×0,5": runner(cid, 0.5), "%s (деревья по ранней остановке)" % cid: runner(cid, 1.0), "число деревьев ×2": runner(cid, 2.0)}


def design(cid: str, S: pd.DataFrame, parts: dict) -> pd.DataFrame:
    return pd.concat([S] + [parts[k].reindex(S.index) for k in EXTRA[cid]], axis=1)


def one_config(cid: str, tab, S, parts, w, cp0, f_a0) -> dict:
    X = design(cid, S, parts)
    run = runner(cid)(tab, X)
    f = WF.forecasts(tab, run["g"]).reindex(w.index).to_numpy()
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    qk = [Q.ql(rv[m], w[k].to_numpy()[m]) for k in ("K2", "K3")]
    a = pd.DataFrame({"d": w.index, "t_plus": tp, "rv": rv, "H": tab["h"].reindex(w.index).to_numpy(), "g": run["g"].reindex(w.index).to_numpy(), "F": f})
    t = dict(metrics=EV.metrics_table(rv, {cid: f, **refs}, tp), go=EV.go_conditions(rv[m], f[m], tp[m], qk[0], qk[1], cp0), tests=S1.tests_all(w, f, cp0),
             stress=CV.stress_table(rv, f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, tp, tab["h"].reindex(w.index).to_numpy()), sens=TB.sens_rows(tab, X, w, variants(cid), f, f_a0))
    return dict(cid=cid, X=X, run=run, f=f, a=a, t=t, names=list(X.columns))


def decide_a3(info: pd.DataFrame) -> dict:
    """Правило плана: A3 считается, если A1 или A2 несёт новую информацию против A0 (п. 2.2, цели 2017–2021). Автоматически, без ручного выбора."""
    s = info[(info["window"] == "2017_2021") & (info["y"] == "A0")].set_index("x")
    flags = {x: bool(s.loc[x, "new_information"]) for x in ("A1", "A2")}
    return dict(compute=bool(any(flags.values())), flags=flags, numbers={x: dict(delta=float(s.loc[x, "delta_qlike"]), p_dm_one=float(s.loc[x, "dm_p_one"]), years_neg=int(s.loc[x, "years_neg"])) for x in flags})


def compute() -> dict:
    t0 = time.time()
    r0 = S1.compute()
    cp0, w = r0["cp0"], r0["w"]
    tab, S = r0["tab"].set_index("d"), r0["S"].set_index("d")
    bars, daily = FE.load_inputs()
    sigma, sig3, flags = SG.features(bars, daily)
    parts = {"sigma": sigma, "sig3": sig3}
    f_a0, cfg = r0["a0"]["F"].to_numpy(), {}
    for cid in ("A1", "A2"):
        cfg[cid] = one_config(cid, tab, S, parts, w, cp0, f_a0)
    fs = {"A0": f_a0, **{c: cfg[c]["f"] for c in cfg}}
    info = TB.info_table(fs, [("A1", "A0"), ("A2", "A0"), ("A2", "A1")], w, cp0)
    dec = decide_a3(info)
    if dec["compute"]:
        cfg["A3"] = one_config("A3", tab, S, parts, w, cp0, f_a0)
        fs["A3"] = cfg["A3"]["f"]
        info = TB.info_table(fs, [("A1", "A0"), ("A2", "A0"), ("A2", "A1"), ("A3", "A1")], w, cp0)
    r = dict(r0=r0, cfg=cfg, fs=fs, info=info, dec=dec, w=w, tab=tab, ids=["A0"] + list(cfg), vs=TB.vs_sp_table(fs, list(cfg), TB.sp_series(w.index), w), flags=flags.reindex(S.index),
             sigfe=pd.concat([sigma, sig3, flags], axis=1).loc[S.index].reset_index(names="d"))
    r["levels"] = LV.levels_table([row for c in r["ids"] for row in LV.level_rows(c, *level_inputs(r, c))])
    r["coef"] = {c: TB.coef_table(cfg[c]["run"], cfg[c]["names"]) for c in cfg if LEARNER[c] == "LIN"}
    r["imp"] = {c: TB.imp_table(cfg[c]["run"], cfg[c]["names"]) for c in cfg if LEARNER[c] == "GBM"}
    r["checkpoint"] = checkpoint(r)
    r["seconds"] = time.time() - t0
    return r


def level_inputs(r: dict, cid: str) -> tuple:
    if cid == "A0":
        r0 = r["r0"]
        return r0["a0"], r["w"], dict(tests=r0["tests"], stress=r0["stress"], sens=r0["sens"])
    c = r["cfg"][cid]
    return c["a"], r["w"], c["t"]


def checkpoint(r: dict) -> dict:
    cfg, w, info = r["cfg"], r["w"], r["info"]
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    qs = {c: dict(all=float(Q.ql(w["rv"].to_numpy(), r["fs"][c]).mean()), y2017_2021=float(Q.ql(w["rv"].to_numpy()[m], r["fs"][c][m]).mean())) for c in r["ids"]}
    fl = r["flags"]
    cp = dict(program="HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 1b", configs=r["ids"], used=len(r["ids"]), budget=int(r["r0"]["cp0"]["budget_V6"]), N=r["r0"]["cp0"]["N"], N_wide=r["r0"]["cp0"]["N_wide"],
              inputs_hash=IO.inputs_hash(), code_hash_v6_1b=code_hash(), code_hash_v6_1=RG.code_hash(), code_hash_v6_0=IO.code_hash(), code_version=CODE_VERSION, qlike=qs,
              f_sha={c: S1.sha_arr(cfg[c]["f"]) for c in cfg}, sigma_sha=S1.sha_arr(r["sigfe"].drop(columns="d")), n_dates=len(w),
              flags_days={"rs_neg_zero (канал N ≡ 0)": int(fl["flag_rs0"].sum()), "vol_zero (канал Q ≡ 0)": int(fl["flag_vol0"].sum()), "days": len(fl)},
              new_information=info[info["window"] == "2017_2021"][["pair", "delta_qlike", "dm_p_one", "years_neg", "new_information"]].to_dict("records"),
              go={c: [dict(cond=x["cond"], value=x["value"], passed=x["passed"]) for x in cfg[c]["t"]["go"].to_dict("records")] for c in cfg},
              coef_stability={c: TB.coef_stability(r["coef"][c], [n for n in cfg[c]["names"] if n in SG.SIGMA_NAMES + SG.sigma3_names()]) for c in cfg if LEARNER[c] == "LIN"}, note="лист 1b: потолок, не OOS")
    cp["A3"] = "посчитана по правилу 2.2" if r["dec"]["compute"] else {"решение": "не считается, правило 2.2 не выполнено", "A1": r["dec"]["numbers"]["A1"], "A2": r["dec"]["numbers"]["A2"]}
    cp["written_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return cp


def tables(r: dict) -> dict:
    """Имя файла -> (датафрейм, примечание) для CSV в docs/otchety/harj_hybrid_v6/."""
    out = {"A_INFO.csv": (r["info"], "G18/G21: правило «новая информация» плана п. 2.2 (цели 2017–2021); окна 2022–2025 и 2026 показаны, в правиле не участвуют"),
           "A_VS_SP.csv": (r["vs"], "G20: A против V5 SP1–SP4 SBER; охват SP^(1−w)·A^w; абляция «+ спектр», контроли"),
           "A_LEVELS.csv": (r["levels"], "G22: уровни 1–5 этапа 21 блока")}
    for c, v in r["cfg"].items():
        for k, nm in (("metrics", "METRICS"), ("go", "GO"), ("tests", "TESTS"), ("stress", "STRESS"), ("sens", "SENS")):
            out["%s_%s.csv" % (c, nm)] = (v["t"][k], "G18: %s %s (S + Σ%s, %s)" % (c, nm, " + Σ³" if c == "A3" else "", LEARNER[c]))
    out.update({"%s_COEF.csv" % c: (d, "G18: θ LIN по датам переоценки, единицы z") for c, d in r["coef"].items()})
    out.update({"%s_IMP.csv" % c: (d, "G18: важность признаков GBM (gain, splits)") for c, d in r["imp"].items()})
    return out


def parquets(r: dict) -> dict:
    return {**{"%s.parquet" % c: v["a"] for c, v in r["cfg"].items()}, "SIGMA_features.parquet": r["sigfe"]}
