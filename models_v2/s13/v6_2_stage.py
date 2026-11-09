"""V6-2: стадия `2` = C1 (S + Q, LIN), C2 (S + Q, GBM), C3 (S + Q + u·(s1, s2, s3), LIN) и, только по правилу плана п. 2.2, C4 (то же, GBM). Движок листов 1a/1b (признаки S, обучатели, оценка, таблицы) не менялся: добавлены набор Q, индекс u (`v6_2_q`),
переоценка с u на дату (`v6_2_wf`) и строки конфигураций. A0 пересчитывается листом 1a; A1–A3 читаются с диска (прогнозы и SHA из CHECKPOINT 1b) и в стадии 2 только описательно сравниваются. Решение по C4 — автоматически из C_INFO."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s13 import (v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_feat as FE, v6_1_fit as FT, v6_1_reg as RG, v6_1_stage as S1, v6_1_wf as WF,
                           v6_1b_lev as LV, v6_1b_tab as TB, v6_2_q as QF, v6_2_unst as UN, v6_2_wf as W2)

CODE_VERSION = "v6.2"
LEARNER = {"C1": "LIN", "C2": "GBM", "C3": "LIN", "C4": "GBM"}
INTER = {"C1": False, "C2": False, "C3": True, "C4": True}
A_PREV = ("A1", "A2", "A3")


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листа 2 (Д5): все `v6_2_*.py`."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_2_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def runner(cid: str, S, Qd, shift: float = 0):
    """(tab, X) -> результат переоценки; X не используется (матрица строится по дате переоценки). shift — соседняя точка: LIN — шаг сетки λ, GBM — множитель числа деревьев."""
    fit = (lambda A, e: FT.fit_lin(A, e, lam_shift=int(shift))) if LEARNER[cid] == "LIN" else (lambda A, e: FT.fit_gbm(A, e, trees_mult=shift or 1.0))
    return lambda tab, X=None: W2.walk_forward_u(tab, S, Qd, fit, INTER[cid])


def variants(cid: str, S, Qd) -> dict:
    if LEARNER[cid] == "LIN":
        return {"λ − 1 шаг сетки": runner(cid, S, Qd, -1), "%s (λ выбран)" % cid: runner(cid, S, Qd, 0), "λ + 1 шаг сетки": runner(cid, S, Qd, 1)}
    return {"число деревьев ×0,5": runner(cid, S, Qd, 0.5), "%s (деревья по ранней остановке)" % cid: runner(cid, S, Qd, 1.0), "число деревьев ×2": runner(cid, S, Qd, 2.0)}


def one_config(cid: str, tab, S, Qd, w, cp0, f_a0) -> dict:
    run = runner(cid, S, Qd)(tab)
    f = WF.forecasts(tab, run["g"]).reindex(w.index).to_numpy()
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    qk = [Q.ql(rv[m], w[k].to_numpy()[m]) for k in ("K2", "K3")]
    h = tab["h"].reindex(w.index).to_numpy()
    a = pd.DataFrame({"d": w.index, "t_plus": tp, "rv": rv, "H": h, "g": run["g"].reindex(w.index).to_numpy(), "F": f})
    t = dict(metrics=EV.metrics_table(rv, {cid: f, **refs}, tp), go=EV.go_conditions(rv[m], f[m], tp[m], qk[0], qk[1], cp0), tests=S1.tests_all(w, f, cp0),
             stress=CV.stress_table(rv, f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, tp, h), sens=TB.sens_rows(tab, None, w, variants(cid, S, Qd), f, f_a0))
    return dict(cid=cid, run=run, f=f, a=a, t=t, names=W2.names(S, INTER[cid]))


def decide_c4(info: pd.DataFrame) -> dict:
    """Правило плана: C4 считается, если хотя бы одна из C1, C2, C3 несёт новую информацию против A0 (п. 2.2, цели 2017–2021). Автоматически."""
    s = info[(info["window"] == "2017_2021") & (info["y"] == "A0")].set_index("x")
    flags = {x: bool(s.loc[x, "new_information"]) for x in ("C1", "C2", "C3")}
    return dict(compute=bool(any(flags.values())), flags=flags, numbers={x: dict(delta=float(s.loc[x, "delta_qlike"]), p_dm_one=float(s.loc[x, "dm_p_one"]), years_neg=int(s.loc[x, "years_neg"])) for x in flags})


def prev_forecasts(w: pd.DataFrame) -> dict:
    """Прогнозы A1–A3 листа 1b с диска (на общих датах) — только описательное сравнение."""
    return {c: pd.read_parquet(IO.OUT / (c + ".parquet")).set_index("d")["F"].reindex(w.index).to_numpy() for c in A_PREV}


def pairs(with_c4: bool) -> list:
    p = [("C1", "A0"), ("C2", "A0"), ("C3", "A0"), ("C2", "C1"), ("C3", "C1")]
    return p + ([("C4", "C3"), ("C4", "A0")] if with_c4 else [])


def compute() -> dict:
    t0 = time.time()
    r0 = S1.compute()
    cp0, w = r0["cp0"], r0["w"]
    tab, S = r0["tab"].set_index("d"), r0["S"].set_index("d")
    bars, daily = FE.load_inputs()
    Qd = QF.features(bars, daily).reindex(S.index)
    f_a0, cfg = r0["a0"]["F"].to_numpy(), {}
    for cid in ("C1", "C2", "C3"):
        cfg[cid] = one_config(cid, tab, S, Qd, w, cp0, f_a0)
    fs = {"A0": f_a0, **{c: cfg[c]["f"] for c in cfg}}
    info = TB.info_table(fs, pairs(False), w, cp0)
    dec = decide_c4(info)
    if dec["compute"]:
        cfg["C4"] = one_config("C4", tab, S, Qd, w, cp0, f_a0)
        fs["C4"] = cfg["C4"]["f"]
        info = TB.info_table(fs, pairs(True), w, cp0)
    prev = prev_forecasts(w)
    grp = W2.walk_forward_u(tab, S, Qd, None)
    r = dict(r0=r0, cfg=cfg, fs=fs, info=info, dec=dec, w=w, tab=tab, ids=["A0", *A_PREV, *cfg], cids=list(cfg), prev=prev, Qd=Qd.reset_index(names="d"), grp=grp)
    r["vs_a"] = TB.info_table({**fs, **prev}, [(c, a) for c in cfg for a in ("A1", "A2")], w, cp0)
    r["unstable"] = UN.unstable_table(w, grp["high"].reindex(w.index).to_numpy(), grp["thr"].reindex(w.index).notna().to_numpy(), fs, list(cfg))
    r["groups"] = pd.DataFrame({"d": w.index, "u": grp["u"].reindex(w.index).to_numpy(), "thr": grp["thr"].reindex(w.index).to_numpy(), "high": grp["high"].reindex(w.index).to_numpy()})
    r["levels"] = LV.levels_table([row for c in ["A0", *cfg] for row in LV.level_rows(c, *level_inputs(r, c))])
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
    cfg, w, info, cp0 = r["cfg"], r["w"], r["info"], r["r0"]["cp0"]
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    allf = {**r["fs"], **r["prev"]}
    qs = {c: dict(all=float(Q.ql(w["rv"].to_numpy(), allf[c]).mean()), y2017_2021=float(Q.ql(w["rv"].to_numpy()[m], allf[c][m]).mean())) for c in r["ids"]}
    cp = dict(program="HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 2", configs=r["ids"], used=len(r["ids"]), budget=int(cp0["budget_V6"]), N=cp0["N"], N_wide=cp0["N_wide"], inputs_hash=IO.inputs_hash(), code_hash_v6_2=code_hash(),
              code_hash_v6_1=RG.code_hash(), code_hash_v6_0=IO.code_hash(), code_version=CODE_VERSION, qlike=qs, f_sha={c: S1.sha_arr(cfg[c]["f"]) for c in cfg}, f_sha_prev={c: S1.sha_arr(r["prev"][c]) for c in r["prev"]},
              q_sha=S1.sha_arr(r["Qd"].drop(columns="d")), n_dates=len(w), u_groups={"n_high": int(r["groups"]["high"].sum()), "n_valid": int(r["groups"]["thr"].notna().sum())},
              new_information=info[info["window"] == "2017_2021"][["pair", "delta_qlike", "dm_p_one", "years_neg", "new_information"]].to_dict("records"),
              go={c: [dict(cond=x["cond"], value=x["value"], passed=x["passed"]) for x in cfg[c]["t"]["go"].to_dict("records")] for c in cfg},
              coef_stability={c: TB.coef_stability(r["coef"][c], [n for n in cfg[c]["names"] if n not in r["r0"]["S"].columns[1:]]) for c in cfg if LEARNER[c] == "LIN"}, note="лист 2: потолок, не OOS")
    cp["C4"] = "посчитана по правилу 2.2" if r["dec"]["compute"] else {"решение": "не считается, правило 2.2 не выполнено", **{x: r["dec"]["numbers"][x] for x in ("C1", "C2", "C3")}}
    cp["written_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return cp


def tables(r: dict) -> dict:
    """Имя файла -> (датафрейм, примечание) для CSV в docs/otchety/harj_hybrid_v6/."""
    out = {"C_INFO.csv": (r["info"], "G25/G27: правило «новая информация» плана п. 2.2 (цели 2017–2021); окна 2022–2025 и 2026 показаны, в правиле не участвуют"),
           "C_VS_A.csv": (r["vs_a"], "G28: описательно — C против A1 и A2 (Δ в % уровня K1, DM); в правило 2.2 не входит"),
           "C_UNSTABLE.csv": (r["unstable"], "G26: главная проверка — дни с u в верхнем квинтиле (порог по закрытым парам на дату переоценки) против остальных"),
           "C_LEVELS.csv": (r["levels"], "G28: уровни 1–5 этапа 21 блока")}
    for c, v in r["cfg"].items():
        for k, nm in (("metrics", "METRICS"), ("go", "GO"), ("tests", "TESTS"), ("stress", "STRESS"), ("sens", "SENS")):
            out["%s_%s.csv" % (c, nm)] = (v["t"][k], "G25: %s %s (S + Q%s, %s)" % (c, nm, " + u·(s1, s2, s3)" if INTER[c] else "", LEARNER[c]))
    out.update({"%s_COEF.csv" % c: (d, "G25: θ LIN по датам переоценки, единицы z") for c, d in r["coef"].items()})
    out.update({"%s_IMP.csv" % c: (d, "G25: важность признаков GBM (gain, splits)") for c, d in r["imp"].items()})
    return out


def parquets(r: dict) -> dict:
    return {**{"%s.parquet" % c: v["a"] for c, v in r["cfg"].items()}, "Q_features.parquet": r["Qd"], "U_groups.parquet": r["groups"]}
