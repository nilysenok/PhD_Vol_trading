"""V6-5: стадия `5` = CB1 (S + Σ + Q, LIN), F1 и F2 (тот же набор, цель с разбросом по средам-годам, `v6_5_fit`); все три считаются всегда (план V6, п. 7). Главный вывод F — разброс годовых Δ к CB1 и число лет с Δ < 0 (`v6_5_yr`).
Движок листов 1a–4 (признаки S, обучатель LIN, переоценка, оценка, таблицы) не менялся: добавлены объединение наборов, робастные цели и строки конфигураций. Прогнозы A0…D3 и признаки Σ читаются с диска (SHA = CHECKPOINT 1a…4)."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import time
from pathlib import Path

import pandas as pd

from models_v2.s13 import (v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_feat as FE, v6_1_fit as FT, v6_1_reg as RG, v6_1_stage as S1, v6_1_wf as WF, v6_1b_lev as LV,
                           v6_1b_sig as SG, v6_1b_tab as TB, v6_2_q as QF, v6_4_stage as S4, v6_5_fit as FR, v6_5_wf as W5, v6_5_yr as YR)

CODE_VERSION = "v6.5"
CIDS = ("CB1", "F1", "F2")
RULE = [("CB1", "A0"), ("F1", "CB1"), ("F2", "CB1")]
DESC = [("CB1", "A1"), ("CB1", "C1")]


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листа 5 (Д5): все `v6_5_*.py`."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_5_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def runner(cid: str, shift: float = 0):
    """(tab, X) -> результат переоценки; shift — соседняя точка: шаг сетки λ (±1). CB1 — LIN движка, F1/F2 — робастные цели."""
    if cid == "CB1":
        return lambda tab, X: WF.walk_forward(tab, X, lambda A, e: FT.fit_lin(A, e, lam_shift=int(shift)))
    return lambda tab, X: W5.walk_forward_env(tab, X, lambda A, e, tp: FR.fit_robust(A, e, tp, cid, lam_shift=int(shift)))


def variants(cid: str) -> dict:
    return {"λ − 1 шаг сетки": runner(cid, -1), "%s (λ выбран)" % cid: runner(cid, 0), "λ + 1 шаг сетки": runner(cid, 1)}


def expected_sha(cp: dict) -> dict:
    return {"A0": cp["1a"]["a0_f_sha"], **cp["1b"]["f_sha"], **cp["2"]["f_sha"], **cp["3"]["f_sha"], **cp["4"]["f_sha"]}


def one_config(cid: str, tab, X, w, cp0, f_a0) -> dict:
    run = runner(cid)(tab, X)
    f = WF.forecasts(tab, run["g"]).reindex(w.index).to_numpy()
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    qk = [Q.ql(rv[m], w[k].to_numpy()[m]) for k in ("K2", "K3")]
    h = tab["h"].reindex(w.index).to_numpy()
    a = pd.DataFrame({"d": w.index, "t_plus": tp, "rv": rv, "H": h, "g": run["g"].reindex(w.index).to_numpy(), "F": f})
    t = dict(metrics=EV.metrics_table(rv, {cid: f, **refs}, tp), go=EV.go_conditions(rv[m], f[m], tp[m], qk[0], qk[1], cp0), tests=S1.tests_all(w, f, cp0),
             stress=CV.stress_table(rv, f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, tp, h), sens=TB.sens_rows(tab, X, w, variants(cid), f, f_a0))
    return dict(cid=cid, X=X, run=run, f=f, a=a, t=t, names=list(X.columns))


def info_tables(fs: dict, w, cp0) -> pd.DataFrame:
    rule, desc = TB.info_table(fs, RULE, w, cp0), TB.info_table(fs, DESC, w, cp0)
    return pd.concat([rule.assign(role="правило п. 2.2"), desc.assign(role="описательно")], ignore_index=True)


def compute() -> dict:
    t0 = time.time()
    cp = CP0.read()
    cp0 = cp["0"]
    tab, S = S4.panel()
    w = S1.controls()
    bars, daily = FE.load_inputs()
    Qd = QF.features(bars, daily).reindex(S.index)
    sf = pd.read_parquet(IO.OUT / "SIGMA_features.parquet")
    X = pd.concat([S, sf.set_index("d")[SG.SIGMA_NAMES].reindex(S.index), Qd[QF.Q_NAMES]], axis=1)
    prior_ids = cp["4"]["configs"]
    prev = S4.prior_forecasts(w, prior_ids)
    cfg = {c: one_config(c, tab, X, w, cp0, prev["A0"]) for c in CIDS}
    fs = {**prev, **{c: cfg[c]["f"] for c in cfg}}
    r = dict(cp=cp, cp0=cp0, tab=tab, S=S, w=w, X=X, cfg=cfg, fs=fs, prev=prev, prior_ids=prior_ids, ids=[*prior_ids, *CIDS], cids=list(CIDS), sigma_sha=S1.sha_arr(sf.drop(columns="d")), q_sha=S1.sha_arr(Qd[QF.Q_NAMES]))
    r["info"] = info_tables(fs, w, cp0)
    r["years"] = YR.years_table(fs, w)
    r["levels"] = LV.levels_table([row for c in CIDS for row in LV.level_rows(c, cfg[c]["a"], w, cfg[c]["t"])])
    r["coef"] = {c: TB.coef_table(cfg[c]["run"], cfg[c]["names"]) for c in cfg}
    r["conv"] = {c: W5.conv_table(cfg[c]["run"], c) for c in ("F1", "F2")}
    r["checkpoint"] = checkpoint(r)
    r["seconds"] = time.time() - t0
    return r


def conv_summary(r: dict) -> dict:
    return {c: dict(refits=len(t), max_nit=int(t["nit"].max()), max_grad_norm=float(t["grad_norm"].max()), n_not_success=int((~t["success"]).sum())) for c, t in r["conv"].items()}


def checkpoint(r: dict) -> dict:
    cfg, w, cp0, info = r["cfg"], r["w"], r["cp0"], r["info"]
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    qs = {c: dict(all=float(Q.ql(rv, f).mean()), y2017_2021=float(Q.ql(rv[m], f[m]).mean())) for c, f in {**r["prev"], **{c: cfg[c]["f"] for c in cfg}}.items()}
    ys = r["years"][r["years"]["section"] == "итог"][["pair", "window", "sd_delta_qlike", "sd_delta_pct_K1", "mean_delta_pct_K1", "years_neg", "years_n"]]
    return dict(program="HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 5", configs=r["ids"], used=len(r["ids"]), budget=int(cp0["budget_V6"]), N=cp0["N"], N_wide=cp0["N_wide"], inputs_hash=IO.inputs_hash(), code_hash_v6_5=code_hash(),
                code_hash_v6_1=RG.code_hash(), code_hash_v6_0=IO.code_hash(), code_version=CODE_VERSION, qlike=qs, f_sha={c: S1.sha_arr(cfg[c]["f"]) for c in cfg}, f_sha_prev={c: S1.sha_arr(r["prev"][c]) for c in r["prev"]},
                sigma_sha=r["sigma_sha"], q_sha=r["q_sha"], s_features_sha=S1.sha_arr(r["S"].reset_index(drop=True)), n_dates=len(w), params=dict(min_env=FR.MIN_ENV, eta=FR.ETA, k_sd=FR.KSD, optimizer="L-BFGS-B", opt=FR.OPT, grid=list(FT.GRID)),
                new_information=info[(info["window"] == "2017_2021") & (info["role"] == "правило п. 2.2")][["pair", "delta_qlike", "dm_p_one", "years_neg", "new_information"]].to_dict("records"), years_summary=ys.to_dict("records"), convergence=conv_summary(r),
                go={c: [dict(cond=x["cond"], value=x["value"], passed=x["passed"]) for x in cfg[c]["t"]["go"].to_dict("records")] for c in cfg},
                coef_stability={c: TB.coef_stability(r["coef"][c], [n for n in cfg[c]["names"] if n not in r["S"].columns]) for c in cfg}, note="лист 5: потолок, не OOS",
                written_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def tables(r: dict) -> dict:
    """Имя файла -> (датафрейм, примечание) для CSV в docs/otchety/harj_hybrid_v6/."""
    out = {"F_INFO.csv": (r["info"], "G43: правило «новая информация» п. 2.2 (CB1 против A0, F1 и F2 против CB1; цели 2017–2021); описательно — CB1 против A1 и C1"),
           "F_YEARS.csv": (r["years"], "G43: годовые Δ QLIKE F − CB1 по годам цели t⁺, СКО по годам (ddof = 1) на 2017–2021 и 2017–2025, число лет с Δ < 0; 2026 — частичный год"),
           "F_LEVELS.csv": (r["levels"], "G44: уровни 1–5 этапа 21 блока")}
    for c, v in r["cfg"].items():
        for k, nm in (("metrics", "METRICS"), ("go", "GO"), ("tests", "TESTS"), ("stress", "STRESS"), ("sens", "SENS")):
            out["%s_%s.csv" % (c, nm)] = (v["t"][k], "G42: %s %s (S + Σ + Q, %s)" % (c, nm, "LIN" if c == "CB1" else "LIN с робастной целью"))
        out["%s_COEF.csv" % c] = (r["coef"][c], "G42: θ по датам переоценки, единицы z")
    out.update({"%s_CONV.csv" % c: (t, "G42: сходимость L-BFGS-B по датам переоценки (итерации, 2-норма градиента цели в решении)") for c, t in r["conv"].items()})
    return out


def parquets(r: dict) -> dict:
    return {"%s.parquet" % c: v["a"] for c, v in r["cfg"].items()}
