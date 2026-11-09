"""V6-3: стадия `3` = E1 (S + R, LIN), E2 (S + R, GBM) и, только по правилу плана п. 5, E3 (S + R + Σ, LIN); R — признаки шока и восстановления ρ1…ρ6 (`v6_3_r`) после автоматического слияния с Σ (`v6_3_merge`); диагностика обращения времени B (`v6_3_tr`, без бюджета).
Движок листов 1a–2 (признаки S, обучатели, переоценка, оценка, таблицы) не менялся: добавлены набор R, правило слияния, диагностика и строки конфигураций. A0 пересчитывается листом 1a; A1–A3, C1–C4 и Σ читаются с диска (SHA сверяются с CHECKPOINT 1b/2).
Решение по E3 — автоматически: (E1 или E2 против A0 несёт новую информацию) и (A1 или A2 против A0 — по A_INFO.csv)."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import time
from pathlib import Path

import pandas as pd

from models_v2.s13 import (v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_feat as FE, v6_1_fit as FT, v6_1_reg as RG, v6_1_stage as S1, v6_1_wf as WF,
                           v6_1b_lev as LV, v6_1b_sig as SG, v6_1b_tab as TB, v6_3_merge as MG, v6_3_r as RF)

CODE_VERSION = "v6.3"
LEARNER = {"E1": "LIN", "E2": "GBM", "E3": "LIN"}
EXTRA = {"E1": ("R",), "E2": ("R",), "E3": ("R", "sigma")}


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листа 3 (Д5): все `v6_3_*.py`."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_3_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def runner(cid: str, shift: float = 0):
    """(tab, X) -> результат переоценки; shift — соседняя точка: LIN — шаг сетки λ (±1), GBM — множитель числа деревьев (0,5 / 2)."""
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
    h = tab["h"].reindex(w.index).to_numpy()
    a = pd.DataFrame({"d": w.index, "t_plus": tp, "rv": rv, "H": h, "g": run["g"].reindex(w.index).to_numpy(), "F": f})
    t = dict(metrics=EV.metrics_table(rv, {cid: f, **refs}, tp), go=EV.go_conditions(rv[m], f[m], tp[m], qk[0], qk[1], cp0), tests=S1.tests_all(w, f, cp0),
             stress=CV.stress_table(rv, f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, tp, h), sens=TB.sens_rows(tab, X, w, variants(cid), f, f_a0))
    return dict(cid=cid, X=X, run=run, f=f, a=a, t=t, names=list(X.columns))


def flag(v) -> bool:
    return v == "True" if isinstance(v, str) else bool(v)


def new_info(info: pd.DataFrame, xs: tuple) -> dict:
    """x против A0, цели 2017–2021 (план п. 2.2): {x: новая информация} и числа."""
    s = info[(info["window"] == "2017_2021") & (info["y"] == "A0")].set_index("x")
    return {x: dict(new_information=flag(s.loc[x, "new_information"]), delta=float(s.loc[x, "delta_qlike"]), p_dm_one=float(s.loc[x, "dm_p_one"]), years_neg=int(s.loc[x, "years_neg"])) for x in xs}


def decide_e3(info_e: pd.DataFrame, info_a: pd.DataFrame) -> dict:
    """Правило плана: E3 считается, если (E1 или E2 против A0) и (A1 или A2 против A0) несут новую информацию. Автоматически, обе части — против A0 (не против E1/A1)."""
    e, a = new_info(info_e, ("E1", "E2")), new_info(info_a, ("A1", "A2"))
    return dict(compute=bool(any(v["new_information"] for v in e.values()) and any(v["new_information"] for v in a.values())), E=e, A=a)


def prior_forecasts(w: pd.DataFrame, ids: list) -> dict:
    return {c: pd.read_parquet(IO.OUT / (c + ".parquet")).set_index("d")["F"].reindex(w.index).to_numpy() for c in ids}


def pairs(with_e3: bool) -> list:
    return [("E1", "A0"), ("E2", "A0"), ("E2", "E1")] + ([("E3", "A1"), ("E3", "E1")] if with_e3 else [])


def compute() -> dict:
    t0 = time.time()
    r0 = S1.compute()
    cp0, w = r0["cp0"], r0["w"]
    tab, S = r0["tab"].set_index("d"), r0["S"].set_index("d")
    bars, daily = FE.load_inputs()
    Rall = RF.features(bars, daily).reindex(S.index)
    sf = pd.read_parquet(IO.OUT / "SIGMA_features.parquet")
    sigma = sf.set_index("d")
    merge, kept = MG.merge_table(Rall[RF.RHO_NAMES], sigma[SG.SIGMA_NAMES])
    if not kept:
        raise ValueError("все ρ исключены правилом слияния: E1 = A0 (противоречие плану, вопрос дирижёру)")
    parts = {"R": Rall[kept], "sigma": sigma[SG.SIGMA_NAMES]}
    prior_ids = CP0.read()["2"]["configs"][1:]
    prev = prior_forecasts(w, prior_ids)
    f_a0, cfg = r0["a0"]["F"].to_numpy(), {}
    for cid in ("E1", "E2"):
        cfg[cid] = one_config(cid, tab, S, parts, w, cp0, f_a0)
    fs = {"A0": f_a0, **{c: cfg[c]["f"] for c in cfg}, **prev}
    info = TB.info_table(fs, pairs(False), w, cp0)
    dec = decide_e3(info, IO.read_csv("A_INFO.csv"))
    if dec["compute"]:
        cfg["E3"] = one_config("E3", tab, S, parts, w, cp0, f_a0)
        fs["E3"] = cfg["E3"]["f"]
        info = TB.info_table(fs, pairs(True), w, cp0)
    r = dict(r0=r0, cfg=cfg, fs=fs, info=info, dec=dec, w=w, tab=tab, ids=["A0", *prior_ids, *cfg], cids=list(cfg), prev=prev, prior_ids=prior_ids, Rall=Rall.reset_index(names="d"), merge=merge, kept=kept, sigma_sha=S1.sha_arr(sf.drop(columns="d")))
    r["vs_ac"] = TB.info_table(fs, [(c, a) for c in cfg for a in ("A1", "A2", "C2")], w, cp0)
    r["levels"] = LV.levels_table([row for c in ["A0", *cfg] for row in LV.level_rows(c, *level_inputs(r, c))])
    r["coef"] = {c: TB.coef_table(cfg[c]["run"], cfg[c]["names"]) for c in cfg if LEARNER[c] == "LIN"}
    r["imp"] = {c: TB.imp_table(cfg[c]["run"], cfg[c]["names"]) for c in cfg if LEARNER[c] == "GBM"}
    r["checkpoint"] = checkpoint(r)
    r["seconds"] = time.time() - t0
    return r


def attach_tr(r: dict, tr: pd.DataFrame) -> dict:
    """Диагностика обращения времени (`v6_3_tr`, считает оркестратор `run_v6`, не путь прогноза) -> таблица TR_DIAG и сводка в CHECKPOINT."""
    r["tr"] = tr
    r["checkpoint"]["tr_summary"] = {m: {v: int(((tr["mode"] == m) & (tr["verdict"] == v)).sum()) for v in ("инвариантен", "меняет знак", "меняется")} | {"всего": int((tr["mode"] == m).sum())} for m in ("put", "poryadok")}
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
    qs = {c: dict(all=float(Q.ql(w["rv"].to_numpy(), r["fs"][c]).mean()), y2017_2021=float(Q.ql(w["rv"].to_numpy()[m], r["fs"][c][m]).mean())) for c in r["ids"]}
    mg, Ra = r["merge"], r["Rall"]
    cp = dict(program="HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 3", configs=r["ids"], used=len(r["ids"]), budget=int(cp0["budget_V6"]), N=cp0["N"], N_wide=cp0["N_wide"], inputs_hash=IO.inputs_hash(), code_hash_v6_3=code_hash(),
              code_hash_v6_1=RG.code_hash(), code_hash_v6_0=IO.code_hash(), code_version=CODE_VERSION, qlike=qs, f_sha={c: S1.sha_arr(cfg[c]["f"]) for c in cfg}, f_sha_prev={c: S1.sha_arr(r["prev"][c]) for c in r["prev"]},
              r_sha=S1.sha_arr(Ra[RF.RHO_NAMES]), sigma_sha=r["sigma_sha"], n_dates=len(w), r_kept=r["kept"], r_merged_with_sigma=[x for x in RF.RHO_NAMES if x not in r["kept"]],
              r_max_abs_corr_sigma={x: float(mg[mg["rho"] == x]["rho_max_abs"].iloc[0]) for x in RF.RHO_NAMES}, r_edge_days={c: int(Ra[c].sum()) for c in RF.FLAG_NAMES}, days=len(Ra),
              new_information=info[info["window"] == "2017_2021"][["pair", "delta_qlike", "dm_p_one", "years_neg", "new_information"]].to_dict("records"),
              go={c: [dict(cond=x["cond"], value=x["value"], passed=x["passed"]) for x in cfg[c]["t"]["go"].to_dict("records")] for c in cfg},
              coef_stability={c: TB.coef_stability(r["coef"][c], [n for n in cfg[c]["names"] if n in RF.RHO_NAMES]) for c in cfg if LEARNER[c] == "LIN"}, note="лист 3: потолок, не OOS")
    cp["E3"] = "посчитана по правилу" if r["dec"]["compute"] else {"решение": "не считается, правило не выполнено", "E": r["dec"]["E"], "A": r["dec"]["A"]}
    cp["written_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return cp


def tables(r: dict) -> dict:
    """Имя файла -> (датафрейм, примечание) для CSV в docs/otchety/harj_hybrid_v6/."""
    out = {"E_INFO.csv": (r["info"], "G32/G33: правило «новая информация» плана п. 2.2 (цели 2017–2021); окна 2022–2025 и 2026 показаны, в правиле не участвуют"),
           "E_VS_AC.csv": (r["vs_ac"], "G32: описательно — E против A1, A2, C2 (Δ в % уровня K1, DM); в правило 2.2 не входит"),
           "E_LEVELS.csv": (r["levels"], "G34: уровни 1–5 этапа 21 блока"),
           "R_MERGE.csv": (r["merge"], "G30: |корреляция| ρ с признаками Σ на закрытых парах 2015-07-07 … 2016-12-29; ρ с максимумом > 0,8 исключён из R до счёта E1"),
           "TR_DIAG.csv": (r["tr"], "G31: диагностика обращения времени задним числом (не вход прогноза, без бюджета): put — обращение пути (порядок баров обратный, доходности с обратным знаком), poryadok — только порядок; дни 2015-07-07 … 2021-12-30")}
    for c, v in r["cfg"].items():
        for k, nm in (("metrics", "METRICS"), ("go", "GO"), ("tests", "TESTS"), ("stress", "STRESS"), ("sens", "SENS")):
            out["%s_%s.csv" % (c, nm)] = (v["t"][k], "G32: %s %s (S + R%s, %s)" % (c, nm, " + Σ" if c == "E3" else "", LEARNER[c]))
    out.update({"%s_COEF.csv" % c: (d, "G32: θ LIN по датам переоценки, единицы z") for c, d in r["coef"].items()})
    out.update({"%s_IMP.csv" % c: (d, "G32: важность признаков GBM (gain, splits)") for c, d in r["imp"].items()})
    return out


def parquets(r: dict) -> dict:
    return {**{"%s.parquet" % c: v["a"] for c, v in r["cfg"].items()}, "R_features.parquet": r["Rall"]}
