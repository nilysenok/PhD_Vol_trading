"""V6-6: стадия `6` = A0g (S + GBM, без новых признаков; 18-я конфигурация бюджета, на месте CB2). ОТСТУПЛЕНИЕ от предрегистрации (вопрос 1 [V6], умолчание 09.11 10:01): абляционная база, разделяющая вклад новых признаков и вклад нелинейного обучателя.
Обучатель — тот же `v6_1_fit.fit_gbm` (параметры плана, ранняя остановка на последних 252 закрытых парах, тот же clip и та же помесячная переоценка), что в A2, C2, C4, E2: изменён только набор признаков (S). Прежние прогнозы и таблицы читаются с диска и не пересчитываются.
`G_INFO`: правило п. 2.2 для A2, C2, C4, E2 против A0g; рядом прежние строки против A0 (копия A_INFO/C_INFO/E_INFO без изменений); описательно A0g против A0, K1, K3, K8, V4:NTP2."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import time
from pathlib import Path

import pandas as pd

from models_v2.s13 import (v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_fit as FT, v6_1_reg as RG, v6_1_stage as S1, v6_1_wf as WF, v6_1b_lev as LV, v6_1b_tab as TB, v6_4_stage as S4)

CODE_VERSION = "v6.6"
CID = "A0g"
RULE_X = ("A2", "C2", "C4", "E2")
OLD_FILE = {"A2": "A_INFO.csv", "C2": "C_INFO.csv", "C4": "C_INFO.csv", "E2": "E_INFO.csv"}
DESC_Y = ("A0", "K1", "K3", "K8", "V4:NTP2")
ROLE_RULE, ROLE_OLD, ROLE_DESC = "правило п. 2.2 против A0g", "прежний вывод против A0 (без изменений)", "описательно"


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листа 6 (Д5): все `v6_6_*.py`."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_6_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def run_gbm(tab, X, trees_mult: float = 1.0) -> dict:
    """Единый путь GBM листов 1b/2/3 и 6: `WF.walk_forward` + `FT.fit_gbm`. На S + Σ даёт A2 побитно (тест)."""
    return WF.walk_forward(tab, X, lambda A, e: FT.fit_gbm(A, e, trees_mult=trees_mult))


def variants() -> dict:
    return {"число деревьев ×0,5": lambda tab, X: run_gbm(tab, X, 0.5), "A0g (деревья по ранней остановке)": lambda tab, X: run_gbm(tab, X, 1.0), "число деревьев ×2": lambda tab, X: run_gbm(tab, X, 2.0)}


def expected_sha(cp: dict) -> dict:
    return {"A0": cp["1a"]["a0_f_sha"], **cp["1b"]["f_sha"], **cp["2"]["f_sha"], **cp["3"]["f_sha"], **cp["4"]["f_sha"], **cp["5"]["f_sha"]}


def one_config(tab, S, w, cp0, f_a0) -> dict:
    run = run_gbm(tab, S)
    f = WF.forecasts(tab, run["g"]).reindex(w.index).to_numpy()
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    qk = [Q.ql(rv[m], w[k].to_numpy()[m]) for k in ("K2", "K3")]
    h = tab["h"].reindex(w.index).to_numpy()
    a = pd.DataFrame({"d": w.index, "t_plus": tp, "rv": rv, "H": h, "g": run["g"].reindex(w.index).to_numpy(), "F": f})
    t = dict(metrics=EV.metrics_table(rv, {CID: f, **refs}, tp), go=EV.go_conditions(rv[m], f[m], tp[m], qk[0], qk[1], cp0), tests=S1.tests_all(w, f, cp0),
             stress=CV.stress_table(rv, f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, tp, h), sens=TB.sens_rows(tab, S, w, variants(), f, f_a0))
    return dict(cid=CID, run=run, f=f, a=a, t=t, names=list(S.columns))


def old_rows() -> pd.DataFrame:
    """Прежние строки «X против A0» (правило 2.2) из A_INFO / C_INFO / E_INFO — копия без пересчёта."""
    out = []
    for x in RULE_X:
        t = IO.read_csv(OLD_FILE[x])
        out.append(t[t["pair"] == "%s vs A0" % x])
    return pd.concat(out, ignore_index=True)


def info_tables(fs: dict, w, cp0) -> pd.DataFrame:
    rule = TB.info_table(fs, [(x, CID) for x in RULE_X], w, cp0).assign(role=ROLE_RULE)
    desc = TB.info_table(fs, [(CID, y) for y in DESC_Y], w, cp0).assign(role=ROLE_DESC)
    old = old_rows().assign(role=ROLE_OLD)
    return pd.concat([rule, old[list(rule.columns)], desc], ignore_index=True)


def compute() -> dict:
    t0 = time.time()
    cp = CP0.read()
    cp0, prior_ids = cp["0"], cp["5"]["configs"]
    tab, S = S4.panel()
    w = S1.controls()
    prev = S4.prior_forecasts(w, prior_ids)
    c = one_config(tab, S, w, cp0, prev["A0"])
    fs = {**prev, CID: c["f"], **{k: w[k].to_numpy() for k in RG.CTRL}}
    r = dict(cp=cp, cp0=cp0, tab=tab, S=S, w=w, cfg={CID: c}, fs=fs, prev=prev, prior_ids=prior_ids, ids=[*prior_ids, CID], cids=[CID])
    r["info"] = info_tables(fs, w, cp0)
    r["levels"] = LV.levels_table(LV.level_rows(CID, c["a"], w, c["t"]))
    r["imp"] = TB.imp_table(c["run"], c["names"])
    r["checkpoint"] = checkpoint(r)
    r["seconds"] = time.time() - t0
    return r


def checkpoint(r: dict) -> dict:
    c, w, cp0, info = r["cfg"][CID], r["w"], r["cp0"], r["info"]
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    qs = {k: dict(all=float(Q.ql(rv, f).mean()), y2017_2021=float(Q.ql(rv[m], f[m]).mean())) for k, f in {**r["prev"], CID: c["f"]}.items()}
    new = info[(info["window"] == "2017_2021") & (info["role"] != ROLE_DESC)][["role", "pair", "delta_qlike", "delta_pct", "dm_p_one", "years_neg", "new_information"]]
    return dict(program="HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 6", configs=r["ids"], used=len(r["ids"]), budget=int(cp0["budget_V6"]), N=cp0["N"], N_wide=cp0["N_wide"], inputs_hash=IO.inputs_hash(), code_hash_v6_6=code_hash(),
                code_hash_v6_1=RG.code_hash(), code_hash_v6_0=IO.code_hash(), code_version=CODE_VERSION, qlike=qs, f_sha={CID: S1.sha_arr(c["f"])}, f_sha_prev={k: S1.sha_arr(v) for k, v in r["prev"].items()},
                s_features_sha=S1.sha_arr(r["S"].reset_index(drop=True)), n_dates=len(w), features=c["names"], learner="GBM (v6_1_fit.fit_gbm, параметры плана)",
                deviation="отступление от предрегистрации: A0g считается на месте CB2 (вопрос 1 [V6], умолчание 09.11 10:01); обе версии вывода (против A0g и против A0) — в G_INFO",
                new_information=new.to_dict("records"), go=[dict(cond=x["cond"], value=x["value"], passed=x["passed"]) for x in c["t"]["go"].to_dict("records")], note="лист 6: потолок, не OOS",
                written_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def tables(r: dict) -> dict:
    """Имя файла -> (датафрейм, примечание) для CSV в docs/otchety/harj_hybrid_v6/."""
    out = {"G_INFO.csv": (r["info"], "G47: правило п. 2.2 для A2, C2, C4, E2 против A0g; рядом прежний вывод против A0 (копия A/C/E_INFO); описательно A0g против A0, K1, K3, K8, V4:NTP2. Отступление от предрегистрации, вопрос 1"),
           "G_LEVELS.csv": (r["levels"], "G48: уровни 1–5 этапа 21 блока для A0g")}
    for k, nm in (("metrics", "METRICS"), ("go", "GO"), ("tests", "TESTS"), ("stress", "STRESS"), ("sens", "SENS")):
        out["A0g_%s.csv" % nm] = (r["cfg"][CID]["t"][k], "G46: A0g %s (S, GBM)" % nm)
    out["A0g_IMP.csv"] = (r["imp"], "G46: важность признаков GBM (gain, splits)")
    return out


def parquets(r: dict) -> dict:
    return {"A0g.parquet": r["cfg"][CID]["a"]}
