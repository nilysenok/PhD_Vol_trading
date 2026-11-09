"""V6-4: стадия `4` = D1, D2, D3 (историческая аналогия, `v6_4_an`) и контроль K9 (глобальная калибровка H·mean R, не конфигурация, в `ALL_TRIALS` не пишется). Обучателей нет; таблица H и сводка S читаются с диска (SHA S = CHECKPOINT 1a).
Движок оценки (метрики, «go», тесты, стресс, уровни, правило 2.2) — прежний. Предыдущие прогнозы A0–A3, C1–C4, E1–E3 читаются с диска (SHA = CHECKPOINT 1a/1b/2/3) и только описательно сравниваются."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s13 import (v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_fit as FT, v6_1_reg as RG, v6_1_stage as S1, v6_1b_lev as LV, v6_1b_tab as TB, v6_4_an as AN, v6_4_regime as RGM)

CODE_VERSION = "v6.4"
CIDS = ("D1", "D2", "D3")
MAIN = {"D1": "m1_50", "D2": "m2_100", "D3": "m3_50"}
VARS = {"D1": [("k = 25", "m1_25"), ("D1 (k = 50)", "m1_50"), ("k = 100", "m1_100")], "D2": [("k = 50", "m2_50"), ("D2 (k = 100)", "m2_100"), ("k = 200", "m2_200")], "D3": [("κ = 25", "m3_25"), ("D3 (κ = 50)", "m3_50"), ("κ = 100", "m3_100")]}
VAR_ARGS = dict(d1=AN.VAR_D1, d2=AN.VAR_D2, kappa=AN.VAR_KAPPA)
LIN_POOL = ("A0", "A1", "A3", "C1", "C3", "E1", "E3")
DESC = ("A2", "C2", "E2")


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листа 4 (Д5): все `v6_4_*.py`."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_4_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def panel() -> tuple:
    """(tab, S) с диска (листы 1a): index d. SHA S сверяется в `gate_bad`."""
    return pd.read_parquet(IO.OUT / "H_table.parquet").set_index("d"), pd.read_parquet(IO.OUT / "S_features.parquet").set_index("d")


def expected_sha(cp: dict) -> dict:
    return {"A0": cp["1a"]["a0_f_sha"], **cp["1b"]["f_sha"], **cp["2"]["f_sha"], **cp["3"]["f_sha"]}


def prior_forecasts(w: pd.DataFrame, ids: list) -> dict:
    return {c: pd.read_parquet(IO.OUT / (c + ".parquet")).set_index("d")["F"].reindex(w.index).to_numpy() for c in ids}


def forecast_of(m, h) -> tuple:
    g = np.log(np.asarray(m, float))
    return g, FT.forecast(h, g)


def sens(cid: str, st: pd.DataFrame, w: pd.DataFrame, h, f_k9) -> pd.DataFrame:
    rv, tp, rows = w["rv"].to_numpy(), w["t_plus"].to_numpy(), []
    for name, col in VARS[cid]:
        f = forecast_of(st[col].to_numpy(), h)[1]
        row = dict(variant=name, params=col)
        for win in ("all", "2017_2021"):
            m = Q.wmask(tp, win)
            q, q1, q9 = Q.ql(rv[m], f[m]), Q.ql(rv[m], w["K1"].to_numpy()[m]), Q.ql(rv[m], f_k9[m])
            row.update({"qlike_" + win: float(q.mean()), "d_pct_K1_" + win: 100 * float((q - q1).mean()) / float(q1.mean()), "d_pct_K9_" + win: 100 * float((q - q9).mean()) / float(q1.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def one_config(cid: str, st, w, cp0, h, f_k9) -> dict:
    g, f = forecast_of(st[MAIN[cid]].to_numpy(), h)
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    qk = [Q.ql(rv[m], w[k].to_numpy()[m]) for k in ("K2", "K3")]
    a = pd.DataFrame({"d": w.index, "t_plus": tp, "rv": rv, "H": h, "g": g, "F": f, "M": st[MAIN[cid]].to_numpy(), "n_pool": st["n_pool"].to_numpy()})
    t = dict(metrics=EV.metrics_table(rv, {cid: f, **refs, "K9": f_k9}, tp), go=EV.go_conditions(rv[m], f[m], tp[m], qk[0], qk[1], cp0), tests=S1.tests_all(w, f, cp0),
             stress=CV.stress_table(rv, f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, tp, h), sens=sens(cid, st, w, h, f_k9))
    return dict(cid=cid, f=f, a=a, t=t)


def info_tables(fs: dict, w, cp0) -> tuple:
    """Правило 2.2: D против K9 (Y = K9); описательно: K9 против K1, D против A2, C2, E2 и лучшей LIN-конфигурации (по QLIKE 2017–2021)."""
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    lin = min((c for c in LIN_POOL), key=lambda c: float(Q.ql(rv[m], fs[c][m]).mean()))
    rule = TB.info_table(fs, [(c, "K9") for c in CIDS], w, cp0)
    desc = TB.info_table(fs, [("K9", "K1")] + [(c, y) for c in CIDS for y in (*DESC, lin)], w, cp0)
    return pd.concat([rule.assign(role="правило п. 2.2"), desc.assign(role="описательно")], ignore_index=True), lin


def compute() -> dict:
    t0 = time.time()
    cp = CP0.read()
    cp0 = cp["0"]
    tab, S = panel()
    w = S1.controls()
    st = AN.run(tab, S, w.index, var=VAR_ARGS)
    h = tab["h"].reindex(w.index).to_numpy()
    g9, f_k9 = st["m_glob"].pipe(np.log).to_numpy(), h * st["m_glob"].to_numpy()
    cfg = {c: one_config(c, st, w, cp0, h, f_k9) for c in CIDS}
    prior_ids = cp["3"]["configs"]
    prev = prior_forecasts(w, prior_ids)
    fs = {**prev, "K1": w["K1"].to_numpy(), "K9": f_k9, **{c: cfg[c]["f"] for c in cfg}}
    info, lin = info_tables(fs, w, cp0)
    k9 = pd.DataFrame({"d": w.index, "t_plus": w["t_plus"].to_numpy(), "rv": w["rv"].to_numpy(), "H": h, "g": g9, "F": f_k9, "M": st["m_glob"].to_numpy(), "n_pool": st["n_pool"].to_numpy()})
    reg = RGM.with_verdict(RGM.table(w["rv"].to_numpy(), cfg["D2"]["f"], f_k9, w["t_plus"].to_numpy(), st["tercile"].to_numpy(), h, st["m2_100"].to_numpy()))
    r = dict(cp=cp, cp0=cp0, tab=tab, S=S, w=w, st=st, h=h, cfg=cfg, fs=fs, info=info, lin=lin, k9=k9, f_k9=f_k9, reg=reg, prev=prev, prior_ids=prior_ids, ids=[*prior_ids, *CIDS], cids=list(CIDS))
    r["levels"] = LV.levels_table([row for c in CIDS for row in LV.level_rows(c, cfg[c]["a"], w, cfg[c]["t"])])
    r["checkpoint"] = checkpoint(r)
    r["seconds"] = time.time() - t0
    return r


def checkpoint(r: dict) -> dict:
    cfg, w, cp0 = r["cfg"], r["w"], r["cp0"]
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    qs = {c: dict(all=float(Q.ql(rv, f).mean()), y2017_2021=float(Q.ql(rv[m], f[m]).mean())) for c, f in {**{c: r["prev"][c] for c in r["prior_ids"]}, "K9": r["f_k9"], **{c: cfg[c]["f"] for c in cfg}}.items()}
    info, reg = r["info"], r["reg"]
    return dict(program="HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 4", configs=r["ids"], used=len(r["ids"]), budget=int(cp0["budget_V6"]), N=cp0["N"], N_wide=cp0["N_wide"], inputs_hash=IO.inputs_hash(), code_hash_v6_4=code_hash(),
                code_hash_v6_1=RG.code_hash(), code_hash_v6_0=IO.code_hash(), code_version=CODE_VERSION, qlike=qs, f_sha={c: S1.sha_arr(cfg[c]["f"]) for c in cfg}, k9_f_sha=S1.sha_arr(r["f_k9"]), f_sha_prev={c: S1.sha_arr(r["prev"][c]) for c in r["prev"]},
                s_features_sha=S1.sha_arr(r["S"].reset_index(drop=True)), n_dates=len(w), params=dict(k_d1=AN.K_D1, k_d2=AN.K_D2, k_h=AN.K_H, kappa=AN.KAPPA, z=AN.ZN, clip="[e^-1, e]"), best_lin=r["lin"],
                new_information=info[(info["window"] == "2017_2021") & (info["role"] == "правило п. 2.2")][["pair", "delta_qlike", "dm_p_one", "years_neg", "new_information"]].to_dict("records"),
                go={c: [dict(cond=x["cond"], value=x["value"], passed=x["passed"]) for x in cfg[c]["t"]["go"].to_dict("records")] for c in cfg}, regime_verdict=reg["verdict"].iloc[-1],
                corr_lnM_lnH={x.window: x.corr_lnM_lnH for x in reg[reg["section"] == "корреляция"].itertuples()}, note="лист 4: потолок, не OOS; K9 — контроль, не конфигурация, в N не входит",
                written_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def tables(r: dict) -> dict:
    """Имя файла -> (датафрейм, примечание) для CSV в docs/otchety/harj_hybrid_v6/."""
    out = {"D_INFO.csv": (r["info"], "G38: правило «новая информация» п. 2.2 (Y = K9, цели 2017–2021); описательно — K9 против K1, D против A2, C2, E2 и лучшей LIN (%s)" % r["lin"]),
           "D_REGIME.csv": (r["reg"], "G38: Δ D2 − K9 по терцилям H_t (пороги по пулу дня d), корреляция ln M_D2 и ln H_t, вывод"),
           "D_LEVELS.csv": (r["levels"], "G39: уровни 1–5 этапа 21 блока"),
           "K9_METRICS.csv": (EV.metrics_table(r["w"]["rv"].to_numpy(), {"K9": r["f_k9"], **{k: r["w"][k].to_numpy() for k in RG.CTRL}}, r["w"]["t_plus"].to_numpy()), "G37: контроль K9 = H·mean R пула дня d (не конфигурация)")}
    notes = dict(metrics="QLIKE, RMSE, MAE, R², bias по окнам", go="условия go 1–5", tests="Δ к контролям, DM, p Холма, охват", stress="стресс-окна", sens="соседние точки (k, κ), не конфигурации")
    for c, v in r["cfg"].items():
        for k, nm in (("metrics", "METRICS"), ("go", "GO"), ("tests", "TESTS"), ("stress", "STRESS"), ("sens", "SENS")):
            out["%s_%s.csv" % (c, nm)] = (v["t"][k], "G37: %s %s (аналоги Z_D)" % (c, notes[k]))
    return out


def parquets(r: dict) -> dict:
    return {**{"%s.parquet" % c: v["a"] for c, v in r["cfg"].items()}, "K9.parquet": r["k9"]}
