"""V6-1a: стадия `1a` = пересчёт A0 в памяти (`compute`, чистая функция от входов), запись (`write`, не в `--check`) и сверка с диском (`check`, только чтение). A0 = S + LIN, база H = K1, одна конфигурация из 18.
Выходы: `data/regen/harj_hybrid_v6/{H_table,S_features,A0}.parquet`; `docs/otchety/harj_hybrid_v6/{A0_METRICS,A0_GO,A0_TESTS,A0_STRESS,A0_SENS}.csv`, `ALL_TRIALS.csv` (строка `Ж-227:A0`), `CHECKPOINT.json` ключ `1a`."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC, v4_fin_data as FD, v4_io as IO4, v4_rqc_calib as RQ
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_feat as FE, v6_1_fit as FT, v6_1_h as HH, v6_1_reg as RG, v6_1_wf as WF

TABLES = {"A0_METRICS.csv": "metrics", "A0_GO.csv": "go", "A0_TESTS.csv": "tests", "A0_STRESS.csv": "stress", "A0_SENS.csv": "sens"}
NOTES = {"metrics": "G15: A0 и контроли — QLIKE, RMSE, MAE, R², bias по окнам", "go": "G15: условия go 1–5 с числами", "tests": "G15: Δ к контролям с интервалами, DM, p Холма, охват", "stress": "G15: стресс-окна",
         "sens": "G15: чувствительность λ на шаг сетки вверх и вниз (соседние точки, не конфигурации)"}
PARQ = {"H_table.parquet": "tab", "S_features.parquet": "S", "A0.parquet": "a0"}
IDS = ["A0"]


def sha_arr(*a) -> str:
    h = hashlib.sha256()
    for x in a:
        h.update(np.ascontiguousarray(np.asarray(x, float)).tobytes())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def controls() -> pd.DataFrame:
    """Общие даты (2271): rv, t⁺, K1, K2, K3, K8, V4:NTP2 — готовые прогнозы SBER листа 0 (не переобучаются)."""
    a = IO4.aligned("SBER")
    w = pd.DataFrame({"t_plus": a["t_plus"].to_numpy(), "rv": a["rv"].to_numpy(float), "K1": a["h"].to_numpy(float), "K2": a["k2"].to_numpy(float), "K3": a["k3"].to_numpy(float)}, index=pd.DatetimeIndex(a["date"]))
    w["K8"] = RQ.k8(w["K1"], w["K2"], w["K3"])
    f = pd.read_parquet(FD.OUT / FD.file_of("NTP2"), columns=["ticker", "date", "t_plus", "F"])
    f = f[f["ticker"] == "SBER"]
    w["V4:NTP2"] = pd.Series(f["F"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(f["date"]))).reindex(w.index).to_numpy()
    return w


def run_lin(tab, S, shift: int = 0) -> dict:
    return WF.walk_forward(tab, S, lambda X, e: FT.fit_lin(X, e, lam_shift=shift))


def sens_table(tab, S, w, f_a0) -> pd.DataFrame:
    rows = []
    for name, sh in (("λ − 1 шаг сетки", -1), ("A0 (λ выбран)", 0), ("λ + 1 шаг сетки", 1)):
        r = run_lin(tab, S, sh)
        f = WF.forecasts(tab, r["g"]).reindex(w.index).to_numpy()
        row = dict(variant=name, lam_counts=json.dumps({str(k): v for k, v in pd.Series([x["model"].lam for x in r["fits"]]).value_counts().sort_index().items()}))
        for win in ("all", "2017_2021"):
            m = Q.wmask(w["t_plus"].to_numpy(), win)
            q, k1 = Q.ql(w["rv"].to_numpy()[m], f[m]), float(Q.ql(w["rv"].to_numpy()[m], w["K1"].to_numpy()[m]).mean())
            row.update({"qlike_" + win: float(q.mean()), "d_pct_K1_" + win: 100 * float((q - Q.ql(w["rv"].to_numpy()[m], w["K1"].to_numpy()[m])).mean()) / k1,
                        "d_pct_A0_" + win: 100 * float((q - Q.ql(w["rv"].to_numpy()[m], f_a0[m])).mean()) / k1})
        rows.append(row)
    return pd.DataFrame(rows)


def tests_all(w, f, cp0) -> pd.DataFrame:
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    t = EV.tests_table(w["rv"].to_numpy(), f, refs, w["t_plus"].to_numpy(), cp0).assign(section="delta_dm")
    cov = [dict(section="coverage", window=win, base=b, **CV.coverage(w["rv"].to_numpy()[m], w[b].to_numpy()[m], f[m]))
           for win in ("all", "2017_2021") for b in ("K8", "K3") for m in [Q.wmask(w["t_plus"].to_numpy(), win)]]
    return pd.concat([t, pd.DataFrame(cov)], ignore_index=True)


def compute() -> dict:
    t0 = time.time()
    cp0 = CP0.read()["0"]
    p, k1 = HH.H.make_panel(HH.daily_sber()), HH.k1_stored()
    tab = HH.h_table(p, k1)
    hc = HH.check_h(tab, k1)
    bars, daily = FE.load_inputs()
    S = FE.assemble(FE.raw_features(bars, daily), tab["h"])
    run = run_lin(tab, S)
    w = controls()
    F = WF.forecasts(tab, run["g"])
    f = F.reindex(w.index).to_numpy()
    a0 = pd.DataFrame({"d": w.index, "t_plus": w["t_plus"].to_numpy(), "rv": w["rv"].to_numpy(), "H": tab["h"].reindex(w.index).to_numpy(), "g": run["g"].reindex(w.index).to_numpy(), "F": f})
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    qk = [Q.ql(w["rv"].to_numpy()[m], w[k].to_numpy()[m]) for k in ("K2", "K3")]
    refs = {k: w[k].to_numpy() for k in RG.CTRL}
    go = EV.go_conditions(w["rv"].to_numpy()[m], f[m], w["t_plus"].to_numpy()[m], qk[0], qk[1], cp0)
    ser = {"A0": f, **{k: v for k, v in refs.items()}}
    r = dict(tab=tab.reset_index(names="d"), S=S.reset_index(names="d"), a0=a0, hc=hc, w=w, run=run, cp0=cp0, go=go, ident=FE.identity_max(bars, daily), k1_sha=IO.sha_file(IO4.path_of("H", "SBER")), bars_sha=IO.sha_file(FE.allowed_files()[0]),
             metrics=EV.metrics_table(w["rv"].to_numpy(), ser, w["t_plus"].to_numpy()), tests=tests_all(w, f, cp0), stress=CV.stress_table(w["rv"].to_numpy(), f, {k: refs[k] for k in ("K1", "K2", "K3", "K8")}, w["t_plus"].to_numpy(), tab["h"].reindex(w.index).to_numpy()),
             sens=sens_table(tab, S, w, f), new_info=EV.new_information(w["rv"].to_numpy(), f, w["K1"].to_numpy(), w["t_plus"].to_numpy()))
    t = tab.reindex(w.index)
    r["align"] = dict(rv_maxdiff=float(np.abs(t["rv_next"] - w["rv"]).max()), h_maxdiff=float(np.abs(t["h"] - w["K1"]).max()), t_plus_equal=bool((t["t_plus"].to_numpy() == w["t_plus"].to_numpy()).all()))
    r["checkpoint"] = checkpoint(r)
    r["seconds"] = time.time() - t0
    return r


def checkpoint(r: dict) -> dict:
    a0, w, m = r["a0"], r["w"], Q.wmask(r["w"]["t_plus"].to_numpy(), "2017_2021")
    q = Q.ql(a0["rv"].to_numpy(), a0["F"].to_numpy())
    lam = pd.Series([x["model"].lam for x in r["run"]["fits"]]).value_counts().sort_index()
    return {"program": "HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 1a", "configs": IDS, "used": len(IDS), "budget": int(r["cp0"]["budget_V6"]),"N": r["cp0"]["N"], "N_wide": r["cp0"]["N_wide"], "inputs_hash": IO.inputs_hash(),
            "code_hash_v6_1": RG.code_hash(), "code_hash_v6_0": IO.code_hash(), "code_version": RG.CODE_VERSION, "k1_sha256": r["k1_sha"], "bars_sha256": r["bars_sha"], "h_check": r["hc"], "align": r["align"],
            "s_features_sha": sha_arr(r["S"].drop(columns="d")), "a0_f_sha": sha_arr(a0["F"]), "n_dates": len(a0), "n_refits": len(r["run"]["fits"]), "lam_counts": {str(k): int(v) for k, v in lam.items()},
            "qlike_all": float(q.mean()), "qlike_2017_2021": float(q[m].mean()), "go": r["go"].to_dict("records"), "go_all": bool(r["go"]["passed"].all()), "new_information_vs_K1": r["new_info"],
            "note": "лист 1a: A0 = S + LIN; потолок, не OOS", "written_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def stable(d: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in d.items() if k != "written_at"}))


def write(r: dict) -> list:
    IO.OUT.mkdir(parents=True, exist_ok=True)
    for f, k in PARQ.items():
        r[k].to_parquet(IO.OUT / f, index=False)
    out = [RG.write_csv(r[k], f, NOTES[k]) for f, k in TABLES.items()]
    cp, old = CP0.read(), CP0.read().get("1a")
    entry = {**r["checkpoint"], "written_at": old["written_at"]} if old and stable(old) == stable(r["checkpoint"]) else r["checkpoint"]
    CP0.CP.write_text(json.dumps({**cp, "1a": entry}, ensure_ascii=False, indent=1))
    if "Ж-227:A0," not in RG.CP0.ALL_TRIALS.read_text(encoding="utf-8"):      # строки следующих листов (A1–A3) при перезаписи 1a не стираются
        RG.write_trials(IDS)
    return out


def gate_bad(r: dict) -> list:
    hc, a0, S, bad = r["hc"], r["a0"], r["S"], []
    if hc["n_warm"] != HH.N_WARM or hc["first_warm"] != HH.WARM_FIRST or hc["max_diff_early"] != 0.0:
        bad.append("G10: прогрев H: %s" % hc)
    prior = IO.read_csv("CONTROLS_SBER.csv").set_index("item").loc["SBER:K1", "sha_v5"]
    if r["k1_sha"] != prior:
        bad.append("G10: SHA K1 изменился")
    if r["ident"] > FE.TOL_ID or S.drop(columns="d").isna().any().any():
        bad.append("G11: тождество Σr² = RV (%.2e) или NaN в S" % r["ident"])
    if len(a0) != 2271 or not (np.isfinite(a0["F"]) & (a0["F"] > 0)).all():
        bad.append("G13/G15: A0 не на 2271 датах или F ≤ 0 / не конечно")
    al = r["align"]
    if al["rv_maxdiff"] != 0.0 or al["h_maxdiff"] != 0.0 or not al["t_plus_equal"]:
        bad.append("G13/G15: цель или H таблицы обучения ≠ общим датам листа 0: %s" % al)
    z = a0["g"] == 0
    if z.any() and not (a0.loc[z, "F"] == a0.loc[z, "H"]).all():
        bad.append("G12: g = 0, но F ≠ H побитно")
    return bad


def check(r: dict) -> list:
    bad = gate_bad(r)
    for f, k in PARQ.items():
        if not (IO.OUT / f).exists() or not FC.frames_equal(pd.read_parquet(IO.OUT / f), r[k], 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    for f, k in TABLES.items():
        if not (IO.REP / f).exists() or not FC.frames_equal(IO.read_csv(f), r[k], 1e-9):
            bad.append("%s: нет файла или диск ≠ пересчёт" % f)
    cp = CP0.read().get("1a")
    if cp is None or stable(cp) != stable(r["checkpoint"]):
        bad.append("G15: CHECKPOINT ключ 1a на диске ≠ пересчёту")
    txt = RG.CP0.ALL_TRIALS.read_text(encoding="utf-8") if RG.CP0.ALL_TRIALS.exists() else ""
    if RG.trials_bad(txt) or "Ж-227:A0," not in txt:      # с листа 1b в реестре идут и строки A1–A3: проверяется состав, не равенство одной строке
        bad.append("G15: ALL_TRIALS.csv: нет строки Ж-227:A0 или недопустимые строки: %s" % RG.trials_bad(txt))
    return bad
