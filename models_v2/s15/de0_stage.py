"""DE0: стадия — compute() (всё в памяти), write() (ПИШЕТ CSV, отчёт, CHECKPOINT ключ `0`), check() (ТОЛЬКО ЧИТАЕТ: пересчёт в памяти, сверка с диском, хэш выходов до и после).
Конфигурации не расходуются (0); обучения нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de0_core as C, de0_data as DATA, de0_doc as DOC, de0_groups as G, de0_io as IO, de0_quest as Q, de0_registry as R, de0_tables as T

TOL = 1e-12


def _fin_all_vs_forecast(d: dict) -> tuple:
    """QLIKE V7 `FIN_ALL.csv` (окна) против пересчёта из сохранённых прогнозов; -> (макс. невязка, {id: q})."""
    fin = pd.read_csv(IO.V7 / "FIN_ALL.csv", comment="#")
    year, rec, mx = np.asarray(d["tp"].year), {}, 0.0
    for s, v in d["all_series"].items():
        l = C.qlike_day(d["rv"], v)
        q = {"2018_2021": l[C.wmask(year, "2018_2021")].mean(), "2022_2025": l[C.wmask(year, "2022_2025")].mean(), "all": l.mean()}
        for w, x in q.items():
            row = fin[(fin["series"] == s) & (fin["window"] == w)]
            if len(row):
                mx = max(mx, abs(float(row.iloc[0]["qlike"]) - float(x)))
        rec[s] = dict(q_2018_2021=q["2018_2021"], q_2022_2025=q["2022_2025"], q_all_2018_2026=q["all"])
    return mx, rec


def ridge_feats(idx: pd.DatetimeIndex) -> dict:
    sf = DATA.sigma_s_features(idx)
    cache = {}

    def z_of(Y: int) -> np.ndarray:
        if Y not in cache:
            days, z = DATA.z_file(Y, 0)
            pos = days.get_indexer(idx)                  # дни после конца года Y и дни без z (напр. 31.12) — NaN: такие строки в обучение и прогноз не идут
            out = np.full((len(idx), z.shape[1]), np.nan)
            out[pos >= 0] = z[pos[pos >= 0]]
            cache[Y] = out
        return cache[Y]

    return {"S": lambda Y: sf["S"], "Σ": lambda Y: sf["SIGMA"], "z": z_of, "S+Σ+z": lambda Y: np.hstack([sf["S"], sf["SIGMA"], z_of(Y)])}


def compute() -> tuple:
    """-> (tables, numbers)."""
    d, store = DATA.load(), DATA.load_store()
    idx, tp, rv, f = d["idx"], d["tp"], d["rv"], d["f"]
    year = np.asarray(tp.year)
    feat = G.feature_table(d, store, DATA.z_file)
    lab = G.labels(feat)
    t = {"DE0_GAP_YEARS": T.gap_years(rv, f, year), "DE0_GAP_BASE": T.gap_base(rv, f, year), "DE0_UNDER_OVER": T.under_over_table(rv, f, year), "DE0_TAILS": T.tails_table(rv, f, year),
         "DE0_CAUSES": T.causes_table(rv, f, year), "DE0_GROUPS": G.group_table(lab, d["tab"].index.get_indexer(idx), rv, f, year), "DE0_ACF": T.acf_table(rv, f, year), "DE0_RUNS": T.runs_table(rv, f, year),
         "DE0_RIDGE": T.ridge_table(rv, f, year, idx, tp, ridge_feats(idx))}
    t["DE0_QUESTIONS"] = Q.answers(t["DE0_GROUPS"], t["DE0_ACF"], t["DE0_RUNS"], t["DE0_RIDGE"])
    mx_fin, rec = _fin_all_vs_forecast(d)
    reg = R.build()
    t["DE0_REGISTRY"] = reg
    gy = [C.identity_years(t["DE0_GAP_YEARS"][t["DE0_GAP_YEARS"]["hybrid"] == h].drop(columns="hybrid")) for h in T.HYB]
    ck = R.checks(reg, {k: v for k, v in rec.items() if k.startswith("cfg")})
    num = dict(n_dates=int(len(idx)), first_d=str(idx[0].date()), last_d=str(idx[-1].date()), max_fin_all_v7_vs_forecast=float(mx_fin), registry_max_window_vs_years=ck["max_window_vs_years"], registry_max_table_vs_forecast=ck["max_table_vs_forecast"],
               max_identity_years=float(max(gy)), max_identity_under_over=float(t["DE0_UNDER_OVER"]["identity"].max()), max_identity_tails=float(t["DE0_TAILS"]["identity"].max()), max_identity_causes=float(t["DE0_CAUSES"]["identity"].max()),
               n_registry_rows=int(len(reg)), n_group_rows=int(len(t["DE0_GROUPS"])), configs_spent=0)
    return t, num


def violations(num: dict) -> list:
    bad = []
    if num["max_fin_all_v7_vs_forecast"] > TOL * 100 or num["registry_max_table_vs_forecast"] > TOL * 100:
        bad.append("QLIKE таблиц V7 ≠ пересчёту из прогнозов")
    if num["registry_max_window_vs_years"] > 1e-9:
        bad.append("окна V1–V6 ≠ взвешенные годовые QLIKE")
    for k in ("max_identity_years", "max_identity_under_over", "max_identity_tails", "max_identity_causes"):
        if num[k] > TOL:
            bad.append("тождество %s: %.3g" % (k, num[k]))
    return bad


def write() -> None:
    t, num = compute()
    bad = violations(num)
    if bad:
        raise SystemExit("DE0: " + "; ".join(bad))
    for k in IO.TABLES:
        IO.write_csv(t[k], k)
    DOC.write(t)
    IO.merge_checkpoint("0", dict(numbers=num, sha256={k: IO.sha_file(IO.REP / (k + ".csv")) for k in IO.TABLES}, sha256_report=IO.sha_file(IO.DOC), code=IO.code_hash(), tables=IO.TABLES))


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0 расхождений); хэш выходов до и после сравнивается."""
    before = IO.outputs_hash()
    t, num = compute()
    bad = violations(num)
    cp = IO.read_cp().get("0", {})
    for k in IO.TABLES:
        p = IO.REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != IO.csv_text(t[k]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif cp.get("sha256", {}).get(k) != IO.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % k)
    if not IO.DOC.exists() or IO.DOC.read_text(encoding="utf-8") != DOC.build(t):
        bad.append("DE0_REPORT.md ≠ генератору")
    if cp.get("numbers") != num:
        bad.append("CHECKPOINT numbers ≠ пересчёту")
    if IO.outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad
