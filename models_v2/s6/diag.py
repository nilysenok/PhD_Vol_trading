"""PREDICTIONS / ANNUAL_PERFORMANCE / REGIME_PERFORMANCE для каждого алгоритма отдельно (п. 4 задания; R5: режимы только по данным <= d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common, pit
from models_v2.s6 import paths6

ALG = {"xgboost": ("XGB", "Б-2 fx_b2_g132_b"), "lightgbm": ("LGBM", "LightGBM-аналог Б-2 (наследованные HP I2)")}
JUMP_SHARE = 0.3        # скачок по факту: j/rv дня > 0.3 (объявлено до счёта)


def sigma_of(alg: str) -> pd.DataFrame:
    c = common.controls()
    if alg == "xgboost":
        return c.rename(columns={"s2_b2": "s2"})[["d", "target_date", "rv", "s2", "s2_harj"]]
    fc = pd.read_parquet(paths6.OUT / "lightgbm" / "fc_lgbm_b2.parquet")
    if not (fc["d"].to_numpy("datetime64[ns]") == c["d"].to_numpy("datetime64[ns]")).all():
        raise ValueError("сетка LightGBM-аналога не равна сетке Б-2")
    return pd.DataFrame({"d": c["d"], "target_date": c["target_date"], "rv": c["rv"], "s2": fc["sigma2"].to_numpy(float), "s2_harj": c["s2_harj"]})


def regime_pit(rv_w: np.ndarray, stress: np.ndarray, include_current: bool = False) -> np.ndarray:
    """Режим на d: квинтиль RV_w по 252 предыдущим значениям (pit.past_bucket) и стресс Г4. include_current — мутация № 22."""
    q = pit.past_bucket(rv_w, [.2, .4, .6, .8], include_current=include_current)
    return np.array(["q%d|%s" % (int(a), "стресс" if s else "спокойно") if a == a else "q_nan|%s" % ("стресс" if s else "спокойно") for a, s in zip(q, stress)])


def inputs(base: pd.DataFrame) -> pd.DataFrame:
    """Режимные входы на дате d (без факта цели): rv_w из daily_p1, стресс Г4 из stress_p3, число признаков Б-2 из PREDICTIONS Б-2."""
    dl = pd.read_parquet(common.OUT / "daily_p1.parquet")
    st = pd.read_parquet(common.OUT / "stress_p3.parquet").set_index("d")["stress"]
    x = pd.DataFrame({"d": base["d"], "rv_w": dl["rv_w"].reindex(base["d"]).to_numpy(), "j": dl["j"].reindex(base["d"]).to_numpy(), "stress": st.reindex(base["d"]).to_numpy(bool)})
    return x


def n_features() -> int:
    old = pd.read_parquet(common.REP / "PREDICTIONS.parquet")
    return sum(c.startswith("f_") for c in old.columns)


def predictions(alg: str) -> pd.DataFrame:
    a, _ = ALG[alg]
    b = sigma_of(alg)
    x = inputs(b)
    reg = regime_pit(x["rv_w"].to_numpy(float), x["stress"].to_numpy())
    out = pd.DataFrame({"date": b["target_date"], "RV_actual": b["rv"], a + "_forecast": b["s2"], "HAR_forecast": b["s2_harj"],
                        "QLIKE_" + a: metrics.qlike(b["rv"], b["s2"]), "QLIKE_HAR": metrics.qlike(b["rv"], b["s2_harj"]), "regime": reg,
                        "features": "Б-2 (m_mkx): %d признаков" % n_features()})
    return out


def _m(rv, s2, qa, qh) -> dict:
    e = np.log(rv) - np.log(s2)
    sst = float(((np.log(rv) - np.log(rv).mean()) ** 2).sum())
    return {"n": len(rv), "qlike": float(qa.mean()), "rmse_ln": float(np.sqrt((e ** 2).mean())), "mae_ln": float(np.abs(e).mean()), "bias_ln": float(e.mean()),
            "r2_ln": float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan, "share_better_than_har": float((qa < qh).mean()), "delta_vs_har": float(qa.mean() - qh.mean())}


def periods(td: pd.Series, qa: np.ndarray, qh: np.ndarray) -> dict:
    y = td.dt.year.to_numpy()
    top20 = np.argsort(-np.abs(qa - qh))[:20]
    m = {"2017–2021": y <= 2021, "2022": y == 2022, "2023–2025": (y >= 2023) & (y <= 2025), "2026": y == 2026, "весь период": np.ones(len(y), bool),
         "без 24.02.2022": (td != "2022-02-24").to_numpy()}
    keep = np.ones(len(y), bool)
    keep[top20] = False
    m["без топ-20 дней (по |ΔQLIKE к HAR|)"] = keep
    return m


def annual(alg: str, pr: pd.DataFrame) -> pd.DataFrame:
    a, _ = ALG[alg]
    td, rv, s2 = pr["date"], pr["RV_actual"].to_numpy(float), pr[a + "_forecast"].to_numpy(float)
    qa, qh = pr["QLIKE_" + a].to_numpy(), pr["QLIKE_HAR"].to_numpy()
    rows = []
    y = td.dt.year.to_numpy()
    for yy in paths6.YEARS:
        mk = y == yy
        rows.append({"slice": "год %d" % yy, "kind": "год", **_m(rv[mk], s2[mk], qa[mk], qh[mk])})
    for lab, mk in periods(td, qa, qh).items():
        rows.append({"slice": lab, "kind": "период", **_m(rv[mk], s2[mk], qa[mk], qh[mk])})
    return pd.DataFrame(rows)


def regime_masks(pr: pd.DataFrame, a: str) -> list:
    """[(вид, метка, маска, пометка PIT/по факту, исключающий_набор)]."""
    rv, s2 = pr["RV_actual"].to_numpy(float), pr[a + "_forecast"].to_numpy(float)
    q = pr["regime"].str.split("|").str[0].to_numpy()
    st = pr["regime"].str.endswith("стресс").to_numpy()
    tert = np.quantile(rv, [1 / 3, 2 / 3])
    out =[("квинтиль RV_w (PIT)", k, q == k, "PIT", True) for k in sorted(set(q))]
    out += [("спокойный/стресс Г4 (PIT)", "стресс" if v else "спокойно", st == v, "PIT", False) for v in (True, False)]
    out += [("RV цели по факту", "низкая (нижний терциль)", rv <= tert[0], "по факту, не PIT", False), ("RV цели по факту", "высокая (верхний терциль)", rv >= tert[1], "по факту, не PIT", False)]
    out += [("завышение/занижение", "завышение (RV<σ̂²)", rv < s2, "по факту, не PIT", False), ("завышение/занижение", "занижение (RV>σ̂²)", rv > s2, "по факту, не PIT", False)]
    return out


def jump_and_trans(pr: pd.DataFrame, jshare: np.ndarray) -> list:
    """Скачки (по факту: j/rv цели > JUMP_SHARE), окно ±5 дат, после отрицательного дня (прокси), переходы квинтиля."""
    j = jshare > JUMP_SHARE
    near = np.zeros(len(j), bool)
    for i in np.flatnonzero(j):
        near[max(0, i - 5):i + 6] = True
    q = pr["regime"].str.extract(r"q(\d|_nan)")[0]
    qn = pd.to_numeric(q, errors="coerce")
    dq = qn.diff()
    return [("скачки (по факту)", "день скачка", j, "по факту, не PIT", False), ("скачки (по факту)", "±5 дат от скачка", near, "по факту, не PIT", False),
            ("переход режима (PIT)", "квинтиль вверх", (dq > 0).to_numpy(), "PIT", False), ("переход режима (PIT)", "квинтиль вниз", (dq < 0).to_numpy(), "PIT", False),
            ("переход режима (PIT)", "без изменения", (dq == 0).to_numpy(), "PIT", False)]


def regimes(alg: str, pr: pd.DataFrame) -> pd.DataFrame:
    a, _ = ALG[alg]
    dl = pd.read_parquet(common.OUT / "daily_p1.parquet")
    js = (dl["j"] / dl["rv"]).reindex(pr["date"]).to_numpy(float)
    old = pd.read_parquet(common.REP / "PREDICTIONS.parquet")
    neg = (old["f_rs_minus"] > old["f_rs_plus"]).to_numpy()
    sl = regime_masks(pr, a) + jump_and_trans(pr, js) + [("после отрицательного дня d (прокси: rs− > rs+ на d)", "да", neg, "PIT (на d)", False)]
    rv, s2 = pr["RV_actual"].to_numpy(float), pr[a + "_forecast"].to_numpy(float)
    qa, qh = pr["QLIKE_" + a].to_numpy(), pr["QLIKE_HAR"].to_numpy()
    rows = []
    for kind, lab, mk, tag, excl in sl:
        if mk.sum() < 2:
            continue
        rows.append({"kind": kind, "label": lab, "marking": tag, "exclusive_set": excl, **_m(rv[mk], s2[mk], qa[mk], qh[mk]), "share_underforecast": float((rv[mk] > s2[mk]).mean())})
    return pd.DataFrame(rows)


def checks(alg: str, pr: pd.DataFrame, an: pd.DataFrame, rg: pd.DataFrame) -> list:
    """A11: n по исключающему срезу = 2271; взвешенная QLIKE годов = общая (< 1e-12); QLIKE пересчитан из прогнозов."""
    a, _ = ALG[alg]
    yr = an[an.kind == "год"]
    tot = an[an.slice == "весь период"].iloc[0]
    bad = []
    if int(rg[rg.exclusive_set]["n"].sum()) != paths6.N_GRID:
        bad.append("сумма n режимов")
    if abs(float((yr.qlike * yr.n).sum() / yr.n.sum()) - tot.qlike) >= 1e-12 or int(yr.n.sum()) != paths6.N_GRID:
        bad.append("годы != общий")
    if np.abs(np.asarray(metrics.qlike(pr["RV_actual"], pr[a + "_forecast"])) -pr["QLIKE_" + a].to_numpy()).max() >= 1e-12:
        bad.append("QLIKE пересчёт")
    return bad


def stage(args) -> int:
    paths6.mkdirs()
    rc = 0
    for alg in ALG:
        d = paths6.REP / alg
        if args.check:
            pr = pd.read_parquet(d / "PREDICTIONS.parquet")
            bad = checks(alg, pr, paths6.read_csv(d / "ANNUAL_PERFORMANCE.csv"), paths6.read_csv(d / "REGIME_PERFORMANCE.csv"))
            print("A11", alg, "нарушений", len(bad), bad)
            rc |= bool(bad)
            continue
        pr = predictions(alg)
        an, rg = annual(alg, pr), regimes(alg, pr)
        pr.to_parquet(d / "PREDICTIONS.parquet")
        paths6.write_csv(an, d / "ANNUAL_PERFORMANCE.csv", "%s; по годам цели и периодам; база HAR-J(б)" % ALG[alg][1])
        paths6.write_csv(rg, d / "REGIME_PERFORMANCE.csv", "%s; режимы PIT (по данным <= d) и по факту (помечено)" % ALG[alg][1])
        print(alg, "строк", len(pr), "QLIKE", an[an.slice == "весь период"].qlike.iloc[0])
    return int(rc)
