"""P17, G16: выходы — статус строк реестра, `P17_SUMMARY.csv` (описательный QLIKE по группам M16 / SBER / E11 / 17 / 28 и годам, без выбора), сверка с результатами на 28 (должны остаться побитно прежними)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s10 import hp_hyb_out as O, hp_paths as P, hp_p17 as X

ORDER = ["A1", "B1", "C1", "D1", "A2", "A3", "A4", "B2", "C2", "D2"]
HASH28 = {"engine_panel.parquet": "565a479b25033d1e", "k4567.parquet": "2ceb56c1e38df0b2", "panel_base.parquet": "6e2988c6a8615ac6", "hyb_xgb.parquet": "c864e31cfe764676", "hyb_lgbm.parquet": "c5d7cbcb3a4c4948"}
S28 = "обучение на 28 (Ж-220: не результат на 17; в выводы не идёт)"
S17 = "пересчёт на 17 (Ж-220, У-19…У-21; не новая конфигурация)"


def check_28_unchanged() -> list:
    """Результаты на 28 — тот же SHA, что до P17 (G8–G10)."""
    return ["%s: SHA %s != %s (результат на 28 изменён)" % (n, P.sha16(P.OUT / n), h) for n, h in HASH28.items() if P.sha16(P.OUT / n) != h]


def write_status() -> int:
    """`ALL_TRIALS_STATUS.csv`: ключ строки реестра -> статус (строки программы Ж-217, LOTO и p17; прежний реестр не редактируется — только дописывается)."""
    rows = []
    for ln in O.REG.read_text(encoding="utf-8").splitlines():
        key = ln.split(",", 1)[0]
        if key.startswith("Ж-217:") or key.startswith("Ж-217-LOTO:"):
            rows.append((key, S28))
        elif key.startswith("Ж-220-p17:"):
            rows.append((key, S17))
    pd.DataFrame(rows, columns=["key", "status"]).to_csv(P.REP / "ALL_TRIALS_STATUS.csv", index=False, lineterminator="\n")
    return len(rows)


def set_checkpoint(**kw) -> None:
    """CHECKPOINT.json: leaf P17 и сведения p17; `used` не меняется (У-21)."""
    with O.lock():
        cp = json.loads(O.CP.read_text())
        cp.update(leaf="P17", updated=time.strftime("%Y-%m-%d %H:%M"))
        cp["p17"] = {**cp.get("p17", {}), **kw}
        O.CP.write_text(json.dumps(cp, ensure_ascii=False, indent=1))


def load_all() -> pd.DataFrame:
    """Строки панели 28 × K1, K4, K5 и 20 гибридов (xgb:A1 … lgbm:D2) с обучением на 17."""
    k = pd.read_parquet(P.OUT / "k45_p17.parquet")
    for m in ("xgb", "lgbm"):
        h = pd.read_parquet(P.OUT / ("hyb_%s_p17.parquet" % m))
        h = h[["ticker", "date"] + ["sigma2_" + n for n in ORDER]].rename(columns={"sigma2_" + n: "sigma2_%s:%s" % (m, n) for n in ORDER})
        k = k.merge(h, on=["ticker", "date"], how="left")
    return k


def summary(k: pd.DataFrame, first_target: str = P.FIRST_TARGET) -> pd.DataFrame:
    """QLIKE на ОБЩИХ строках (все модели конечны и > 0), цели с first_target. Строки: группа × {all, год цели} × модель — среднее по бумагам группы; n_tickers, n_rows."""
    cols = [c for c in k.columns if c.startswith("sigma2_") and c != "sigma2_k1"] + ["sigma2_k1"]
    k = k[k["t_plus"] >= pd.Timestamp(first_target)]
    k = k[np.logical_and.reduce([np.isfinite(k[c]) & (k[c] > 0) for c in cols])]
    q = pd.DataFrame({c[7:]: metrics.qlike(k["rv_tplus"], k[c]) for c in cols}, index=k.index)
    q["ticker"], q["group"], q["period"] = k["ticker"], k["group"], "all"
    q = pd.concat([q, q.assign(period=k["t_plus"].dt.year.astype(str))], ignore_index=True)
    models = [c[7:] for c in cols]
    per = q.groupby(["group", "ticker", "period"])[models].mean().reset_index()
    cnt = q.groupby(["group", "ticker", "period"]).size().rename("n").reset_index()
    per = per.merge(cnt, on=["group", "ticker", "period"])
    rows = []
    for name, sel in (("M16", per["group"] == "M16"), ("SBER", per["group"] == "SBER"), ("E11", per["group"] == "E11"), ("17", per["group"].isin(["M16", "SBER"])), ("28", per["group"].notna())):
        g = per[sel].groupby("period")
        a = g[models].mean().join(g["n"].sum()).join(g["ticker"].nunique().rename("n_tickers")).reset_index().assign(group=name)
        rows.append(a)
    t = pd.concat(rows, ignore_index=True).melt(id_vars=["group", "period", "n", "n_tickers"], value_vars=models, var_name="model", value_name="qlike")
    return t[["group", "period", "model", "n", "n_tickers", "qlike"]]
