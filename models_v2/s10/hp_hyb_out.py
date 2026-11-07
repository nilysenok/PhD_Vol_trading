"""P2: выход гибридов — реестр `ALL_TRIALS.csv` (585 строк Ж-214 без изменений + строки программы), `CHECKPOINT.json` (used), таблицы выбора и времени, описательные метрики по группам и годам.
Файлы общие для модулей XGB и LGBM; запись — под файловой блокировкой (два процесса). Таблицы — описательно, без выбора (выбор финалистов — лист P3 по R9)."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as P

PROGRAM, N_START = "Ж-217", 585
REG_OLD, REG, CP, LOCK = P.ROOT / "docs/otchety/harj_hybrid_v2/ALL_TRIALS.csv", P.REP / "ALL_TRIALS.csv", P.REP / "CHECKPOINT.json", P.OUT / "hyb.lock"
HASHES = {"engine_panel.parquet": "565a479b25033d1e", "k4567.parquet": "2ceb56c1e38df0b2", "panel_base.parquet": "6e2988c6a8615ac6"}


@contextlib.contextmanager
def lock():
    with open(LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def check_hashes() -> list:
    return ["%s: SHA %s != %s" % (n, P.sha16(P.OUT / n), h) for n, h in HASHES.items() if P.sha16(P.OUT / n) != h]


def register(module: str, name: str, arch: str, config: dict) -> bool:
    """Строка реестра конфигурации (одна на (модуль, имя)); повтор — не дублируется. Затем used в CHECKPOINT = число строк программы. -> True, если строка добавлена."""
    with lock():
        if not REG.exists():
            shutil.copyfile(REG_OLD, REG)
        pre = "%s:%s:%s:" % (PROGRAM, module, name)
        rows = REG.read_text(encoding="utf-8").splitlines()
        added = not any(r.startswith(pre) for r in rows)
        if added:
            cfg = {"name": name, "module": module, "arch": arch, "kind": "panel_hybrid", "trial": True, **config}
            js = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
            row = [pre + hashlib.sha256(js.encode()).hexdigest()[:12], PROGRAM, "harj_panel_hybrid", arch, js, "trial"]
            with open(REG, "a", newline="", encoding="utf-8") as f:
                if not REG.read_bytes().endswith(b"\n"):
                    f.write("\n")
                csv.writer(f, lineterminator="\n").writerow(row)
        used = sum(1 for r in REG.read_text(encoding="utf-8").splitlines() if r.startswith(PROGRAM + ":"))
        cp = json.loads(CP.read_text())
        cp.update(leaf="P2", used=used, N_registry=N_START + used, updated=time.strftime("%Y-%m-%d %H:%M"))
        CP.write_text(json.dumps(cp, ensure_ascii=False, indent=1))
        return added


def merge_csv(df: pd.DataFrame, name: str, note: str, module: str) -> None:
    """P2_CHOICE.csv / P2_TIME.csv: строки модуля заменяются, строки другого модуля сохраняются."""
    with lock():
        p = P.REP / name
        old = pd.read_csv(p, comment="#") if p.exists() else pd.DataFrame()
        keep = old[old["module"] != module] if len(old) else old
        P.write_csv(pd.concat([keep, df.assign(module=module)], ignore_index=True), name, note)


def _row_metrics(k: pd.DataFrame, sigs: list) -> pd.DataFrame:
    out = {"ticker": k["ticker"].to_numpy(), "group": k["group"].to_numpy(), "rv": k["rv_tplus"].to_numpy(float)}
    for s in sigs:
        f, rv = k["sigma2_" + s].to_numpy(float), out["rv"]
        out.update({"q_" + s: rv / f - np.log(rv / f) - 1.0, "se_" + s: (rv - f) ** 2, "ae_" + s: np.abs(rv - f), "sl_" + s: (np.log(rv) - np.log(f)) ** 2})
    out["lrv"] = np.log(out["rv"])
    return pd.DataFrame(out)


def _per_ticker(m: pd.DataFrame, sigs: list, period: str) -> pd.DataFrame:
    """Метрики бумаги в периоде: QLIKE, RMSE, MAE по RV; R² по RV и по ln RV (1 − SSE/SST внутри бумаги и периода)."""
    rows = []
    for (t, g), x in m.groupby(["ticker", "group"]):
        sst, sstl = ((x["rv"] - x["rv"].mean()) ** 2).sum(), ((x["lrv"] - x["lrv"].mean()) ** 2).sum()
        for s in sigs:
            rows.append(dict(ticker=t, group=g, period=period, model=s, n=len(x), qlike=x["q_" + s].mean(), rmse=float(np.sqrt(x["se_" + s].mean())), mae=x["ae_" + s].mean(),
                             r2_rv=1 - x["se_" + s].sum() / sst if sst > 0 else np.nan, r2_ln=1 - x["sl_" + s].sum() / sstl if sstl > 0 else np.nan))
    return pd.DataFrame(rows)


def describe(k: pd.DataFrame, sigs: list, first_target: str = P.FIRST_TARGET) -> pd.DataFrame:
    """Описательные метрики на ОБЩИХ датах бумаги (все ряды sigs конечны и > 0), цели с first_target: по бумаге и по группам (среднее по бумагам) × {all, all_no_gap, год цели}."""
    k = k[k["t_plus"] >= pd.Timestamp(first_target)]
    k = k[np.logical_and.reduce([np.isfinite(k["sigma2_" + s]) & (k["sigma2_" + s] > 0) for s in sigs])]
    m = _row_metrics(k, sigs)
    yr = k["t_plus"].dt.year.to_numpy()
    per = [("all", np.ones(len(k), bool)), ("all_no_gap", ~k["gap"].to_numpy(bool))] + [(str(y), yr == y) for y in sorted(set(yr))]
    t = pd.concat([_per_ticker(m[mk], sigs, name) for name, mk in per], ignore_index=True)
    vals = ["qlike", "rmse", "mae", "r2_rv", "r2_ln"]
    rows = [t.assign(scope="ticker", name=t["ticker"], n_tickers=1)]
    for gname, sel in (("M16", t["group"] == "M16"), ("SBER", t["group"] == "SBER"), ("E11", t["group"] == "E11"), ("all28", t["group"].notna())):
        g = t[sel].groupby(["period", "model"])
        a = g[vals].mean().join(g["n"].sum()).join(g["ticker"].nunique().rename("n_tickers")).reset_index()
        rows.append(a.assign(scope="group", name=gname))
    return pd.concat(rows, ignore_index=True)[["scope", "name", "period", "model", "n", "n_tickers"] + vals]
