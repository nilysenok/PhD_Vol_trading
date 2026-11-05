"""B2: строки таблиц (GATES_B2 §5), правило «перспективно», реестр trials B2 (только дописывание)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import dm
from models_v2.s5 import common, registry

FEB24 = pd.Timestamp("2022-02-24")
CHEAP = 0.0010
TRIALS_B2 = common.REP / "OPTUNA_TRIALS_B2.csv"


def masks(td: pd.Series) -> dict:
    m = common.period_masks(td)
    t = pd.Series(td).reset_index(drop=True)
    return {"all": m["all"], "p2017_2021": m["p2017_2021"], "y2022": m["y2022"], "p2023_2025": m["p2023_2025"], "y2026": m["y2026"],
            "no_20220224": (t != FEB24).to_numpy(), "no_2022": ~m["y2022"]}


def crit_wf(fc: pd.DataFrame) -> float:
    """Среднее годовых средних QLIKE(б) по полным годам цели 2017–2025."""
    q = pd.Series(metrics.qlike(fc["rv"], fc["sigma2"]), index=pd.DatetimeIndex(fc["target_date"]).year)
    a = q.groupby(level=0).mean()
    return float(a.loc[2017:2025].mean())


def win_years(fc, har) -> tuple:
    yr = pd.DatetimeIndex(fc["target_date"]).year
    qm, qh = (pd.Series(metrics.qlike(x["rv"], x["sigma2"]), index=yr).groupby(level=0) for x in (fc, har))
    n = qm.size()
    ok = n[n >= 30].index
    return float((qm.mean()[ok] < qh.mean()[ok]).mean()), len(ok)


def row(name: str, fc: pd.DataFrame, har: pd.DataFrame, b2: pd.DataFrame, st: dict, meta: dict) -> dict:
    """fc, har, b2 — d/target_date/sigma2/rv на одной сетке 2271. meta: trial_id, direction, crit_loyo, crit_loyo_b2, crit_wf_b2."""
    for o in (har, b2):
        if not (fc["d"].to_numpy() == o["d"].to_numpy()).all() or len(fc) != 2271:
            raise ValueError("сетка строки %s не равна контролю" % name)
    mk, out = masks(fc["target_date"]), {"model": name, "trial_id": meta["trial_id"], "direction": meta["direction"]}
    q = lambda x, k: float(metrics.qlike(x["rv"][mk[k]], x["sigma2"][mk[k]]).mean())  # noqa: E731
    for k in mk:
        out["qlike_" + k] = q(fc, k)
    for k in ("p2017_2021", "y2022", "p2023_2025"):
        out["d_b2_" + k] = out["qlike_" + k] - q(b2, k)
    s = metrics.summary(fc["rv"], fc["sigma2"])
    out.update(mean_r=s["mean_r"], rmse_ln=s["rmse_ln"], mae_ln=s["mae_ln"], n=s["n"])
    out.update(mse_train=st["mse_train"], mse_block=st["mse_block"], mse_ratio=st["mse_block"] / st["mse_train"], trees=st["trees"],
               n_feat=st["n_feat"], n_fits=st["n_fits"], secs=st["secs"])
    for lab, o in (("harj", har), ("b2", b2)):
        d = dm.dm(fc["rv"], o["sigma2"], fc["sigma2"])   # плюс = модель лучше; в таблице — loss(модель) − loss(база)
        out["dloss_%s_mean" % lab], out["dloss_%s_dm_p" % lab] = (-d["mean_d"], d["p"])
    nf = mk["no_20220224"]
    out["dloss_harj_no_20220224"] = float((metrics.qlike(fc["rv"][nf], fc["sigma2"][nf]) - metrics.qlike(har["rv"][nf], har["sigma2"][nf])).mean())
    out["win_years_harj"], out["years_counted"] = win_years(fc, har)
    out["crit_wf"], out["crit_loyo"] = crit_wf(fc), meta["crit_loyo"]
    out["perspective"], out["crit_kind"] = promising(out, meta, name)
    out["cheap_ok"] = bool(out["qlike_all"] - q(b2, "all") <= CHEAP) if name in ("w_weekly", "w_monthly") else ""
    return out


def promising(r: dict, meta: dict, name: str = "") -> tuple:
    """GATES_B2 §5: crit строки < crit Б-2 того же вида И разность к HAR-J(б) без 24.02.2022 <= 0."""
    use_loyo = bool(np.isfinite(r["crit_loyo"]))
    a, b = (r["crit_loyo"], meta["crit_loyo_b2"]) if use_loyo else (r["crit_wf"], meta["crit_wf_b2"])
    return bool(a < b and r["dloss_harj_no_20220224"] <= 0), "loyo" if use_loyo else "wf"


def trial_row(cfg, r: dict, secs: float, wave: str = "B") -> dict:
    keep = ("qlike_all", "qlike_p2017_2021", "qlike_y2022", "qlike_p2023_2025", "dloss_harj_no_20220224")
    return {"trial_id": cfg.trial_id(), "волна": wave, "направление": cfg.direction, "конфигурация": repr(cfg),
            "метрика_внутр_валидации": "crit_loyo=%s;crit_wf=%.6f" % (r["crit_loyo"], r["crit_wf"]),
            "метрики_описания": ";".join("%s=%.6f" % (k, r[k]) for k in keep), "время": "%.1f" % secs, "seed": 42, "повторный_seed": ""}


def register(rows: list, path: Path = TRIALS_B2) -> int:
    """Дописывает только новые trial_id; возвращает число добавленных."""
    old = pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=registry.TRIAL_COLS)
    new = [r for r in rows if r["trial_id"] not in set(old["trial_id"])]
    if new:
        pd.concat([old, pd.DataFrame(new)], ignore_index=True)[list(registry.TRIAL_COLS)].to_csv(path, index=False)
    return len(new)


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M")
