"""Потоки B1: ежедневные прогнозы по сетке, фолды LOYO, эксперты R2. Кэш: parquet + ключ (код + входы), resume по ключу."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.b1 import designs, jobs, world


def _cached(name: str, key: str, build):
    p, k = world.B1_DIR / (name + ".parquet"), world.B1_DIR / (name + ".key")
    if p.exists() and k.exists() and k.read_text() == key:
        return pd.read_parquet(p), True
    df = build()
    df.to_parquet(p)
    k.write_text(key)
    return df, False


def raw_stream(w, specs: dict, name: str, procs: int = 4) -> pd.DataFrame:
    """Строка на дату сетки: row (индекс d в F), f, s2 (ES-блок), k, tr_loss, es_loss, n_train."""
    def build():
        res = jobs.run(w.F, specs, w.i2, w.eps, "block", [(name, int(d)) for d in w.grid], procs)
        r = [res[(name, int(d))] for d in w.grid]
        out = pd.DataFrame(r)
        out.insert(0, "row", w.grid.astype(int))
        return out
    return _cached("raw_" + name, world.key(), build)[0]


def folds(w, specs: dict, name: str, procs: int = 4) -> pd.DataFrame:
    """Фолды LOYO: Y, y, ci (индекс сжатия c в spec.cs), q. HP года Y, обучение — цели < 01.01.Y кроме года y."""
    def build():
        tasks = [(name, Y, y) for Y in range(2017, 2027) for y in range(2015, Y)]
        res = jobs.run(w.F, specs, w.i2, w.eps, "fold", tasks, procs)
        rec = [{"Y": a[1], "y": a[2], "ci": ci, "q": q} for a, qs in res.items() for ci, q in enumerate(qs)]
        return pd.DataFrame(rec).sort_values(["ci", "Y", "y"]).reset_index(drop=True)
    return _cached("loyo_" + name, world.key(), build)[0]


def raw_experts(w, specs: dict, name: str, procs: int = 4) -> pd.DataFrame:
    """R2: по каждой группе k — f_k, s2_k, n_k на дату сетки (NaN — пар < N_MIN)."""
    def build():
        res = jobs.run(w.F, specs, w.i2, w.eps, "experts", [(name, int(d)) for d in w.grid], procs)
        rec = []
        for d in w.grid:
            r = {"row": int(d)}
            for k, (f, s2, n) in enumerate(res[(name, int(d))]):
                r.update({"f%d" % k: f, "s2_%d" % k: s2, "n%d" % k: n})
            rec.append(r)
        return pd.DataFrame(rec)
    return _cached("raw_" + name, world.key(), build)[0]


def frame_of(w, rows_idx, s2) -> pd.DataFrame:
    rows_idx = np.asarray(rows_idx)
    return pd.DataFrame({"d": w.F.dates[rows_idx], "target_date": w.F.dates[rows_idx + 1], "rv": w.F.rv[rows_idx + 1], "sigma2": np.asarray(s2, float)})


def fc_stream(w, spec, raw: pd.DataFrame, c: float) -> tuple:
    """-> (fc, extra): n_nonpos / n_floor (уровневые прогнозы), потери обучение / блок остановки и их отношение."""
    s2, n_np, n_fl = designs.sigma2(spec, raw["row"].to_numpy(), raw["f"].to_numpy(), raw["s2"].to_numpy(), w.eps, c)
    tr, es = float(raw["tr_loss"].mean()), float(raw["es_loss"].mean())
    return frame_of(w, raw["row"].to_numpy(), s2), {"n_nonpos": n_np, "n_floor": n_fl, "loss_train": tr, "loss_es": es, "loss_ratio": es / tr,
                                                     "trees_mean": float(raw["k"].mean()), "n_train_min": int(raw["n_train"].min())}


def expert_s2(w, spec, ex: pd.DataFrame, k: int) -> np.ndarray:
    rows = ex["row"].to_numpy()
    return np.exp(spec.off[rows] + ex["f%d" % k].to_numpy() + ex["s2_%d" % k].to_numpy() / 2)   # NaN, где эксперта нет
