"""B2: конфигурации (GATES_B2 §3), сборка матрицы признаков, обучение блока (окно / веса / частота / CUSUM), прогнозы."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models_v2.s4 import design, fit, rows, runner
from models_v2.s5.b2 import cusum

BASE_N, I_PROXY, I_IMOEX = 28, 11, 22   # 11 базовых | 11 прокси | 6 IMOEX
DROP = {"no_imoex": range(22, 28), "no_market": range(11, 28), "no_semivar": (7, 8), "no_gap": (9,)}


@dataclass(frozen=True)
class Cfg:
    name: str
    direction: str = "feature_engineering"
    add: tuple = ()          # имена добавленных столбцов F*
    drop: str = ""           # ключ DROP
    window: int = 0          # rolling N пар (0 — expanding)
    hl: int = 0              # полураспад весов, строк (0 — нет)
    freq: str = "d"          # d | w | m
    cusum_h: float = 0.0     # 0 — выкл

    def trial_id(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()[:12]


def window_cfgs() -> list:
    r = "rolling_recency"
    return ([Cfg("w_roll%d" % n, r, window=n) for n in (252, 504, 756, 1008)] + [Cfg("w_weekly", r, freq="w"), Cfg("w_monthly", r, freq="m"),
            Cfg("w_exp1y", r, hl=252), Cfg("w_exp2y", r, hl=504), Cfg("w_cusum_h5", r, cusum_h=5.0), Cfg("w_cusum_h8", r, cusum_h=8.0)])


def make_dz(dz0: design.Dz, feats: pd.DataFrame | None, cfg: Cfg) -> design.Dz:
    """Матрица конфигурации: Б-2 минус DROP[cfg.drop] плюс столбцы cfg.add из feats (уже нормированы)."""
    X = dz0.X
    if cfg.drop:
        keep = [j for j in range(X.shape[1]) if j not in set(DROP[cfg.drop])]
        X = X[:, keep]
    if cfg.add:
        X = np.column_stack([X] + [feats[c].to_numpy(float) for c in cfg.add])
    return design.Dz(X, dz0.y, dz0.off)


def blocks(F, grid, freq: str) -> dict:
    """{T: [d…]}: T — первая дата сетки в дне/неделе (ISO)/месяце; прогноз остальных дат блока — той же моделью."""
    dt = pd.DatetimeIndex(F.dates[grid])
    key = {"d": dt, "w": dt.isocalendar().year.astype(int) * 100 + dt.isocalendar().week.astype(int), "m": dt.year * 100 + dt.month}[freq]
    out, seen = {}, {}
    for d, k in zip(grid, np.asarray(key)):
        T = seen.setdefault(k if freq != "d" else int(d), int(d))
        out.setdefault(T, []).append(int(d))
    return out


def train_pairs(dz, F, cfg: Cfg, T: int, alarm=None) -> np.ndarray:
    idx = runner.pairs_upto(dz, F, T)
    if cfg.window:
        idx = idx[-cfg.window:]
    if cfg.cusum_h:
        idx = cusum.window_pairs(idx, alarm, T)
    return idx


def fit_block(F, dz, cfg: Cfg, hp, T: int, d_idx, alarm=None) -> dict:
    t0 = time.time()
    idx = train_pairs(dz, F, cfg, T, alarm)
    w = fit.weights(np.arange(len(F.rv)), T, cfg.hl) if cfg.hl else None
    m = fit.fit_i1(dz.X, dz.y, idx, hp, w, None)
    return {"f": m["predict"](dz.X[np.asarray(d_idx)]), "k": m["k"], "s2_es": m["s2_es"], "s2_tr": m["s2_tr_prelim"],
            "n_train": m["n_train"], "min_t": int(idx.min()), "max_t": int(idx.max()), "secs": time.time() - t0}


@dataclass
class World:
    F: object
    grid: np.ndarray
    dzs: dict                    # {cfg.name: Dz}
    cfgs: dict                   # {cfg.name: Cfg}
    i2: dict                     # {год: HP}
    alarms: dict = field(default_factory=dict)   # {h: bool[n]}


_W: World | None = None


def _init(w):
    global _W
    _W = w


def _job(a):
    name, T, ds = a
    c = _W.cfgs[name]
    hp = _W.i2[rows.year_of(_W.F, ds[0])]
    return name, T, fit_block(_W.F, _W.dzs[name], c, hp, T, ds, _W.alarms.get(c.cusum_h))


def run_all(w: World, names, procs: int = 4) -> dict:
    """{name: {T: результат блока}} для конфигураций names."""
    jobs = [(n, T, ds) for n in names for T, ds in blocks(w.F, w.grid, w.cfgs[n].freq).items()]
    out = {n: {} for n in names}
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(w,)) as p:
        for n, T, r in p.imap_unordered(_job, jobs, chunksize=8):
            out[n][T] = r
    return out


def forecasts(w: World, name: str, res: dict) -> pd.DataFrame:
    """d, target_date, sigma2 (вариант б), rv по сетке; плюс ID блока T."""
    F, dz = w.F, w.dzs[name]
    rec = []
    for T, r in res.items():
        ds = np.array(blocks(w.F, w.grid, w.cfgs[name].freq)[T])
        rec.append(pd.DataFrame({"i": ds, "sigma2": np.exp(dz.off[ds] + r["f"] + r["s2_es"] / 2), "T": T}))
    df = pd.concat(rec).sort_values("i")
    i = df["i"].to_numpy()
    return pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "sigma2": df["sigma2"].to_numpy(), "rv": F.rv[i + 1], "T": df["T"].to_numpy()})


def block_stats(w: World, name: str, res: dict) -> dict:
    v = list(res.values())
    return {"trees": float(np.mean([r["k"] for r in v])), "mse_train": float(np.mean([r["s2_tr"] for r in v])),
            "mse_block": float(np.mean([r["s2_es"] for r in v])), "n_fits": len(v), "secs": float(sum(r["secs"] for r in v)),
            "n_feat": int(w.dzs[name].X.shape[1])}
