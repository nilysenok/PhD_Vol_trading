"""S-A-IMPL-3: проверки I3.G3, I3.G4 (только чтение; `--check` печатает sha256 выходов до и после). Запуск — `python3 -W ignore -m models_v2.s18.a_walk --check | --check-pit`.
- `check_pit(var="A2", n_months=3, seed=0) -> list`: для случайных месяцев D (seed 0 ансамбля): порча RV пар с t⁺ > D (×7) и баров дней после последнего дня месяца (плюс вечерние бары последнего дня) не меняет θ и ln H_A, ln C_A всех дней месяца (побитно).
- `check() -> list`: 4 parquet в `data/regen/harj_struct/A/`: 2198 дат = окно M, число переоценок = числу месяцев D_M с датами (106: в D_M нет дат 2022-03; спецификация считала 107 — отступление), λ из сетки, F и X1_A конечны и > 0, у A0 ω = ψ = c = 0; seeds A0 побитно одинаковы; A2p при перестановке = id ≡ A2 побитно (θ и ln C_A месяца); пересчёт одного месяца (все seeds) совпадает с parquet; sha256 выходов до и после равны."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from models_v2.s17.fwd_src import LAST_SESSION, Src
from models_v2.s18 import a_feat as A
from models_v2.s18 import a_feat_chk as FC
from models_v2.s18 import a_layers as L
from models_v2.s18 import a_model as M
from models_v2.s18 import a_walk as W


def _tree() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(W.A_DIR.glob("*.parquet"))} if W.A_DIR.is_dir() else {}


def _month(x, k: int):
    xd = pd.DatetimeIndex(x["d"])
    pos = W.refit_positions(xd, W.FIRST)
    return pos, pos[k], (pos + [len(xd)])[k + 1]


def check_pit(var: str = "A2", n_months: int = 3, seed: int = 0) -> list:
    rng, bad, src = np.random.default_rng(seed), [], Src()
    bars, idays = src.bars("SBER", LAST_SESSION), src.index_days(LAST_SESSION)
    x, _ = L.inputs()
    k1, cfg, bmix = M.load_inputs()
    D = M.build_data(A.build(bars, idays), k1, cfg, bmix)
    for k in rng.choice(np.arange(3, 100), n_months, replace=False):
        pos, a, b = _month(x, int(k))
        D0, d_last = x["d"].iloc[a], x["d"].iloc[b - 1]
        ref = W.walk_variant(var, D, x, positions={a}, seeds=(0,))[0][2]
        D2 = M.build_data(A.build(FC.corrupt_day(bars, d_last, int(rng.integers(1_000_000))), idays), k1, cfg, bmix)
        D2.rv_next = np.where(D2.tp > np.datetime64(D0), D2.rv_next * 7.0, D2.rv_next)
        new = W.walk_variant(var, D2, x, positions={a}, seeds=(0,))[0][2]
        same = all(np.array_equal(ref[q], new[q]) for q in ("theta", "lnH", "lnC"))
        print("PIT %s месяц D = %s (дней %d): θ, ln H_A, ln C_A побитно равны: %s" % (var, D0.date(), b - a, same))
        bad += [] if same else ["порча будущего изменила переоценку %s" % D0.date()]
    return bad


def _identity_perm(x, D) -> list:
    pos, a, b = _month(x, 12)
    dp = W._day_pos(D, x)
    pool = W.closed_rows(D.tp, x["d"].iloc[a])
    old, r = A.PERM, {}
    try:
        A.PERM = np.arange(A.NS)
        for v in ("A2", "A2p"):
            r[v] = W.fit_month(D, pool, dp[a:b], v, seeds=(0,))
    finally:
        A.PERM = old
    same = all(np.array_equal(r["A2"][q], r["A2p"][q]) for q in ("theta", "lnH", "lnC"))
    print("A2p при перестановке = id ≡ A2 побитно (месяц %s): %s" % (x["d"].iloc[a].date(), same))
    return [] if same else ["A2p при π = id не равен A2"]


def _repro(x, D, outs: dict) -> list:
    bad, k = [], 60
    pos, a, b = _month(x, k)
    dp = W._day_pos(D, x)
    for v, df in outs.items():
        r = W.fit_month(D, W.closed_rows(D.tp, x["d"].iloc[a]), dp[a:b], v)
        err = float(np.max(np.abs(np.exp(r["lnC"]) / df["C_A"].to_numpy()[a:b] - 1.0)))
        print("пересчёт месяца %s, %s: max отн. |C_A/parquet − 1| = %.2e" % (x["d"].iloc[a].date(), v, err))
        bad += [] if err <= 1e-10 else ["%s: пересчёт месяца ≠ parquet (%.2e)" % (v, err)]
    return bad


def check() -> list:
    h0, bad, outs = _tree(), [], {}
    x, _ = L.inputs()
    D = M.load_data()
    for v, f in W.OUTS.items():
        p = W.A_DIR / f
        if not p.exists():
            bad.append("нет %s" % f)
            continue
        df = outs[v] = pd.read_parquet(p)
        ok = len(df) == 2198 and (df["d"].to_numpy() == x["d"].to_numpy()).all() and df["refit_date"].nunique() == len(W.refit_positions(pd.DatetimeIndex(x["d"]), W.FIRST))
        ok = ok and df["lam"].isin(W.LAMS).all() and np.isfinite(df[["F", "X1_A", "C_A", "H_A"]].to_numpy()).all() and (df[["F", "X1_A"]] > 0).all().all()
        ok = ok and (v != "A0" or (df[W.NAMES[5:26]] == 0).all().all())
        print("%s: строк %d, переоценок %d, λ %s: %s" % (v, len(df), df["refit_date"].nunique(), sorted(df["lam"].unique()), "ok" if ok else "FAIL"))
        bad += [] if ok else ["%s: состав выхода" % v]
    V = D.view(np.arange(120, 500))
    s0, s1 = (M.fit(V, "A0", 1e-3, s, True) for s in (0, 1))
    print("A0: seeds 0 и 1 побитно равны: %s" % np.array_equal(s0, s1))
    bad += [] if np.array_equal(s0, s1) else ["A0: seeds различаются"]
    bad += _identity_perm(x, D) + (_repro(x, D, outs) if outs else [])
    h1 = _tree()
    print("sha256 выходов до/после равны: %s (%d файлов)" % (h0 == h1, len(h1)))
    return bad + ([] if h0 == h1 else ["--check изменил выходы"])
