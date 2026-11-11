"""S-A-IMPL-1: проверки I1.G1–I1.G3 (только чтение). Запуск через `python3 -W ignore -m models_v2.s18.a_feat --check | --check-k1 | --check-pit`.
- `check(F=None) -> list`: S(HAR, 1) против rv / rv_w / rv_m (≤ 1e-12 отн.; маска NaN K1 = «позиция в сегменте < длины прямоугольника»), ln S ≤ 1e-12 абс., c = 0 ⇒ RV_b = RV побитно.
- `check_k1(F=None) -> list`: exp(f + s²/2) по коэффициентам K1 против sigma2 на всех строках K1, ≤ 1e-12 отн.
- `check_pit(F=None, n_days=3, seed=0) -> list`: порча баров дней > d и баров d после 18:40 не меняет ln S дней ≤ d при случайных (a, b); mean_present b = 1 (1e-12); a ≥ 0, Σa = 1 (1e-12).
Каждая функция возвращает список замечаний (пусто — ок)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s10 import hp_har as HH
from models_v2.s17.fwd_src import LAST_SESSION, Src
from models_v2.s18 import a_feat as A

TOL = 1e-12


def _all_rows(F):
    rows = np.arange(len(F.dates))
    return rows, A.lag_idx(F, rows)


def check(F=None) -> list:
    F, bad = F or A.build_sber(), []
    rows, lag = _all_rows(F)
    S = np.exp(A.ln_S(F, A.har_theta(), lag).numpy())
    for k, ref in enumerate((F.rv, F.rv_w, F.rv_m)):
        ok = F.pos >= A.RECT[k] - 1
        if not np.array_equal(ok, np.isfinite(ref)):
            bad.append("маска NaN %s не совпала" % A.KS[k])
        rel = float(np.max(np.abs(S[ok, k] / ref[ok] - 1.0)))
        ab = float(np.max(np.abs(np.log(S[ok, k]) - np.log(ref[ok]))))
        print("S[%s]: max отн. %.2e, max |ln S − ln ref| %.2e, строк %d" % (A.KS[k], rel, ab, int(ok.sum())))
        bad += [] if rel <= TOL and ab <= TOL else ["S[%s] ≠ %s: отн. %.2e, абс. ln %.2e" % (A.KS[k], ("rv", "rv_w", "rv_m")[k], rel, ab)]
    rb = A.rv_b_exact(F, np.ones((3, A.NS)))
    same = bool((rb == F.rv[:, None]).all())
    print("c = 0: RV_b == RV побитно: %s (%d дней × 3)" % (same, len(F.rv)))
    return bad + ([] if same else ["c = 0: RV_b ≠ RV побитно"])


def check_k1(F=None) -> list:
    F, k1 = F or A.build_sber(), A.load_k1()
    rows = F.dates.get_indexer(k1["d"])
    if (rows < 0).any():
        return ["дата K1 не в дневном ряду: %d" % int((rows < 0).sum())]
    ln = A.ln_S(F, A.har_theta(), A.lag_idx(F, rows)).numpy()
    f = k1["b0"] + k1["b_d"] * ln[:, 0] + k1["b_w"] * ln[:, 1] + k1["b_m"] * ln[:, 2] + k1["b_J"] * np.log1p(F.j[rows])
    sig = np.exp(f.to_numpy() + k1["s2"].to_numpy() / 2)
    rel = float(np.max(np.abs(sig / k1["sigma2"].to_numpy() - 1.0)))
    n_dm = int((k1["d"] >= "2018-01-03").sum())
    print("K1: строк %d (из них d ≥ 2018-01-03: %d); max отн. |exp(f + s²/2)/sigma2 − 1| = %.2e" % (len(k1), n_dm, rel))
    return [] if rel <= TOL else ["sigma2 K1 не воспроизведена: %.2e" % rel]


def rand_theta(rng) -> dict:
    return dict(omega=torch.as_tensor(rng.uniform(0, 1, 3)), psi=torch.as_tensor(rng.uniform(-8, 8, (3, 2))), c=torch.as_tensor(rng.normal(0, 0.7, (3, 4))))


def corrupt_day(bars: pd.DataFrame, d: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Бары дней > d портятся (`hp_har.corrupt_after`), к дню d добавляются вечерние бары (begin > 18:40) с чужими ценами."""
    out = HH.corrupt_after(bars, d, seed)
    last = out[out["begin"].dt.normalize() == d].tail(1).copy()
    ev = pd.concat([last.assign(begin=d + pd.Timedelta(hours=19 + i), close=last["close"].to_numpy() * (2.0 + i), open=last["open"].to_numpy() * 0.3) for i in range(3)])
    return pd.concat([out, ev], ignore_index=True)


def check_pit(F=None, n_days: int = 3, seed: int = 0) -> list:
    src, rng, bad = Src(), np.random.default_rng(seed), []
    bars, idays = src.bars("SBER", LAST_SESSION), src.index_days(LAST_SESSION)
    F = F or A.build(bars, idays)
    th = rand_theta(rng)
    a_bad = 0
    for k in range(3):
        for p in range(0, len(F.dates), 97):
            ok = (np.arange(A.LAGS[k]) <= F.pos[p])[None, :]
            a = A.lag_weights(ok, k, th["omega"][k], th["psi"][k]).numpy()
            a_bad += int((a < 0).any() or abs(a.sum() - 1.0) > TOL)
    bad += ["a вне симплекса: %d" % a_bad] if a_bad else []
    E = A.slot_exp(th["c"]).numpy()
    mp = (F.pres.astype(float) @ E.T) / F.n[:, None]
    mean_b = np.array([(np.where(F.pres, E[k][None, :] / mp[:, [k]], 0.0).sum(1) / F.n) for k in range(3)])
    print("a: Σ = 1, ≥ 0; max |mean_present b − 1| = %.2e" % float(np.abs(mean_b - 1.0).max()))
    bad += [] if np.abs(mean_b - 1.0).max() <= TOL else ["mean_present b ≠ 1"]
    for d in F.dates[rng.integers(300, len(F.dates) - 300, n_days)]:
        G = A.build(corrupt_day(bars, d, int(rng.integers(1_000_000))), idays)
        r0 = F.dates.get_loc(d)
        rows = np.arange(r0 - 150, r0 + 1)
        x0 = A.ln_S(F, th, A.lag_idx(F, rows)).numpy()
        x1 = A.ln_S(G, th, A.lag_idx(G, rows)).numpy()
        diff = float(np.abs(x0 - x1).max())
        print("PIT d = %s: max |Δ ln S| по 151 дню до d: %.2e" % (d.date(), diff))
        bad += [] if diff == 0.0 else ["порча будущего изменила S до %s: %.2e" % (d.date(), diff)]
    return bad


def main(argv: list) -> int:
    fn = {"--check": check, "--check-k1": check_k1, "--check-pit": check_pit}
    f = next((fn[a] for a in argv if a in fn), None)
    if f is None:
        raise SystemExit("укажите --check, --check-k1 или --check-pit")
    bad = f()
    print("%s: %s" % (argv[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
