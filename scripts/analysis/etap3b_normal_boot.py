#!/usr/bin/env python3
"""etap3b п.4: Жарк-Бера, асимметрия, эксцесс МСП A/B/C/D; парный бутстрап
dСортино, dOmega (C, D против A), контроль dШарп C vs A; бутстрап dQLIKE
гибрид V1 против HAR-J (пул, h=1/5/22). Схема бутстрапа как в
moex_volatility/scripts/fix_all_bootstrap_ce.py: seed 42, iid randint(0,n,n),
одни индексы для обеих серий, 10 000 реплик, p = доля реплик с дельтой <= 0.
Без scipy: p(JB) = exp(-JB/2) (хи-квадрат, 2 степени свободы).
Результат печатается и пишется в data/out/etap3b_normal_boot.json.
"""
import json
import warnings
from math import exp, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"
DR = ROOT / "results/final/strategies/msp/perstrat_daily_returns"
WF = ROOT / "results/final/data/predictions_walkforward"
HALT = ("2022-02-28", "2022-03-24")
N_BOOT, BPY = 10_000, 252


def sharpe(x):
    return x.mean() * sqrt(BPY) / x.std(ddof=1)


def sortino(x):  # полуотклонение от 0: sqrt(mean(min(x,0)^2))
    return x.mean() * sqrt(BPY) / np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))


def omega(x):  # порог 0
    return x[x > 0].sum() / -x[x < 0].sum()


def moments(x):
    n, z = len(x), x - x.mean()
    m2, m3, m4 = (z**2).mean(), (z**3).mean(), (z**4).mean()
    sk, ku = m3 / m2**1.5, m4 / m2**2
    jb = n / 6 * (sk**2 + (ku - 3) ** 2 / 4)
    return dict(n=n, skew=sk, kurt=ku, excess=ku - 3, jb=jb, p=exp(-jb / 2))


def boot_pair(f, a, b, seed=42):
    n, rng = len(a), np.random.RandomState(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.randint(0, n, n)
        d[i] = f(a[idx]) - f(b[idx])
    return dict(obs=f(a) - f(b), p=float(np.mean(d <= 0)),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


def msp_block():
    ser = {k: pd.read_csv(DR / f"MSP_{k}_inst.csv", parse_dates=["date"]).set_index("date") for k in "ABCD"}
    common = ser["A"].index
    for k in "BCD":
        common = common.intersection(ser[k].index)
    halt = int(common.to_series().between(*HALT).sum())
    x = {k: ser[k].loc[common, "excess_ret"].values for k in "ABCD"}
    out = dict(n_days=len(common), halt_days_in_series=halt, first=str(common[0].date()),
               last=str(common[-1].date()))
    out["sharpe"] = {k: sharpe(v) for k, v in x.items()}
    out["moments"] = {k: moments(v) for k, v in x.items()}
    top = int(np.argmax(np.abs(x["A"])))  # оговорка: день-выброс (22.02.2022) убран
    out["top_day"] = str(common[top].date())
    out["moments_ex_top"] = {k: moments(np.delete(v, top)) for k, v in x.items()}
    out["control_sharpe_CvA"] = boot_pair(sharpe, x["C"], x["A"])
    for name, f in (("sortino", sortino), ("omega", omega)):
        out[name] = {k: f(x[k]) for k in "ABCD"}
        for k in "CD":
            out[f"d_{name}_{k}vA"] = boot_pair(f, x[k], x["A"])
    return out


def load(name, h):
    d = pd.read_parquet(WF / f"{name}_h{h}_annual.parquet")
    d = d[(d.rv_actual > 0) & ~d.date.between(*HALT)]
    p = d.rv_pred.clip(lower=1e-10)
    d = d.assign(q=d.rv_actual / p - np.log(d.rv_actual / p) - 1)
    return d.set_index(["date", "ticker"]).q


def qlike_block():
    out = {}
    for h in (1, 5, 22):
        a, b = load("har", h), load("hybrid_adaptive", h)
        j = pd.concat([a, b], axis=1, keys=["har", "hyb"]).dropna()
        da = (j.har - j.hyb).values  # >0: гибрид лучше
        n, rng = len(da), np.random.RandomState(42)
        d = np.array([da[rng.randint(0, n, n)].mean() for _ in range(N_BOOT)])
        # вторая схема (оговорка): блоки по датам, все тикеры дня вместе
        g = j.assign(dd=da).groupby(level="date").dd.agg(["sum", "count"])
        s, c, m = g["sum"].values, g["count"].values, len(g)
        rng2 = np.random.RandomState(42)
        d2 = np.empty(N_BOOT)
        for i in range(N_BOOT):
            k = rng2.randint(0, m, m)
            d2[i] = s[k].sum() / c[k].sum()
        out[h] = dict(n=n, har=float(j.har.mean()), hyb=float(j.hyb.mean()), diff=float(da.mean()),
                      p_iid=float(np.mean(d <= 0)), lo=float(np.percentile(d, 2.5)),
                      hi=float(np.percentile(d, 97.5)), p_dates=float(np.mean(d2 <= 0)), n_dates=m)
    return out


def main():
    res = dict(msp=msp_block(), qlike=qlike_block())
    ctrl = res["msp"]["control_sharpe_CvA"]
    print("КОНТРОЛЬ dШарп C vs A:", ctrl)
    ok = abs(ctrl["obs"] - 1.19) < 0.01 and ctrl["p"] < 0.001
    res["control_ok"] = bool(ok)
    out = Path(__file__).resolve().parents[2] / "data/out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "etap3b_normal_boot.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if not ok:
        raise SystemExit("СТОП: контроль не прошёл")


if __name__ == "__main__":
    main()
