"""Ж-068: децили силы по сигналам «голой идеи» (tolkovaniya.md, [Ж-068]). Второй расчёт к (а) Ж-066.
Сигнал = тикер × день d, где условие входа Sk выполнено при снятых фильтрах: движок сигналов (`get_family(sid).generate`)
вызывается на копии `ind` с нейтральными входами фильтров (формулы условия не копируются, модули не меняются).
Доходность = log(close[d+τ]/close[d])·сторона, τ — торговых дней, без комиссии/размера/стопов; d+τ ≤ 2019-12-31.
Диагностика, не выбор: train 2018–2019, без дивидендов, до аудита. Запуск: FASTBT_MAIN=<копия> python3 -m fastbt.diag_zh068"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tables/diag/zh068_a_decili_signaly.csv"
STATUS = "диагностика, train 2018–2019, без дивидендов, до аудита; голая идея: фильтры сняты, без комиссии/размера/стопов; в ledger и реестр отбора не добавлено"
START, END = np.datetime64("2018-01-01"), np.datetime64("2019-12-31")
TAU_S5 = 5          # у S5 в точке tau = net; берём h = 5 — как Evaluator при R = net (etap1_eval.py `_rows`)
NOISE, SAT = 10, 1.0 - 1e-12
NAMES = ["S1", "S2", "S3", "S4", "S5", "S6"]


def fwd_ret(close, d: int, tau: int) -> float:
    """Лог-доходность окна (d, d+τ]: от close d до close d+τ."""
    return float(np.log(close[d + tau] / close[d]))


def signed(r: float, side: int) -> float:
    return r * side


def in_window(d: int, tau: int, last: int) -> bool:
    """Хвост: d+τ не позже последнего дня train — без заглядывания в 2020."""
    return d + tau <= last


def observations(sig, strength, close, dates, tau: int) -> list:
    """[(дата d, сторона, сила, доходность)] по дням с sig != 0, d ≥ 2018-01-01, d+τ ≤ 2019-12-31."""
    last = int(np.searchsorted(dates, END, side="right")) - 1
    out = []
    for d in np.flatnonzero(sig):
        if dates[d] < START or not in_window(int(d), tau, last):
            continue
        s = int(np.sign(sig[d]))
        out.append((dates[d], s, float(strength[d]), signed(fwd_ret(close, int(d), tau), s)))
    return out


def nw_t(x, lag: int) -> float:
    """t среднего с поправкой Newey–West (Бартлетт, лаг `lag`); ряд — в порядке следования. n < 2 или var ≤ 0 — nan."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    v = float((e * e).sum()) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * float((e[k:] * e[:-k]).sum()) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else float("nan")


def _row(label, obs, lag, flag):
    r = [o[3] for o in obs]
    n = len(r)
    return [label, n, min((o[2] for o in obs), default=float("nan")), max((o[2] for o in obs), default=float("nan")),
            100 * float(np.mean(r)) if n else float("nan"), sum(x > 0 for x in r) / n if n else float("nan"),
            nw_t(r, lag), flag]


def deciles(obs: list, tau: int) -> list:
    """Децили по рангу силы среди несатурированных (сила < 1); сила = 1 — отдельная строка «насыщ.». Внутри строки —
    порядок (дата, сторона); лаг NW = τ−1. Метки: «шум» при n < 10, «ничья по силе» при общей границе силы."""
    obs = sorted(obs, key=lambda o: (o[0], o[1]))
    sat = [o for o in obs if o[2] >= SAT]
    ok = [o for o in obs if o[2] < SAT]
    order = sorted(range(len(ok)), key=lambda i: (ok[i][2], i))
    grp = {k: [] for k in range(1, 11)}
    for r, i in enumerate(order):
        grp[10 * r // len(ok) + 1].append(ok[i])
    rows = [_row(str(k), sorted(g, key=lambda o: (o[0], o[1])), tau - 1, ["шум"] if len(g) < NOISE else [])
            for k, g in grp.items()]
    for a, b in zip(rows, rows[1:]):
        if a[1] and b[1] and a[3] == b[2]:
            a[7].append("ничья по силе"), b[7].append("ничья по силе")
    rows.append(_row("насыщ.", sat, tau - 1, ["шум"] if len(sat) < NOISE else []))
    return [r[:7] + ["; ".join(r[7])] for r in rows]


def neutral_ind(sid: str, ind: dict) -> dict:
    """Копия `ind` с входами фильтров, при которых они пропускают всё (S1/S2: vol_regime, свечи, объём, vf_pass;
    S3/S4: ADX > 20, vf_pass, у S3 ещё vol/SMA20 > 1; S5/S6 filter_set 1: ADX < 35, vf_pass). Силы это не касается."""
    x, n = dict(ind), len(ind["vf_pass"])
    x["vf_pass"] = np.ones(n, bool)
    if sid in ("S1", "S2"):
        x.update(vol_regime=np.zeros(n), red_count5=np.full(n, 5.0), green_count5=np.full(n, 5.0), vol_sma5=np.full(n, np.nan))
    elif sid in ("S3", "S4"):
        x["adx14"] = np.full(n, 100.0)
        if sid == "S3":
            x["vol_sma20"] = np.full(n, np.nan)             # ok = False -> vol/1.0 > 1
    else:
        x["adx14"] = np.zeros(n)
    return x


def tau_of(pt: dict) -> int:
    return TAU_S5 if pt["tau"] == "net" else int(pt["tau"])


def run_sid(sid: str) -> dict:
    from fastbt import etap1
    from fastbt.data import TICKERS
    from fastbt.diag_zh066_run import point
    from fastbt.indicators import compute_base
    from fastbt.signals.interface import get_family
    from fastbt.sigprov import SIG_GRIDS
    pt = point(sid)
    sp, tau = {k: pt[k] for k in SIG_GRIDS[sid]}, tau_of(pt)
    obs, obs_f, dig = [], 0, hashlib.sha1()
    for t in TICKERS:
        td = etap1.load_ticker_nf(t)
        k = int(np.searchsorted(td.dates, END, side="right"))            # режем до 2019-12-31 сразу
        td = dataclasses.replace(td, **{f.name: getattr(td, f.name)[:k] for f in dataclasses.fields(td)
                                        if f.name not in ("ticker", "sigma")})
        dig.update(td.digest().encode())
        ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
        fam = get_family(sid)
        base, bare = fam.generate(td, ind, [dict(sp)]), fam.generate(td, neutral_ind(sid, ind), [dict(sp)])
        assert not ((base.sig[0] != 0) & (bare.sig[0] == 0)).any(), "сигнал с фильтрами не вошёл в голую идею"
        obs_f += len(observations(base.sig[0], base.strength[0], td.close, td.dates, tau))
        obs += observations(bare.sig[0], base.strength[0], td.close, td.dates, tau)
        assert all(math.isfinite(o[3]) for o in obs), t
    return dict(sid=sid, tau=tau, obs=obs, n_filtered=obs_f, digest=dig.hexdigest()[:16])


def main() -> None:
    from fastbt.diag_zh066_run import journal
    import csv
    rows, info = [], {}
    for sid in NAMES:
        r = run_sid(sid)
        ntr = len(journal(sid))
        assert len(r["obs"]) >= ntr, (sid, len(r["obs"]), ntr)           # G3
        rows += [[sid, r["tau"], *x] for x in deciles(r["obs"], r["tau"])]
        info[sid] = (r["tau"], len(r["obs"]), ntr, r["n_filtered"], r["digest"])
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# статус: {STATUS}\n# коммит кода: {code}\n# хэш входов: " + "; ".join(f"{s}:{v[4]}" for s, v in info.items()) + "\n")
        w = csv.writer(fh)
        w.writerow(["S", "tau", "дециль", "n", "сила_мин", "сила_макс", "доходность_ср_%", "доля_плюс", "t_NW", "метка"])
        for r in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, float) else x for x in r])
    for s, v in info.items():
        print(s, "tau", v[0], "сигналов (голая идея)", v[1], "сделок (а)", v[2], "сигналов с фильтрами", v[3])


if __name__ == "__main__":
    main()
