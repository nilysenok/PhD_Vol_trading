"""Запуск OAT (Ж-035, Э1-а′): адаптер оценщика кодера 1 к `sweep_drivers.oat_sweep` владельца + CSV.

evaluate(params, period) замкнут на объект и комиссию; отдаёт sharpe, by_year и остальные колонки строки.
Период — только train 2018–2019 (validation 2020–2021 и ≥ 2022 не читаются: оценщик получает end ≤ 2020-01-01).
Выход на S{k} и комиссию: `S{k}_oat_{comm}.csv` (строка = проба; плато по соседям этого параметра; DSR с N = n_trials)
и `S{k}_oat_tornado_{comm}.csv`. Параллельность — процессы по стратегиям, ≤ 8. `sweep_drivers.py` не правится.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from fastbt import metriki_e as me
from fastbt import sweep_drivers as sd
from fastbt import sweep_stat as ss

MAX_JOBS = 8
COMM_PCT = (0.035, 0.04)
STATUS = "разработка, train 2018–2019"
OUT_DIR = Path("docs/tables/sweep_e1")
PERIODS = ss.Periods()


def load_setka(path_or_dict) -> dict:
    d = path_or_dict if isinstance(path_or_dict, dict) else yaml.safe_load(Path(path_or_dict).read_text("utf-8"))
    if d.get("period", "train") != "train":
        raise ValueError("sweep_run считает только train")
    for n, vals in d["grids"].items():
        if len(set(vals)) != len(vals):
            raise ValueError(f"сетка {n}: повторяющиеся значения")
        if d["base"].get(n) not in list(vals):
            raise ValueError(f"исходная точка {n}={d['base'].get(n)!r} не узел сетки {list(vals)}")
    return d


def make_evaluate(ev, comm_pct: float, sig_cache: dict | None = None):
    """Замыкание evaluate(params, period) для sweep_drivers; повтор точки берётся из памяти (calls считает расчёты)."""
    sig_cache, memo, comm = ({} if sig_cache is None else sig_cache), {}, comm_pct / 100.0

    def evaluate(params: dict, period: str) -> dict:
        if period != "train":
            raise PermissionError("validation и ≥ 2022 закрыты: только train")
        years = PERIODS.years(period)
        key = json.dumps(sorted(params.items()), default=str)
        if key in memo:
            return memo[key]
        sp = {k: v for k, v in params.items() if k in ev.sig_names}
        sk = json.dumps(sorted(sp.items()), default=str)
        if sk not in sig_cache:
            sig_cache[sk] = ev.signals(sp)
        evaluate.calls += 1
        df = ss.year_slice(ev.pnl(sig_cache[sk], params, comm, pd.Timestamp(f"{max(years) + 1}-01-01")), years)
        x = df["excess"].values
        by = ss.yearly_sharpes(df, years)
        row = ss.metrics_row(df, years)
        row.update(by_year=by, sr_d=float(me.sr_dnevnoy(x)), n_days=len(x), skew=float(pd.Series(x).skew()),
                   kurt=float(pd.Series(x).kurt()) + 3.0)
        memo[key] = row
        return row

    evaluate.calls = 0
    return evaluate


def oat_run(base: dict, grids: dict, evaluate) -> dict:
    """`sweep_drivers.oat_sweep` + плато по соседям ±1 узел этого параметра (Шарп) + DSR всех проб (N = n_trials)."""
    res = sd.oat_sweep(evaluate, base, grids, "train")
    rows, trials = res["rows"], [r["sr_d"] for r in res["rows"]]
    sh = {(r["param"], r["value"]): r["sharpe"] for r in rows if r["param"] is not None}
    for r in rows:
        p = r["param"]
        if p is None:
            r["plateau"], r["n_sosedey"] = float("nan"), 0
        else:
            vals = list(grids[p])
            i = vals.index(r["value"])
            s = [sh[(p, vals[j])] for j in (i - 1, i + 1) if 0 <= j < len(vals)]
            r["plateau"], r["n_sosedey"] = (sum(s) / len(s) if s else float("nan")), len(s)
        r.update(N=res["n_trials"], dsr=sd.dsr(r["sr_d"], trials, r["n_days"], r["skew"], r["kurt"])["dsr"])
        for k in ("sr_d", "skew", "kurt"):
            r.pop(k)
    tor = []
    for p, t in res["tornado"].items():
        s = [r["sharpe"] for r in rows if r["param"] == p]
        tor.append(dict(param=p, **t, sharpe_min=min(s), sharpe_max=max(s), sharpe_range=max(s) - min(s)))
    res["rows"], res["tornado"] = pd.DataFrame(rows), pd.DataFrame(tor)
    return res


def code_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "fastbt"], capture_output=True, text=True).stdout.strip()
        return h + ("+правки" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "git?"


def header(name: str, d: dict, ev, n: int) -> str:
    inp = hashlib.sha1(json.dumps([d["obj"], d["base"], {k: list(v) for k, v in d["grids"].items()}], default=str,
                                  sort_keys=True).encode()).hexdigest()[:12]
    return (f"# статус: {STATUS}; объект {name}; {datetime.now():%d.%m.%Y %H:%M}\n"
            f"# код: {code_version()}; evaluator {ev.code_id()}; sweep_drivers владельца без правок\n"
            f"# вход: сетка {inp}; ряд {ev.series_digest()[:12]}; проб (n_trials, N в DSR) {n}\n")


def run_setka(name: str, setka, ev, comms=COMM_PCT, out_dir=OUT_DIR) -> dict:
    """Все комиссии одной стратегии; сигналы считаются один раз. Возвращает {comm: (res, секунды, расчётов)}."""
    d, out, sig, got = load_setka(setka), Path(out_dir), {}, {}
    out.mkdir(parents=True, exist_ok=True)
    for c in comms:
        t0, evaluate = time.time(), make_evaluate(ev, c, sig)
        res = oat_run(d["base"], d["grids"], evaluate)
        h = header(name, d, ev, res["n_trials"])
        (out / f"{name}_oat_{c:g}.csv").write_text(h + res["rows"].assign(comm_pct=c).to_csv(index=False), "utf-8")
        (out / f"{name}_oat_tornado_{c:g}.csv").write_text(h + res["tornado"].assign(comm_pct=c).to_csv(index=False), "utf-8")
        got[c] = (res, time.time() - t0, evaluate.calls)
    return got


def _job(args) -> str:
    path, out_dir, comms = args
    m = re.match(r"S\d+", Path(path).name)
    name = m.group(0) if m else Path(path).stem
    d = load_setka(path)
    from fastbt.sweep_sk import SkEvaluator
    got = run_setka(name, d, SkEvaluator(d["obj"]["sid"], d["obj"].get("tickers")), comms, out_dir)
    return "; ".join(f"{name} {c}%: проб {r['n_trials']}, расчётов {n}, {t:.1f} с" for c, (r, t, n) in got.items())


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("yaml", nargs="+")
    a.add_argument("--out", default=str(OUT_DIR))
    a.add_argument("--comm", type=float, nargs="+", default=list(COMM_PCT), help="комиссии, %% на сторону")
    a.add_argument("--jobs", type=int, default=MAX_JOBS)
    ns = a.parse_args(argv)
    if not 1 <= ns.jobs <= MAX_JOBS:
        raise ValueError(f"--jobs в 1..{MAX_JOBS}")
    with ProcessPoolExecutor(min(ns.jobs, len(ns.yaml))) as ex:
        for line in ex.map(_job, [(p, ns.out, tuple(ns.comm)) for p in ns.yaml]):
            print(line)


if __name__ == "__main__":
    main()
