"""Режимы итераций Э1-в (Ж-031): факторный 3³, Моррис, Optuna — вызовы `sweep_drivers` владельца, своего счёта нет.

Цель — настройка 2018 (`setup`), контроль 2019 (`control`) отдельными колонками `ctrl_*`, как в OAT (`sweep_run`). Поиск идёт при
комиссии отбора 0,04 и наполняет ledger; пересчёт (0,035) — те же точки. N в DSR = уникальные пробы ledger при 0,04, накопленные
по стратегии между запусками (`<out>/_ledger/<S>/ledger.json`: OAT + factorial + morris + optuna); DSR — `sd.dsr` по всем пробам.
Условный блок, чьё условие не выполнено, сводится к базе (`fix`): дубль точки пробой не считается. Выбор — по плато, не по пику.
Выход: `S{k}_{mode}_{comm}.csv`, `S{k}_top3.csv` (factorial и optuna, колонка `mode`). seed 20261027.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from fastbt import sweep_drivers as sd
from fastbt.sweep import Sweep, make_evaluator
from fastbt.sweep_run import MAX_JOBS, METR, header, ledger_dir, load, periods
from fastbt.sweep_spec import Spec, pkey

SEED, TOP_K, RADIUS, MIN_NB = 20261027, 3, 1, 3
MODES = ("factorial", "morris", "optuna")


def levels(vals: list, v) -> list:
    """Три узла: база в центре, на краю сетки — три ближайших узла."""
    if len(vals) < 3 or v not in vals:
        raise ValueError(f"факторный 3 уровня: в сетке {vals} нужно ≥ 3 узлов и база {v!r}")
    j = min(max(vals.index(v), 1), len(vals) - 2)
    return vals[j - 1:j + 2]


def pick_params(arg: str, out: Path, name: str, c0: float) -> list:
    """`p1,p2,p3` или `tornado:N` — первые N параметров торнадо OAT при комиссии отбора."""
    if not arg.startswith("tornado:"):
        return [p for p in arg.split(",") if p]
    f = out / f"{name}_oat_tornado_{c0:g}.csv"
    return list(pd.read_csv(f, comment="#")["param"][:int(arg.split(":")[1])])


def forced(exp: dict, keys: list) -> dict:
    """Условия блоков выбранных параметров (подставляются в точку); конфликт условий или условный ключ среди выбранных — отказ."""
    cond = {}
    for k in keys:
        if k not in exp["grid"]:
            raise ValueError(f"параметра {k} нет в сетке")
        for ck, cv in exp["uslovie"].get(k, {}).items():
            if cond.setdefault(ck, cv) != cv or ck in keys:
                raise ValueError(f"условие блока {k}: {ck}={cv} конфликтует с другим выбранным параметром")
    return cond


def make_fix(exp: dict):
    base, usl = exp["base"], exp["uslovie"]

    def fix(p: dict) -> dict:
        q = dict(p)
        for k, cond in usl.items():
            if k in q and any(q.get(ck) != cv for ck, cv in cond.items()):
                q[k] = base[k]
        return q
    return fix


def closure(sw: Sweep, c: float, pre: dict, fix):
    return lambda p, per: sw.evaluate(fix({**pre, **p}), per, c)


def point_cols(sw: Sweep, p: dict, per: tuple, c: float) -> dict:
    t, k = sw.evaluate(p, per[0], c), sw.evaluate(p, per[1], c)
    d = dict(objective=sd.objective_stable(t), sr_d=t["sr_d"], T=t["T"], skew=t["skew"], kurt=t["kurt"],
             **{f"tune_{m}": t[m] for m in METR}, tune_sharpe_2018=t["by_year"][2018])
    return dict(d, **{f"ctrl_{m}": k[m] for m in METR}, ctrl_sharpe_2019=k["by_year"][2019])


def plateau_sparse(rows: list, keys: list, grid: dict) -> list:
    """Плато Optuna: среднее цели по посчитанным точкам в ±RADIUS узлов сетки по каждому ключу (с самой точкой, как у модуля)."""
    ix = [[grid[k].index(r[k]) if r[k] in grid[k] else -99 for k in keys] for r in rows]
    for i, r in enumerate(rows):
        nb = [o["objective"] for j, o in enumerate(rows)
              if all(abs(a - b) <= RADIUS for a, b in zip(ix[i], ix[j])) and o["objective"] == o["objective"]]
        r.update(plateau=sum(nb) / len(nb) if nb else float("nan"), neighbors=len(nb), edge=len(nb) < MIN_NB)
    return sorted(rows, key=lambda r: -(r["plateau"] if r["plateau"] == r["plateau"] else -1e9))


def search(sw, exp, mode, o, per, c0, fix):
    """Поиск при комиссии отбора -> (точки, ключи, сетка уровней, счёт проб модуля)."""
    if mode == "factorial":
        keys = o["params_list"]
        if len(keys) != 3:
            raise ValueError(f"факторный 3³: нужно ровно 3 параметра, дано {keys}")
        pre = forced(exp, keys)
        base = {**exp["base"], **pre}
        grid = {k: levels(list(exp["grid"][k]), base[k]) for k in keys}
        res = sd.factorial_sweep(closure(sw, c0, {}, fix), base, grid, per[0])
        return [{k: r[k] for k in base} for r in res["rows"]], keys, grid, res["n_trials"]
    grid = {k: list(v) for k, v in exp["grid"].items()}
    res = sd.optuna_search(closure(sw, c0, {}, fix), exp["base"], grid, per[0], o["trials"], SEED, n_jobs=o["n_jobs"])
    pts = {pkey(q): q for q in (fix({k: r[k] for k in exp["base"]}) for r in res["rows"])}
    return list(pts.values()), list(grid), grid, res["n_trials"]


def finish(sw, pts, keys, grid, mode, per, c, trials, n_mod) -> list:
    rows = [dict(point_cols(sw, p, per, c), **{k: p[k] for k in keys}, _p=p) for p in pts]
    if mode == "factorial":
        rows = sd.pick_by_plateau(dict(rows=rows, keys=keys, grid=grid), len(rows), RADIUS, MIN_NB)
    else:
        rows = plateau_sparse(rows, keys, grid)
    for i, r in enumerate(rows, 1):
        p = r.pop("_p")
        for k in keys:
            r.pop(k)
        d = sd.dsr(r.pop("sr_d"), trials, r.pop("T"), r.pop("skew"), r.pop("kurt"))["dsr"]
        r.update(rank=i, mode=mode, N=len(trials), n_trials_modul=n_mod, comm_pct=c, dsr=d, _pk=pkey(p),
                 **{f"p.{k}": v for k, v in p.items()})
    return rows


def morris_rows(sw, exp, o, per, c, pre, fix) -> list:
    keys = o["params_list"] or list(exp["grid"])
    res = sd.morris_screen(closure(sw, c, pre, fix), {k: list(exp["grid"][k]) for k in keys}, per[0], o["paths"], SEED)
    return [dict(param=k, **e, n_evaluations=res["n_evaluations"], comm_pct=c) for k, e in res["effects"].items()]


def top3_table(done: dict, comms: tuple) -> pd.DataFrame:
    t = pd.DataFrame(done[comms[0]][:TOP_K])
    for c in comms[1:]:
        m = {r["_pk"]: r for r in done[c]}
        for col in ("tune_sharpe", "ctrl_sharpe"):
            t[f"{col}_c{c:g}"] = [m[k][col] for k in t["_pk"]]
    return t


def write(out: Path, name: str, mode: str, h: str, done: dict, comms: tuple):
    for c, rows in done.items():
        t = pd.DataFrame(rows)
        (out / f"{name}_{mode}_{c:g}.csv").write_text(h + t.drop(columns="_pk", errors="ignore").to_csv(index=False), "utf-8")
    if mode != "morris":
        f, new = out / f"{name}_top3.csv", top3_table(done, comms)
        old = pd.read_csv(f, comment="#") if f.exists() else pd.DataFrame()
        old = old[old["mode"] != mode] if len(old) else old
        f.write_text(h + pd.concat([old, new.drop(columns="_pk")], ignore_index=True).to_csv(index=False), "utf-8")


def run_mode(name: str, setka, mode: str, o: dict, out_dir, evaluator=None) -> dict:
    spec_d, exp, raw = load(setka)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = Spec(**spec_d, out=ledger_dir(out, name))
    sw = Sweep(spec, evaluator or make_evaluator(spec))
    per, comms, fix, n0, t0 = periods(raw), spec.comm_pct, make_fix(exp), len(sw.ledger), time.time()
    o = dict(o, params_list=pick_params(o["params"], out, name, comms[0]) if o.get("params") else [])
    if mode == "morris":
        pre = forced(exp, o["params_list"]) if o["params_list"] else {}
        mr = {c: morris_rows(sw, exp, o, per, c, pre, fix) for c in comms}
        done = {c: [dict(r, N=len(sw.trials_sr())) for r in rows] for c, rows in mr.items()}
    else:
        pts, keys, grid, n_mod = search(sw, exp, mode, o, per, comms[0], fix)
        trials = sw.trials_sr()
        done = {c: finish(sw, pts, keys, grid, mode, per, c, trials, n_mod) for c in comms}
    sw.save()
    h = (header(name, raw, sw.ev, sw) + f"# режим {mode}; seed {SEED}; params {o['params_list'] or 'все'}; "
         f"проб в ledger до {n0}, после {len(sw.ledger)}\n")
    write(out, name, mode, h, done, comms)
    return dict(done=done, ledger=sw.ledger, n_before=n0, secs=time.time() - t0)


def _job(a) -> str:
    path, mode, o, out_dir = a
    m = re.match(r"S\d+", Path(path).name)
    r = run_mode(m.group(0) if m else Path(path).stem, path, mode, o, out_dir)
    return f"{path}: {mode} время {r['secs']:.0f} с, проб в ledger {r['n_before']} -> {len(r['ledger'])}"


def run_cli(ns):
    if not 1 <= ns.jobs <= MAX_JOBS or not 1 <= ns.n_jobs <= MAX_JOBS:
        raise ValueError(f"--jobs и --n-jobs в 1..{MAX_JOBS}")
    if ns.mode == "factorial" and not ns.params:
        raise ValueError("--mode factorial: нужен --params p1,p2,p3 или tornado:3")
    o = dict(params=ns.params, paths=ns.paths, trials=ns.trials, n_jobs=ns.n_jobs)
    with ProcessPoolExecutor(min(ns.jobs, len(ns.yaml))) as ex:
        for line in ex.map(_job, [(p, ns.mode, o, ns.out) for p in ns.yaml]):
            print(line)
