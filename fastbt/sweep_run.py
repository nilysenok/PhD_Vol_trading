"""Запуск OAT (Ж-035/Ж-036, Э1-а′): сетка Э1-б -> `sweep_setka.expand` -> `sweep_drivers.oat_sweep` владельца на оценщике.

Цель OAT — настройка 2018 (`setup`, Ж-031); контроль 2019 (`control`) в цель не входит и выводится отдельной группой колонок
`ctrl_*`; метрики цели — `tune_*`. Validation и ≥ 2022 не читаются (оценщик закрывает). Комиссии — из сетки: отбор 0,04 %
(первая, наполняет ledger проб), пересчёт 0,035 %. N в DSR = число уникальных проб в ledger оценщика при комиссии отбора;
`n_trials_modul` — счёт модуля владельца (база в каждой группе повторно), только справка. Процессы — по стратегиям, ≤ 6.
Выход: `S{k}_oat_{comm}.csv` и `S{k}_oat_tornado_{comm}.csv` в docs/tables/sweep_e1/ с шапкой статуса.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from fastbt import sweep_drivers as sd
from fastbt import sweep_setka as sk
from fastbt.sweep import Sweep, make_evaluator
from fastbt.sweep_spec import Spec, pkey

MAX_JOBS = 6                       # Ж-036: у T1 не больше 6 процессов
STATUS = "разработка, цель tune 2018, контроль ctrl 2019 (вне цели)"
OUT_DIR = Path("docs/tables/sweep_e1")
PERIOD_MAP = {"tune2018": "setup", "ctrl2019": "control"}   # имена периодов сетки #305 -> имена оценщика


def periods(s: dict) -> tuple:
    """(цель, контроль) оценщика по `period` сетки; validation и прочее — отказ."""
    p = s.get("period", {"tsel": "tune2018", "kontrol": "ctrl2019"})
    if p["tsel"] not in PERIOD_MAP or p["kontrol"] not in PERIOD_MAP or p.get("validation", "zakryto") != "zakryto":
        raise ValueError(f"период сетки не tune2018/ctrl2019 при закрытой validation: {p}")
    return PERIOD_MAP[p["tsel"]], PERIOD_MAP[p["kontrol"]]


METR = ("sharpe", "msr", "sortino", "mdd", "oborot_y", "dni_v_rynke", "sdelok_y")


def load(path_or_dict) -> tuple:
    """Сетка Э1-б -> (spec_dict, развёртка). Комиссии: отбор первой, затем пересчёт (в % на сторону)."""
    s = path_or_dict if isinstance(path_or_dict, dict) else yaml.safe_load(Path(path_or_dict).read_text("utf-8"))
    k = s.get("komissiya", {"otbor": 0.0004, "pereschet": [0.00035]})
    comm = tuple(round(c * 100, 6) for c in [k["otbor"], *k["pereschet"]])
    exp = sk.expand(s)
    return dict(obj=dict(kind="Sk", sid=s["obekt"]["sid"]), base=exp["base"], comm_pct=comm), exp, s


def groups(exp: dict) -> list:
    """[(база, {ключ: сетка})]: безусловные параметры вместе, условный блок — со своим условием (`oat_base`)."""
    free = {k: v for k, v in exp["grid"].items() if k not in exp["uslovie"]}
    out = [(dict(exp["base"]), free)] if free else []
    return out + [(sk.oat_base(exp, k), {k: v}) for k, v in exp["grid"].items() if k in exp["uslovie"]]


def oat_rows(sw: Sweep, exp: dict, comm: float, per: tuple) -> tuple:
    """OAT на периоде цели per[0] при комиссии comm (%): (строки без DSR, n_trials модуля); per[1] — контроль, вне цели."""
    rows, n_mod = [], 0
    for gi, (base, grid) in enumerate(groups(exp)):
        res = sd.oat_sweep(lambda p, per: sw.evaluate(p, per, comm), base, grid, per[0])
        n_mod += res["n_trials"]
        for r in res["rows"]:
            if r["param"] is None and gi > 0 and base == exp["base"]:
                continue
            p = {k: r[k] for k in base}      # ровно переданные оценщику ключи (с условием блока) — тот же pkey, что в ledger
            row = dict(param=r["param"] or "base", group=gi, value=r["value"], objective=r["objective"], sr_d=r["sr_d"],
                       T=r["T"], skew=r["skew"], kurt=r["kurt"], **{f"tune_{m}": r[m] for m in METR},
                       tune_sharpe_2018=r["sharpe_2018"])
            c = sw.evaluate(p, per[1], comm)
            row.update({f"ctrl_{m}": c[m] for m in METR}, ctrl_sharpe_2019=c["by_year"][2019])
            rows.append(dict(row, **{f"p.{k}": v for k, v in p.items()}))
    return rows, n_mod


def finish(rows: list, exp: dict, trials: list, n_mod: int, comm: float) -> tuple:
    """Плато (±1 узел этого параметра, Шарп tune), DSR(N = len(trials)), торнадо по цели."""
    sh = {(r["param"], r["value"]): r["tune_sharpe"] for r in rows if r["param"] != "base"}
    for r in rows:
        if r["param"] == "base":
            r["plateau"], r["n_sosedey"] = float("nan"), 0
        else:
            vals = list(exp["grid"][r["param"]])
            i = vals.index(r["value"])
            s = [sh[(r["param"], vals[j])] for j in (i - 1, i + 1) if 0 <= j < len(vals) and (r["param"], vals[j]) in sh]
            r["plateau"], r["n_sosedey"] = (sum(s) / len(s) if s else float("nan")), len(s)
        r.update(N=len(trials), n_trials_modul=n_mod, comm_pct=comm,
                 dsr=sd.dsr(r.pop("sr_d"), trials, r.pop("T"), r.pop("skew"), r.pop("kurt"))["dsr"])
    t = pd.DataFrame(rows)
    tor = []
    for p, g in t[t.param != "base"].groupby("param", sort=False):
        bs = t[t.param == "base"]
        g = pd.concat([g, bs[bs.group == g.group.iloc[0]] if (bs.group == g.group.iloc[0]).any() else bs.iloc[:1]])
        tor.append(dict(param=p, objective_min=g.objective.min(), objective_max=g.objective.max(),
                        objective_range=g.objective.max() - g.objective.min(), sharpe_min=g.tune_sharpe.min(),
                        sharpe_max=g.tune_sharpe.max(), sharpe_range=g.tune_sharpe.max() - g.tune_sharpe.min(),
                        n=len(g), comm_pct=comm))
    return t, pd.DataFrame(tor).sort_values("objective_range", ascending=False)


def code_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "fastbt"], capture_output=True, text=True).stdout.strip()
        return h + ("+правки" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "git?"


def header(name: str, s: dict, ev, sw: Sweep) -> str:
    inp = hashlib.sha1(json.dumps(s, default=str, sort_keys=True).encode()).hexdigest()[:12]
    return (f"# статус: {STATUS}; объект {name}; {datetime.now():%d.%m.%Y %H:%M}\n"
            f"# код: {code_version()}; evaluator {ev.code_id()}; sweep_drivers владельца без правок\n"
            f"# вход: сетка {inp}; ряд {ev.series_digest()[:12]}; проб в ledger (N в DSR) {len(sw.ledger)}\n")


def run_setka(name: str, setka, evaluator=None, out_dir=OUT_DIR) -> dict:
    """Одна стратегия: отбор 0,04 наполняет ledger, затем пересчёт 0,035; DSR — по полному ledger. {comm: (таблица, торнадо)}."""
    spec_d, exp, raw = load(setka)
    spec = Spec(**spec_d)
    ev = evaluator or make_evaluator(spec)
    sw = Sweep(spec, ev)
    per = periods(raw)
    raw_rows = {c: oat_rows(sw, exp, c, per) for c in spec.comm_pct}
    for r in raw_rows[spec.comm_pct[0]][0]:      # ctrl затёр SR пробы контролем: в ledger — SR цели (setup)
        sw.ledger[pkey({k[2:]: v for k, v in r.items() if k.startswith("p.")})] = r["sr_d"]
    trials, out, got = sw.trials_sr(), Path(out_dir), {}
    out.mkdir(parents=True, exist_ok=True)
    h = header(name, raw, ev, sw)
    for c, (rows, n_mod) in raw_rows.items():
        t, tor = finish(rows, exp, trials, n_mod, c)
        (out / f"{name}_oat_{c:g}.csv").write_text(h + t.to_csv(index=False), "utf-8")
        (out / f"{name}_oat_tornado_{c:g}.csv").write_text(h + tor.to_csv(index=False), "utf-8")
        got[c] = (t, tor)
    got["ledger"] = sw.ledger
    return got


def _job(args) -> str:
    path, out_dir = args
    m = re.match(r"S\d+", Path(path).name)
    name = m.group(0) if m else Path(path).stem
    got = run_setka(name, path, None, out_dir)
    return f"{name}: проб {len(got['ledger'])}, строк {len(got[0.04][0])} × {len(got) - 1} комиссий"


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("yaml", nargs="+")
    a.add_argument("--out", default=str(OUT_DIR))
    a.add_argument("--jobs", type=int, default=MAX_JOBS)
    ns = a.parse_args(argv)
    if not 1 <= ns.jobs <= MAX_JOBS:
        raise ValueError(f"--jobs в 1..{MAX_JOBS}")
    with ProcessPoolExecutor(min(ns.jobs, len(ns.yaml))) as ex:
        for line in ex.map(_job, [(p, ns.out) for p in ns.yaml]):
            print(line)


if __name__ == "__main__":
    main()
