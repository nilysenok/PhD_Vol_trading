"""Запуск OAT по сеткам (Ж-034, Э1-а′): yaml сетки -> `Sweep` (режим oat, train 2018–2019) -> CSV с шапкой статуса.

Выход на объект S{k}: `S{k}_oat_{comm}.csv` (строка = точка; param/value; плато = среднее Шарпа по соседям ±1 узел
по ЭТОМУ параметру; N накопленное) и `S{k}_oat_tornado.csv` (размах Шарпа по параметру).
Validation 2020–2021 не читается: период жёстко train, `Sweep` сам закрывает годы validation.
Параллельность — по сеткам (процессы), не больше 8.
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

from fastbt.sweep import Spec, Sweep, make_evaluator

MAX_JOBS = 8
STATUS = "разработка, train 2018–2019"
OUT_DIR = Path("docs/tables/sweep_e1")
CACHE_DIR = Path("data/fastbt/sweep_sig")  # data/fastbt/ в .gitignore
_KEYS = ("obj", "base", "grids", "comm_pct", "periods")


def load_setka(path_or_dict) -> dict:
    d = path_or_dict if isinstance(path_or_dict, dict) else yaml.safe_load(Path(path_or_dict).read_text("utf-8"))
    if "period" in d and d["period"] != "train":
        raise ValueError("sweep_run считает только train; validation — отдельный режим движка")
    for n, vals in d["grids"].items():
        if len(set(vals)) != len(vals):
            raise ValueError(f"сетка {n}: повторяющиеся значения")
        if d["base"].get(n) not in list(vals):
            raise ValueError(f"исходная точка {n}={d['base'].get(n)!r} не узел сетки {list(vals)}")
    return d


def n_points(grids: dict) -> int:
    return sum(len(v) - 1 for v in grids.values()) + 1


def code_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "fastbt"], capture_output=True, text=True).stdout.strip()
        return h + ("+правки" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "git?"


def _changed(row, base: dict, names) -> str | None:
    ch = [n for n in names if row[n] != base[n]]
    if len(ch) > 1:
        raise ValueError(f"OAT: точка отличается от базы по {ch}")
    return ch[0] if ch else None


def oat_tables(raw: pd.DataFrame, base: dict, grids: dict) -> pd.DataFrame:
    """Таблица Sweep -> OAT-таблица: param/value и плато по соседям именно этого параметра."""
    names = list(grids)
    t = raw.copy()
    t["param"] = [_changed(r, base, names) or "base" for _, r in t.iterrows()]
    t["value"] = [r[p] if p != "base" else "" for (_, r), p in zip(t.iterrows(), t["param"])]
    sh = {}
    for (_, r), p in zip(t.iterrows(), t["param"]):
        for n in names if p == "base" else [p]:
            sh[(n, r[n], r["comm_pct"])] = r["sharpe"]
    plat, nn = [], []
    for (_, r), p in zip(t.iterrows(), t["param"]):
        if p == "base":
            plat.append(r["plateau"]); nn.append(r["n_sosedey"]); continue
        vals, i = list(grids[p]), list(grids[p]).index(r[p])
        s = [sh[(p, vals[j], r["comm_pct"])] for j in (i - 1, i + 1) if 0 <= j < len(vals)]
        plat.append(sum(s) / len(s) if s else float("nan")); nn.append(len(s))
    t["plateau"], t["n_sosedey"] = plat, nn
    first = ["param", "value", "comm_pct"]
    return t[first + [c for c in t.columns if c not in first and c not in names] + names]


def tornado(t: pd.DataFrame, base: dict) -> pd.DataFrame:
    """Размах Шарпа по параметру: по всем узлам сетки (база входит в каждую), на каждую комиссию."""
    b = t[t["param"] == "base"]
    rows = []
    for (p, c), g in t[t["param"] != "base"].groupby(["param", "comm_pct"]):
        g = pd.concat([g, b[b["comm_pct"] == c]])
        vals = [base[p] if v == "" else v for v in g["value"]]
        rows.append(dict(param=p, comm_pct=c, sharpe_min=g["sharpe"].min(), sharpe_max=g["sharpe"].max(),
                         razmah=g["sharpe"].max() - g["sharpe"].min(), n_znach=len(g),
                         znach_max=vals[int(g["sharpe"].values.argmax())]))
    return pd.DataFrame(rows).sort_values(["comm_pct", "razmah"], ascending=[True, False]).reset_index(drop=True)


def header(name: str, d: dict, ev, n: int) -> str:
    inp = hashlib.sha1(json.dumps([d["obj"], d["base"], {k: list(v) for k, v in d["grids"].items()}], default=str,
                                  sort_keys=True).encode()).hexdigest()[:12]
    return (f"# статус: {STATUS}; объект {name}; {datetime.now():%d.%m.%Y %H:%M}\n"
            f"# код: {code_version()}; evaluator {ev.code_id()}\n"
            f"# вход: сетка {inp}; ряд {ev.series_digest()[:12]}; точек {n}; N накоплено (колонка N)\n")


def run_oat(setka, evaluator=None, cache_dir=CACHE_DIR, comm_pct=None) -> dict:
    d = load_setka(setka)
    spec = Spec(**{k: (tuple(d[k]) if k == "comm_pct" else d[k]) for k in _KEYS if k in d}, period="train", mode="oat")
    if comm_pct is not None:
        spec.comm_pct = tuple(comm_pct)
    ev = evaluator or make_evaluator(spec)
    t0 = time.time()
    res = Sweep(spec, ev, cache_dir).run()
    t = oat_tables(res["table"], d["base"], d["grids"])
    exp = n_points(d["grids"])
    if len(t) != exp * len(spec.comm_pct):
        raise AssertionError(f"OAT: {len(t)} строк, ожидалось {exp}×{len(spec.comm_pct)}")
    return dict(table=t, tornado=tornado(t, d["base"]), n_points=exp, N=res["N"], seconds=time.time() - t0,
                spec=spec, d=d, ev=ev, pbo=res["pbo"])


def write(name: str, r: dict, out_dir=OUT_DIR) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h, files = header(name, r["d"], r["ev"], r["n_points"]), []
    for c, g in r["table"].groupby("comm_pct"):
        f = out / f"{name}_oat_{c:g}.csv"
        f.write_text(h + g.to_csv(index=False), "utf-8")
        files.append(f)
    f = out / f"{name}_oat_tornado.csv"
    f.write_text(h + r["tornado"].to_csv(index=False), "utf-8")
    return files + [f]


def _job(args) -> str:
    path, out_dir, cache_dir, comm = args
    name = re.match(r"S\d+", Path(path).name)
    name = name.group(0) if name else Path(path).stem
    r = run_oat(path, cache_dir=cache_dir, comm_pct=comm)
    write(name, r, out_dir)
    return f"{name}: {r['n_points']} точек × {len(r['spec'].comm_pct)} комм., N={r['N']}, время S{name[1:]}: {r['seconds']:.1f} с"


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("yaml", nargs="+")
    a.add_argument("--out", default=str(OUT_DIR))
    a.add_argument("--cache-dir", default=str(CACHE_DIR))
    a.add_argument("--comm", type=float, nargs="+", help="комиссии, %% на сторону (по умолчанию 0.035 0.04)")
    a.add_argument("--jobs", type=int, default=MAX_JOBS)
    ns = a.parse_args(argv)
    if not 1 <= ns.jobs <= MAX_JOBS:
        raise ValueError(f"--jobs в 1..{MAX_JOBS}")
    work = [(p, ns.out, ns.cache_dir, ns.comm) for p in ns.yaml]
    with ProcessPoolExecutor(min(ns.jobs, len(work))) as ex:
        for line in ex.map(_job, work):
            print(line)


if __name__ == "__main__":
    main()
