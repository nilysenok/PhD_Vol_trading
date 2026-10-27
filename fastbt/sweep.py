"""Движок перебора (Ж-031, Э1-а): обёртка над существующим быстрым слоем, движок расчёта не переписывает.

Вход — yaml (`Spec.from_yaml`): объект, база и сетки, период, комиссии (% на сторону), режим.
Режимы: oat | factorial | morris | optuna | validation. Выход — таблица на точку (`Sweep.run`).
Validation в целевую функцию не входит: `Sweep` не отдаёт годы validation, пока не вызван режим
`validation`, а он пишет флаг «validation открыта: <время>»; повторное открытие — только с записью
об изменении (`reopen_note`), N в DSR удваивается.
Кэш сигналов (`SignalCache`) отдельно от PnL: ключ = хэш (объект, параметры сигнала, ряд, код).
Evaluator: `sig_names`, `series_digest()`, `code_id()`, `signals(sig_params)`, `pnl(sigs, params, comm, end)`
-> DataFrame [excess, exposure, turnover, sides] по дням < end.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import metriki_e as me
from fastbt import sweep_modes as sm
from fastbt import sweep_stat as ss

MODES = ("oat", "factorial", "morris", "optuna", "validation")


@dataclass
class Spec:
    obj: dict
    base: dict
    grids: dict
    period: str = "train"
    comm_pct: tuple = (0.035, 0.04)
    mode: str = "oat"
    periods: ss.Periods = field(default_factory=ss.Periods)
    lam: float = ss.LAM
    morris: dict = field(default_factory=dict)
    optuna: dict = field(default_factory=dict)
    factorial: dict = field(default_factory=dict)
    points: list = field(default_factory=list)
    out: str | None = None
    reopen_note: str | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"режим {self.mode!r} не из {MODES}")
        if (self.mode == "validation") != (self.period == "validation"):
            raise ValueError("период validation — только в режиме validation и наоборот")
        if self.mode == "validation" and not self.points:
            raise ValueError("режим validation считает только заданные точки (points)")
        self.years = self.periods.years(self.period)

    @classmethod
    def from_yaml(cls, path_or_dict):
        import yaml
        d = path_or_dict if isinstance(path_or_dict, dict) else yaml.safe_load(Path(path_or_dict).read_text("utf-8"))
        d = dict(d)
        if "periods" in d:
            d["periods"] = ss.Periods(**{k: tuple(v) for k, v in d["periods"].items()})
        if "comm_pct" in d:
            d["comm_pct"] = tuple(d["comm_pct"])
        return cls(**d)


class SignalCache:
    def __init__(self, d=None):
        self.mem, self.dir, self.hits, self.misses = {}, Path(d) if d else None, 0, 0

    @staticmethod
    def key(obj, sig_params, series_digest, code_id) -> str:
        raw = json.dumps([obj, sorted(sig_params.items()), series_digest, code_id], default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def get(self, k, make):
        if k in self.mem:
            self.hits += 1
            return self.mem[k]
        f = self.dir / f"{k}.pkl" if self.dir else None
        if f is not None and f.exists():
            self.mem[k] = pickle.loads(f.read_bytes())
            self.hits += 1
            return self.mem[k]
        self.misses += 1
        self.mem[k] = make()
        if f is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(self.mem[k]))
        return self.mem[k]


def pkey(p: dict) -> str:
    return json.dumps(sorted(p.items()), default=str)


class Sweep:
    def __init__(self, spec: Spec, evaluator, cache_dir=None):
        self.spec, self.ev, self.cache = spec, evaluator, SignalCache(cache_dir)
        self.out = Path(spec.out) if spec.out else None
        self.comm = [c / 100.0 for c in spec.comm_pct]
        self.ledger, self._mem, self._lock, self._seen = {}, {}, threading.Lock(), {}
        self._val_open = False
        if self.out and (self.out / "ledger.json").exists():
            self.ledger = json.loads((self.out / "ledger.json").read_text("utf-8"))

    # --- расчёт одной точки ---
    def _df(self, p: dict, comm: float, years) -> pd.DataFrame:
        if any(y in self.spec.periods.validation for y in years) and not self._val_open:
            raise PermissionError("validation закрыта: годы 2020–2021 только в режиме validation")
        if max(years) > ss.END_ALLOWED:
            raise PermissionError("данные ≥ 2022 запрещены")
        mk = (pkey(p), comm, tuple(years))
        if mk not in self._mem:
            sp = {k: v for k, v in p.items() if k in self.ev.sig_names}
            k = SignalCache.key(self.spec.obj, sp, self.ev.series_digest(), self.ev.code_id())
            sigs = self.cache.get(k, lambda: self.ev.signals(sp))
            end = pd.Timestamp(f"{max(years) + 1}-01-01")
            self._mem[mk] = ss.year_slice(self.ev.pnl(sigs, p, comm, end), years)
        return self._mem[mk]

    def _register(self, p, df):
        if self.spec.mode != "validation":
            with self._lock:
                self.ledger[pkey(p)] = me.sr_dnevnoy(df["excess"].values)
                self._seen[pkey(p)] = p

    def objective(self, p: dict, lam: float | None = None) -> float:
        df = self._df(p, self.comm[0], self.spec.years)
        self._register(p, df)
        return ss.objective(ss.yearly_sharpes(df, self.spec.years), self.spec.lam if lam is None else lam)

    def partial(self, p: dict) -> float:
        """Шарп первого года периода (для pruning); точка идёт в N."""
        y0 = self.spec.years[:1]
        df = self._df(p, self.comm[0], y0)
        with self._lock:
            self.ledger.setdefault(pkey(p), me.sr_dnevnoy(df["excess"].values))
        return ss.yearly_sharpes(df, y0)[int(y0[0])]

    # --- validation ---
    def open_validation(self) -> dict:
        if self.out is None:
            raise ValueError("режим validation: нужен out (флаг открытия пишется на диск)")
        f = self.out / "validation_flag.json"
        opens = json.loads(f.read_text("utf-8"))["opens"] if f.exists() else []
        if opens and not self.spec.reopen_note:
            raise PermissionError("validation уже открыта; повторно — только с записанным изменением (reopen_note)")
        t = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        opens.append(dict(text=f"validation открыта: {t}", note=self.spec.reopen_note, n_probes=len(self.ledger)))
        self.out.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(dict(opens=opens, n_mult=2 ** (len(opens) - 1)), ensure_ascii=False), "utf-8")
        self._val_open = True
        return dict(opens=opens, n_mult=2 ** (len(opens) - 1))

    # --- сетка и таблица ---
    def neighbors(self, p: dict) -> list:
        out = []
        for n, vals in self.spec.grids.items():
            vals = list(vals)
            if p.get(n) in vals:
                i = vals.index(p[n])
                out += [{**p, n: vals[j]} for j in (i - 1, i + 1) if 0 <= j < len(vals)]
        return out

    def points(self) -> list:
        s = self.spec
        if s.mode == "oat":
            return sm.oat_points(s.base, s.grids)
        if s.mode == "factorial":
            return sm.factorial_points(s.base, s.grids, s.factorial.get("names"), s.factorial.get("points"))
        if s.mode == "validation":
            return [{**s.base, **p} for p in s.points]
        return []

    def run(self) -> dict:
        s, extra, mult = self.spec, {}, 1
        if s.mode == "validation":
            mult = self.open_validation()["n_mult"]
        pts = self.points()
        if s.mode == "morris":
            levels = {n: list(v) for n, v in s.grids.items()}
            extra["morris"] = sm.morris_effects(self.objective, levels, s.morris.get("r", sm.MORRIS_R),
                                                s.morris.get("seed", 0), s.base)
            pts = list(self._seen.values())
        if s.mode == "optuna":
            o = s.optuna
            extra["optuna"] = sm.optuna_run(self, o.get("n_trials", 100), o.get("seed", 0), s.lam, o.get("n_jobs", 1))
            pts = [{**s.base, **t["params"]} for t in extra["optuna"] if t["state"] == "COMPLETE"]
        pts = list({pkey(p): p for p in pts}.values())
        rows = [self._row(p, c) for p in pts for c in self.comm]
        for r in rows:
            r["plateau"], r["n_sosedey"] = self._plateau(r)
        n = len(self.ledger)
        srs = list(self.ledger.values())
        for r in rows:
            r["N"], r["n_mult"] = n * mult, mult
            r["dsr"] = ss.dsr_n(self._df(r["_p"], self.comm[0], s.years)["excess"].values, srs, mult)
        pbo = self._pbo(pts)
        for r in rows:
            r["pbo"] = pbo
        table = pd.DataFrame([{k: v for k, v in r.items() if k != "_p"} for r in rows])
        self._save(table, extra)
        return dict(table=table, pbo=pbo, N=n * mult, **extra)

    def evaluate(self, params: dict, comm_pct: float | None = None) -> dict:
        """Одна строка стандартной таблицы для точки: метрики периода spec.period, плато, N, DSR(N).
        Строка не включает PBO (он по набору точек — драйверы режимов)."""
        comm = self.comm[0] if comm_pct is None else comm_pct / 100.0
        p = {**self.spec.base, **params}
        r = self._row(p, comm)
        r["plateau"], r["n_sosedey"] = self._plateau(r)
        mult = 2 ** max(len(self._opens()) - 1, 0) if self.spec.mode == "validation" else 1
        r["N"], r["n_mult"] = len(self.ledger) * mult, mult
        r["dsr"] = ss.dsr_n(self._df(p, comm, self.spec.years)["excess"].values, list(self.ledger.values()), mult)
        r.pop("_p")
        return r

    def _opens(self) -> list:
        f = self.out / "validation_flag.json" if self.out else None
        return json.loads(f.read_text("utf-8"))["opens"] if f and f.exists() else []

    def _row(self, p, comm) -> dict:
        df = self._df(p, comm, self.spec.years)
        if comm == self.comm[0]:
            self._register(p, df)
        r = dict(_p=p, **p, comm_pct=comm * 100, **ss.metrics_row(df, self.spec.years))
        r["objective"] = ss.objective(ss.yearly_sharpes(df, self.spec.years), self.spec.lam)
        return r

    def _plateau(self, r) -> tuple:
        sh = [ss.metrics_row(self._df(q, r["comm_pct"] / 100, self.spec.years), self.spec.years)["sharpe"]
              for q in self.neighbors(r["_p"])]
        for q in self.neighbors(r["_p"]):
            self._register(q, self._df(q, self.comm[0], self.spec.years))
        return (float(np.mean(sh)) if sh else float("nan")), len(sh)

    def _pbo(self, pts) -> float:
        cols = [self._df(p, self.comm[0], self.spec.years)["excess"] for p in {pkey(p): p for p in pts}.values()]
        return ss.pbo_cscv(pd.concat(cols, axis=1).dropna().values) if len(cols) > 1 else float("nan")

    def _save(self, table, extra):
        if self.out is None:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.out / "tablica.csv", index=False)
        (self.out / "ledger.json").write_text(json.dumps(self.ledger), "utf-8")
        (self.out / "dop.json").write_text(json.dumps(extra, ensure_ascii=False, default=str), "utf-8")


def make_evaluator(spec: Spec):
    kind = spec.obj.get("kind")
    if kind == "Sk":
        from fastbt.sweep_sk import SkEvaluator
        return SkEvaluator(spec.obj["sid"], spec.obj.get("tickers"))
    raise NotImplementedError(f"объект {kind!r}: адаптер сборки A / семей C,E,B,D подключается через Evaluator (Э1-в, Э2)")


def main(argv=None):
    import argparse
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("yaml")
    a.add_argument("--cache-dir")
    ns = a.parse_args(argv)
    spec = Spec.from_yaml(ns.yaml)
    res = Sweep(spec, make_evaluator(spec), ns.cache_dir).run()
    print(f"OK: точек {len(res['table'])}, N={res['N']}")


if __name__ == "__main__":
    main()
