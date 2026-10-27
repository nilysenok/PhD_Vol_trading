"""Сборки A для Э1-г (Ж-031): 36 сборок поверх фиксированных точек S1–S6, оценщик под `Sweep`, драйвер, экспорт A′.

Вход: по одной точке на стратегию (`load_point` из `S{k}_top3.csv`: лучшая по плато, не на краю сетки, комиссия отбора)
и сетка `A_sborki_e1g.yaml` (веса × лимит позиций × лимит на стратегию). Веса и лимиты — `sweep_a_core`.
Оценщик `AEvaluator` отдаёт `Sweep` ту же строку, что и Sk (ledger, N, DSR, excess_d). Данные ≥ 2022 закрыты `Sweep`.
Выбор — только фиксированные точки (`otbor_statistika` п. 3): внутри сборки выбора на единицу наблюдения нет.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import sweep_drivers as sd
from fastbt.sweep import Sweep
from fastbt.sweep_a_core import Panel, assemble
from fastbt.sweep_spec import Spec

YAML = Path(__file__).parent / "designs" / "sweep" / "A_sborki_e1g.yaml"
MAIN_YAML = Path(__file__).parent / "designs" / "final" / "main.yaml"
BASE = {"vesa": "ravnye", "limit_pozitsiy": "net", "limit_na_strategiyu": "net"}   # исходная A
COMMS = (0.04, 0.035)       # % на сторону; первая — комиссия отбора


def load_grid(path=YAML) -> dict:
    s = yaml.safe_load(Path(path).read_text("utf-8"))
    return {p["imya"]: list(p["setka"]) for p in s["bloki"]["sborka"]["parametry"]}


def load_point(csv, comm_pct: float = 0.04) -> dict:
    """Точка стратегии из `S{k}_top3.csv` (`#`-шапка, колонки `p.<имя>`): макс. `plateau` среди строк не на краю при comm_pct."""
    t = pd.read_csv(csv, comment="#")
    t = t[(t["comm_pct"].round(6) == comm_pct) & (~t["edge"].astype(bool))]
    if t.empty:
        raise ValueError(f"{csv}: нет точки не на краю сетки при {comm_pct} %")
    r = t.sort_values("plateau", ascending=False).iloc[0]
    return {c[2:]: (int(v) if float(v).is_integer() else float(v)) if isinstance(v, (int, float, np.number)) else v
            for c, v in r.items() if c.startswith("p.") and not pd.isna(v)}


class AEvaluator:
    """Протокол Evaluator для `Sweep`: сигналов нет (sig_names пуст), всё в `pnl` по панели источника."""
    sig_names: set = set()

    def __init__(self, source):
        self.src, self._wc = source, {}

    def series_digest(self) -> str:
        return self.src.digest()

    def code_id(self) -> str:
        return self.src.code()

    def signals(self, sp: dict):
        return "A"

    def pnl(self, sigs, p: dict, comm: float, end) -> pd.DataFrame:
        pn = self.src.panel(end)
        return assemble(pn, p, comm, self._wc.setdefault(id(pn), {}))


class SkPanelSource:
    """Панель пар по реальным данным: 6 `SkEvaluator` в фиксированных точках, позиции до `end` (≤ 2021, Н6)."""

    def __init__(self, points: dict, tickers=None):
        self.points, self.tickers, self._ev, self._sigs, self._pn = points, tickers, {}, {}, {}

    def _evs(self):
        from fastbt.sweep_sk import SkEvaluator
        for sid in self.points:
            if sid not in self._ev:
                ev = SkEvaluator(sid, self.tickers)
                self._ev[sid], self._sigs[sid] = ev, ev.signals({k: v for k, v in self.points[sid].items() if k in ev.sig_names})
        return self._ev

    def digest(self) -> str:
        raw = "".join(ev.series_digest() for ev in self._evs().values()) + json.dumps(self.points, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def code(self) -> str:
        from fastbt import cache
        return cache.code_hash()

    def panel(self, end) -> Panel:
        if end not in self._pn:
            self._pn[end] = self._build(pd.Timestamp(end))
        return self._pn[end]

    def _build(self, end) -> Panel:
        from fastbt import msp
        from fastbt.data import year_bounds
        from fastbt.sweep_post import build_view, positions
        from fastbt.sweep_sk import START_YEAR
        cols = []                                           # (sid, даты, pos, lr, сила)
        for sid, ev in self._evs().items():
            for t, u in self._sigs[sid].items():
                i0, i1 = year_bounds(u.dates, START_YEAR)[0], int(np.searchsorted(u.dates, np.datetime64(end)))
                td_, ind = ev._td[t]
                v, ex = build_view(sid, u, td_, ind, self.points[sid], ev.sig_names)
                b = np.arange(i0, i1)
                cols.append((sid, u.dates[b], positions(v, ex, i1)[b], u.lr[b], v.strength[0][b]))
        dts = pd.DatetimeIndex(np.unique(np.concatenate([c[1] for c in cols])))
        D, P = len(dts), len(cols)
        pos, lr, st, act = np.zeros((D, P)), np.zeros((D, P)), np.zeros((D, P)), np.zeros((D, P), bool)
        for j, (_, d, ps, l, s) in enumerate(cols):
            k = dts.get_indexer(pd.DatetimeIndex(d))
            pos[k, j], lr[k, j], st[k, j], act[k, j] = ps, l, s, True
        rate = msp.rate_cbr()
        kr = rate.reindex(rate.index.union(dts)).sort_index().ffill().bfill().reindex(dts).values
        return Panel(dts, np.array([c[0] for c in cols]), pos, lr, act, st, kr / 365.0 * msp.cal_days(dts))


class SynthPanelSource:
    """Синтетическая панель для тестов и `--dry`: n_t тикеров × стратегии sids, позиции — блоки ±1, шум в доходностях."""

    def __init__(self, n_t: int = 4, sids=("S1", "S2", "S3"), seed: int = 0, drift: float = 0.0):
        self.dates = pd.bdate_range("2018-01-01", "2021-12-31")
        r, D = np.random.default_rng(seed), len(self.dates)
        P = n_t * len(sids)
        self.sid = np.array([s for s in sids for _ in range(n_t)])
        self.lr = r.normal(drift, 0.01, (D, P))
        sign = np.sign(r.normal(size=(D // 7 + 1, P)))
        self.pos = np.repeat(sign, 7, axis=0)[:D] * (r.random((D, P)) < 0.6)
        self.st, self.seed = np.abs(r.normal(size=(D, P))), seed

    def digest(self) -> str:
        return hashlib.sha1(self.lr.tobytes() + self.pos.tobytes()).hexdigest()

    def code(self) -> str:
        return "synth"

    def panel(self, end) -> Panel:
        m = self.dates < pd.Timestamp(end)
        pos = self.pos[m].copy()
        return Panel(self.dates[m], self.sid, pos, self.lr[m], np.ones(pos.shape, bool), self.st[m], np.zeros(m.sum()))


def make_sweep(source, out=None, reopen_note=None) -> Sweep:
    """`Sweep` сборок: ledger `A` в `out` (ledger.json, ledger_excess.json, validation_flag.json)."""
    spec = Spec(obj={"kind": "A_sborka"}, base=dict(BASE), comm_pct=COMMS, out=out, reopen_note=reopen_note)
    return Sweep(spec, AEvaluator(source))


def run_sborki(sw: Sweep, grid: dict | None = None) -> dict:
    """36 сборок: `factorial_sweep` на setup (0,04), затем строки setup/control × 0,04/0,035. Топ-2 по цели при плато ≥ 3 соседей."""
    grid = grid or load_grid()
    fact = sd.factorial_sweep(lambda p, per: sw.evaluate(p, per, COMMS[0]), dict(BASE), grid, "setup")
    keys, rows = fact["keys"], []
    for r in fact["rows"]:
        pt = {k: r[k] for k in keys}
        row = dict(pt, objective=r["objective"], plateau_neighbors=sd.plateau_score(fact["rows"], keys, fact["grid"], pt)[1])
        for c in COMMS:
            for per, tag in (("setup", "tune"), ("control", "ctrl")):
                x = sw.evaluate(pt, per, c)
                row.update({f"{tag}_{m}_c{c:g}": x[m] for m in ("sharpe", "msr", "sortino", "mdd", "oborot_y", "dni_v_rynke")})
        s = sw.evaluate(pt, "setup", COMMS[0])
        row["dsr"] = sd.dsr(s["sr_d"], sw.trials_sr(), s["T"], s["skew"], s["kurt"])["dsr"]
        rows.append(row)
    t = pd.DataFrame(rows)
    top2 = t[t.plateau_neighbors >= 3].sort_values("objective", ascending=False).head(2)
    return dict(table=t, top2=top2.reset_index(drop=True), n=len(sw.trials_sr()), n_modul=fact["n_trials"])


def export_a_prime(points: dict, sborka: dict, path, main=MAIN_YAML) -> dict:
    """A′ в формате дизайна `final/main.yaml` (ключи main.yaml сохранены) + блок `a_prime` (точки стратегий и сборка).
    Конвейер s9 лишние ключи отбрасывает (`dev.make_config`); чтение `a_prime` — адаптер T3 (❓)."""
    d = yaml.safe_load(Path(main).read_text("utf-8"))
    d["a_prime"] = dict(sborka={k: sborka[k] for k in BASE}, tochki={s: dict(p) for s, p in sorted(points.items())})
    Path(path).write_text("# A′ (Э1-г): сборка поверх фиксированных точек S1–S6; остальное — дизайн main.yaml\n"
                          + yaml.safe_dump(d, allow_unicode=True, sort_keys=False), "utf-8")
    return d
