"""Режим разработки стратегий (шаг 5).

python3 -m fastbt.dev --design <yaml> [--scope dev|test] [--boot-reps 1000] [--workers 12]
python3 -m fastbt.dev --smoke [--data-root DIR]       (1 тикер × 1 стратегия, для leak_detector)

dev  — walk-forward 2017–2021, маска [0]: A 2017–2021, B/C/D 2018–2021 (валидация с 2017).
test — 2022–2026; без docs/otchety/FINAL_DESIGN_strategies.md падает (замок контрольного
периода). Каждый запуск дописывает строку в docs/otchety/dev_log_strategies.md.
Ключи yaml: любые поля Config (fix_*, comm, msp_comm), tickers, sids, vpred (путь к σ_pred).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "docs" / "otchety" / "FINAL_DESIGN_strategies.md"
LOG = ROOT / "docs" / "otchety" / "dev_log_strategies.md"
SCOPES = {"dev": dict(years_a=tuple(range(2017, 2022)), years_bcd=tuple(range(2018, 2022)),
                      val_start=2017),
          "test": dict(years_a=tuple(range(2022, 2027)), years_bcd=tuple(range(2022, 2027)),
                       val_start=2020)}


class TestLocked(SystemExit):
    pass


def check_scope(scope: str) -> None:
    if scope == "test" and not FINAL.exists():
        raise TestLocked(f"контрольный период закрыт: нет docs/otchety/{FINAL.name}")


def load_design(path):
    if path is None:
        return {}, "smoke"
    import yaml
    text = Path(path).read_text()
    return (yaml.safe_load(text) or {}), text


def make_config(design: dict, scope: str):
    from fastbt.config import Config
    from fastbt.design import KEYS, freeze
    kw = {k: v for k, v in design.items() if k in Config.__dataclass_fields__}
    kw["design"] = freeze({k: v for k, v in design.items() if k in KEYS})
    kw.update(SCOPES[scope])
    kw["mask_old"] = False
    return Config(**kw)


def evaluate(tab, years, period: dict) -> dict:
    """Критерии выбора дизайна C и D (план + решение владельца 25.10, 19:20):
    (а) X лучше A в каждом году, одно исключение; (б) худший год X не хуже худшего года A;
    (в) избыточный Шарп МСП X за весь период (годы B/C/D) выше, чем у A.
    Ранжирование прошедших: сначала с положительным избыточным Шарпом за период, среди них —
    максимальный средний по годам ΔШарп X−A; нет положительных — по (а)–(в) тем же ΔШарп."""
    t = tab.set_index("approach")
    ys = [str(y) for y in years if str(y) in t.columns]
    a = t.loc["A", ys].astype(float)
    out = {"A_period": period.get("A")}
    for x in ("B", "C", "D"):
        if x not in t.index:
            continue
        v = t.loc[x, ys].astype(float)
        win = int((v > a).sum())
        c_years = win >= len(ys) - 1
        c_worst = bool(v.min() >= a.min())
        c_period = bool(period.get(x, -np.inf) > period.get("A", np.inf))
        out[x] = dict(win=win, n_years=len(ys), worst=float(v.min()), worst_A=float(a.min()),
                      period=period.get(x), c_years=c_years, c_worst=c_worst,
                      c_period=c_period, passed=bool(c_years and c_worst and c_period),
                      positive=bool(period.get(x, -np.inf) > 0),
                      mean_delta=float((v - a).mean()),
                      years={y: float(v[y]) for y in ys})
    out["A_years"] = {y: float(a[y]) for y in ys}
    return out


def criteria(tab, years, period=None) -> str:
    """Строка критериев для журнала (C и D)."""
    if "A" not in set(tab["approach"]):
        return "—"
    ev = evaluate(tab, years, period or {})
    parts = []
    for x in ("C", "D"):
        if x not in ev:
            continue
        e = ev[x]
        per = "—" if e["period"] is None else f"{e['period']:.2f}"
        pa = "—" if ev["A_period"] is None else f"{ev['A_period']:.2f}"
        parts.append(f"{x}: лучше A {e['win']}/{e['n_years']}, худший {e['worst']:.2f} vs A "
                     f"{e['worst_A']:.2f}, период {per} vs A {pa}, ср. Δ {e['mean_delta']:+.2f} → "
                     + ("да" if e["passed"] else "нет")
                     + (", избыточный Шарп > 0" if e["positive"] else ""))
    return "; ".join(parts)


def period_sharpe(pos, cfg) -> dict:
    """Избыточный Шарп МСП по подходам за годы B/C/D (для A — те же годы)."""
    from fastbt import msp
    from fastbt.report import portfolios
    ports = portfolios(pos, cfg, list(cfg.years_bcd))
    return {ap: msp.sharpe(p["excess"]) for ap, p in ports.items()}


def save_run(it_hash, design_path, note, tab, deltas, ev):
    """JSON прогона в <основная копия>/data/fastbt/dev_runs/ — для ранжирования (fastbt.dev_rank)."""
    from fastbt.cache import MAIN
    d = MAIN / "data" / "fastbt" / "dev_runs"
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(hash=it_hash, design=str(design_path), note=note,
               time=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), eval=ev,
               tab=tab.to_dict(orient="records"), deltas=deltas.to_dict(orient="records"))
    (d / f"{it_hash}.json").write_text(json.dumps(rec, ensure_ascii=False, default=float))


def append_log(it_hash, scope, tab, deltas, secs, crit="—", note=""):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Журнал режима разработки стратегий (fastbt.dev)\n\n"
                       "| № | время | дизайн | область | Шарп МСП по годам (A/B/C/D) | ΔC−A | ΔD−A |"
                       " критерии устойчивости | пометка | с |\n"
                       "|---|---|---|---|---|---|---|---|---|---|\n")
    n = sum(1 for ln in LOG.read_text().splitlines() if ln.startswith("| ") and ln[2:3].isdigit())
    yrs = [c for c in tab.columns if c.isdigit()]
    cells = "; ".join(f"{r['approach']}: " + " ".join(f"{r[y]:.2f}" for y in yrs)
                      for _, r in tab.iterrows())

    def d(name):
        r = deltas[deltas["delta"] == name]
        return "—" if r.empty else f"{r['obs'].iloc[0]:+.2f} (p={r['p'].iloc[0]:.3f})"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with LOG.open("a") as f:
        f.write(f"| {n + 1} | {now} | {it_hash} | {scope} {yrs[0]}–{yrs[-1]} | {cells} | "
                f"{d('C-A')} | {d('D-A')} | {crit} | {note} | {secs:.0f} |\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--design")
    p.add_argument("--scope", choices=list(SCOPES), default="dev")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data-root")
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--note", default="", help="пометка в журнал (например «отладка, старый σ_pred»)")
    a = p.parse_args(argv)
    if a.data_root:
        os.environ["FASTBT_DATA_ROOT"] = a.data_root
    check_scope(a.scope)
    if not a.smoke and not a.design:
        p.error("нужен --design <yaml> или --smoke")
    design, text = load_design(a.design)
    if design.get("vpred"):
        os.environ["FASTBT_VPRED"] = str(design["vpred"])
    import time
    from fastbt.cache import code_hash
    from fastbt.data import TICKERS
    from fastbt.pipeline import run_all
    from fastbt.report import summary
    cfg = make_config(design, a.scope)
    tickers = ["SBER"] if a.smoke else design.get("tickers", TICKERS)
    sids = ["S3"] if a.smoke else design.get("sids", ["S1", "S2", "S3", "S4", "S5", "S6"])
    h = hashlib.sha1((text + json.dumps(cfg.to_dict(), sort_keys=True) + code_hash()).encode())
    t0 = time.time()
    pos, _ = run_all(cfg, tickers, sids, a.workers, log=lambda m: None)
    years = list(cfg.years_a)
    tab, deltas = summary(pos, cfg, years, a.boot_reps)
    secs = time.time() - t0
    per = period_sharpe(pos, cfg)
    ev = evaluate(tab, cfg.years_bcd, per)
    crit = criteria(tab, cfg.years_bcd, per)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not a.smoke:
        save_run(h.hexdigest()[:10], a.design, a.note, tab, deltas, ev)
    if not (a.no_log or a.smoke):
        append_log(h.hexdigest()[:10], a.scope, tab, deltas, secs, crit, a.note)
    print("критерии:", crit)
    print(f"дизайн {h.hexdigest()[:10]}, {secs:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
