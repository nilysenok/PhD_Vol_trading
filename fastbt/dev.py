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
    kw = {k: v for k, v in design.items() if k in Config.__dataclass_fields__}
    kw.update(SCOPES[scope])
    kw["mask_old"] = False
    return Config(**kw)


def append_log(it_hash, scope, tab, deltas, secs):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Журнал режима разработки стратегий (fastbt.dev)\n\n"
                       "| № | время | дизайн | область | Шарп МСП по годам (A/B/C/D) | ΔC−A | ΔD−A | с |\n"
                       "|---|---|---|---|---|---|---|---|\n")
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
                f"{d('C-A')} | {d('D-A')} | {secs:.0f} |\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--design")
    p.add_argument("--scope", choices=list(SCOPES), default="dev")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data-root")
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-log", action="store_true")
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
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not (a.no_log or a.smoke):
        append_log(h.hexdigest()[:10], a.scope, tab, deltas, secs)
    print(f"дизайн {h.hexdigest()[:10]}, {secs:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
