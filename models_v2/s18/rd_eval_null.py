"""S-RCD-EVAL, объявленная диагностика «нулевое распределение полной процедуры» (дирижёр, 12.11 02:47, до QLIKE; не условие RCD-DEC): процедура S-RCD (`rd_walk`, протокол R, форма C-D)
на синтетике `z_walk_syn` (не правится): цель ln RV = ln F⁰ + шум без связи с признаками (логнормальный σ 0,5 и 1 % скачков ×30), реальные признаки, пулы, блоки, сетка с θ = 0, правило 2 из 3.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат, не OOS. Реальная RV подменяется синтетикой.
Публичное API:
- `SEEDS` = 0…9; `OUT` = `S_RCD_NULL.csv`; `run_seed(ctx, seed) -> dict`: QLIKE(R) − QLIKE(θ = 0), % QLIKE(θ = 0) на датах активных месяцев; доля месяцев θ ≠ 0.
- `summary(rows)`, `abs_bounds(seeds)` (5 и 95 % распределения |Δ_null|), `rule_abs(delta, lo, hi)`, `texts(rows)`, `read()`, `run()`, `check()` (только чтение).
Запуск: `python3 -W ignore -m models_v2.s18.rd_eval_null --run | --check`."""
from __future__ import annotations

import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import contextlib
import io
import sys
import time

import numpy as np
import pandas as pd

from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import rd_walk as RW
from models_v2.s18 import z_walk_syn as SYN
from models_v2.s18.z_eval_cfg import REP

SEEDS, OUT = tuple(range(10)), REP / "S_RCD_NULL.csv"
ITEMS = ("qlike_rel_pct", "share_theta_nonzero")
HEAD = ("# исследовательский результат на раскрытой истории, не OOS; гипотеза выбрана после просмотра результатов S-R, S-D, S-L, S-Z\n"
        "# нулевое распределение полной процедуры S-RCD (rd_walk, протокол R, форма C-D): синтетика z_walk_syn (цель ln RV = ln F⁰ + шум без связи с признаками, логнормальный σ 0,5 и 1 % скачков ×30), seeds шума 0–9; "
        "qlike_rel_pct = 100·(QLIKE(R) − QLIKE(θ = 0))/QLIKE(θ = 0) на датах активных месяцев; объявлено 12.11 02:47 до QLIKE; не условие RCD-DEC\n")


def run_seed(ctx, seed: int) -> dict:
    cs = SYN.synth_ctx(ctx, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        parts = RW.walk("rcd", cs)
    rel, lams = SYN.qlike_rel(cs, parts)
    return dict(kind="seed", seed=seed, n_active=len(lams), qlike_rel_pct=100.0 * rel, share_theta_nonzero=float(np.mean([l > 0 for l in lams])))


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    s, out = rows[rows["kind"] == "seed"], []
    for it in ITEMS:
        v = s[it].to_numpy(float)
        out += [dict(kind="summary", item=it, stat=n, value=float(f(v))) for n, f in (("mean", np.mean), ("std", lambda a: np.std(a, ddof=1)), ("q05", lambda a: np.percentile(a, 5)), ("q95", lambda a: np.percentile(a, 95)))]
    return pd.DataFrame(out)


def abs_bounds(seeds: pd.DataFrame) -> tuple:
    """Квантили 5 и 95 % распределения |Δ_null| по seeds шума (при 10 шумах — почти min и max)."""
    a = np.abs(seeds["qlike_rel_pct"].to_numpy(float))
    return float(np.percentile(a, 5)), float(np.percentile(a, 95))


def rule_abs(delta: float, lo: float, hi: float) -> str:
    return "не отличим от шума процедуры" if lo <= abs(delta) <= hi else "вне 5–95 % распределения модуля Δ_null"


def texts(rows: pd.DataFrame) -> str:
    return HEAD + pd.concat([rows, summary(rows)], ignore_index=True).to_csv(index=False, lineterminator="\n", float_format="%.12g")


def read(p=OUT) -> tuple:
    d = pd.read_csv(p, comment="#")
    return d[d["kind"] == "seed"].reset_index(drop=True), d[d["kind"] == "summary"].reset_index(drop=True)


def run() -> None:
    ctx, rows = RC.make_ctx("CK"), []
    for s in SEEDS:
        t0 = time.time()
        rows.append(run_seed(ctx, s))
        print("шум seed %d: QLIKE(R) − QLIKE(θ=0) = %+.4f %%, доля θ ≠ 0 = %.3f, %.0f с" % (s, rows[-1]["qlike_rel_pct"], rows[-1]["share_theta_nonzero"], time.time() - t0), flush=True)
    OUT.write_text(texts(pd.DataFrame(rows)), encoding="utf-8")


def check() -> list:
    if not OUT.exists():
        return ["нет S_RCD_NULL.csv"]
    h0, (s, m) = OUT.read_bytes(), read()
    bad = ["seeds ≠ 0…9"] * (list(s["seed"]) != list(SEEDS))
    want = summary(s.assign(kind="seed")).set_index(["item", "stat"])["value"]
    got = m.set_index(["item", "stat"])["value"]
    bad += ["сводка ≠ пересчёту из строк seeds"] * (not (len(got) == len(want) and np.allclose(got.loc[want.index], want, rtol=1e-9, atol=1e-12)))
    return bad + ["--check изменил файл"] * (OUT.read_bytes() != h0)


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    elif "--check" in sys.argv:
        b = check()
        print("rd_eval_null --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --run или --check")
