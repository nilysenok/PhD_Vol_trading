"""S-Z-EVAL, объявленная диагностика «нулевое распределение процедуры» (журнал 12.11 01:34; `S_Z_SPEC.md`, разд. 7; не условие Z-DEC): S-RCKZ на синтетике `z_walk_syn` (не правится) — цель ln RV = ln F⁰ + шум без связи с z, реальные z̃ (5 seeds), реальные пулы, блоки, сетка, правило 2 из 3.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат, не OOS. RV реальных данных не используется (подменяется синтетикой).
Публичное API:
- `SEEDS` = 0…9 (десять шумов); `OUT` = `docs/otchety/harj_struct/S_Z_NULL.csv`.
- `run_seed(ctx, zm, seed) -> dict`: QLIKE(R) − QLIKE(θ = 0) в % QLIKE(θ = 0) на датах активных месяцев, доля месяцев θ ≠ 0, число активных месяцев.
- `summary(rows) -> DataFrame`: среднее, СКО (ddof = 1), квантили 5 и 95 % (линейные) по seeds для обеих величин — чистая функция строк.
- `rule(delta, lo, hi) -> str`: реальный Δ (со знаком; шум процедуры несимметричен) внутри [q05; q95] ⇒ «не отличим от шума процедуры» (границы — внутри).
- `texts(rows) -> str`, `run()` (считает 10 шумов и пишет `S_Z_NULL.csv`), `check() -> list` (только чтение: seeds 0–9, сводка = пересчёту из строк).
Запуск: `python3 -W ignore -m models_v2.s18.z_eval_null --run | --check`."""
from __future__ import annotations

import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import sys
import time

import numpy as np
import pandas as pd

from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_walk as ZW
from models_v2.s18 import z_walk_syn as SYN
from models_v2.s18.z_eval_cfg import REP

SEEDS, OUT = tuple(range(10)), REP / "S_Z_NULL.csv"
ITEMS = ("qlike_rel_pct", "share_theta_nonzero")
HEAD = ("# исследовательский результат на раскрытой истории, не OOS; механизм выбран после просмотра истории\n"
        "# нулевое распределение процедуры S-RCKZ: синтетика z_walk_syn (цель ln RV = ln F⁰ + шум без связи с z, логнормальный σ 0,5 и 1 %% скачков ×30), реальные z̃, seeds шума 0–9; "
        "qlike_rel_pct = 100·(QLIKE(R) − QLIKE(θ = 0))/QLIKE(θ = 0) на датах активных месяцев; не условие Z-DEC\n")


def run_seed(ctx, zm, seed: int) -> dict:
    cs = SYN.synth_ctx(ctx, seed)
    rel, lams = SYN.qlike_rel(cs, ZW.walk(cs, zm, "CKZ", True, quiet=True))
    return dict(kind="seed", seed=seed, n_active=len(lams), qlike_rel_pct=100.0 * rel, share_theta_nonzero=float(np.mean([l > 0 for l in lams])))


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    s, out = rows[rows["kind"] == "seed"], []
    for it in ITEMS:
        v = s[it].to_numpy(float)
        out += [dict(kind="summary", item=it, stat=n, value=float(f(v))) for n, f in (("mean", np.mean), ("std", lambda a: np.std(a, ddof=1)), ("q05", lambda a: np.percentile(a, 5)), ("q95", lambda a: np.percentile(a, 95)))]
    return pd.DataFrame(out)


def rule(delta: float, lo: float, hi: float) -> str:
    return "не отличим от шума процедуры" if lo <= delta <= hi else "вне 5–95 % нулевого распределения"


def texts(rows: pd.DataFrame) -> str:
    return HEAD.replace("%%", "%") + pd.concat([rows, summary(rows)], ignore_index=True).to_csv(index=False, lineterminator="\n", float_format="%.12g")


def read(p=OUT) -> tuple:
    d = pd.read_csv(p, comment="#")
    return d[d["kind"] == "seed"].reset_index(drop=True), d[d["kind"] == "summary"].reset_index(drop=True)


def run() -> None:
    ctx, (zm, _) = RC.make_ctx("CK"), ZW.load_z()
    rows = []
    for s in SEEDS:
        t0 = time.time()
        rows.append(run_seed(ctx, zm, s))
        print("шум seed %d: QLIKE(R) − QLIKE(θ=0) = %+.4f %%, доля θ ≠ 0 = %.3f, %.0f с" % (s, rows[-1]["qlike_rel_pct"], rows[-1]["share_theta_nonzero"], time.time() - t0), flush=True)
    OUT.write_text(texts(pd.DataFrame(rows)), encoding="utf-8")


def check() -> list:
    if not OUT.exists():
        return ["нет S_Z_NULL.csv"]
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
        print("z_eval_null --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --run или --check")
