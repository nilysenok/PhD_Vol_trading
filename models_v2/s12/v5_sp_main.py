"""V5 лист 2: запуск стадии SP (из `run_v5 --stage SP`). Процесс один.
  (без флагов)       посчитать SP1–SP4 (кэш), собрать таблицы, ALL_TRIALS, CHECKPOINT
  --check            ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, суммарный хэш выходов до и после"""
from __future__ import annotations

import numpy as np

from models_v2.s12 import v5_0_io as IO, v5_sp_stage as ST


def evidence(r: dict) -> list:
    t, cp = r["t"], r["cp"]
    go, enc = t["_go"], t["SP_ENCOMPASS"]
    sp = go[go["series"].isin(ST.SP)].set_index("series")
    e = enc[enc["window"] == "2022_2025"].set_index("base")
    lines = ["G20: расход %s; направление 2: %s; лучшая %s" % (cp.get("SP", {}).get("used"), t["SP_COMPARISON"]["decision_direction2"].iloc[0], t["_best"])]
    lines += ["G20: %s crit_panel %.6f (порог %.6f), без 2017 %.6f (%.6f), без 1 %% %.6f (%.6f); go %s; EqY %.6f" % (s, v.crit_panel, v.thr1, v.crit_no2017, v.thr2, v.crit_trim1, v.thr3, v.go, v.EqY) for s, v in sp.iterrows()]
    lines.append("G21: w(M) 2022–2025 %.3f [%.3f; %.3f]; w(K8) %.3f [%.3f; %.3f]" % (e.loc["M", "w_hat"], e.loc["M", "w_lo90"], e.loc["M", "w_hi90"], e.loc["K8", "w_hat"], e.loc["K8", "w_lo90"], e.loc["K8", "w_hi90"]))
    lines.append("кэш: %s" % {g: v["cached"] for g, v in r["runs"].items()})
    return lines


def main(a) -> int:
    before = IO.outputs_digest()
    r = ST.compute(write=not a.check)
    if not a.check:
        ST.write(r)
        print("стадия SP записана; хэш выходов %s" % IO.outputs_digest())
        return 0
    bad = ST.check(r)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш до %s, после %s" % (before, after))
    print("\n".join(evidence(r)))
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G20–G21: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK", "| бумаг в панели", len(r["P"].tick), "| NaN в F:", int(np.isnan(np.concatenate([v["main"]["F"].to_numpy() for v in r["runs"].values()])).sum()))
    return 1 if bad else 0
