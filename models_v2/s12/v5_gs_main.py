"""V5 лист 1a: запуск стадии GS (из `run_v5 --stage GS`). Процесс один.
  --cfg GSn          посчитать и записать кэш одной конфигурации (GS3–GS5 требуют записанной формы)
  --decide-form      записать в CHECKPOINT.json форму score по GS1, GS2 (до счёта GS3–GS5)
  (без флагов)       собрать таблицы, ALL_TRIALS, CHECKPOINT
  --check            ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, суммарный хэш выходов до и после"""
from __future__ import annotations

import numpy as np

from models_v2.s12 import v5_0_io as IO, v5_gs_panel as PA, v5_gs_stage as ST


def evidence(r: dict) -> list:
    t, cp = r["t"], r["cp"]
    go, cs = t["_go"], t["_crit"]
    gs = go[go["series"].isin(ST.GS)].set_index("series")
    enc = t["GS_ENCOMPASS"]
    e = enc[(enc["window"] == "2022_2025")].set_index("base")
    lines = ["G12: форма score %s (crit GS1 %.6f, GS2 %.6f); расход %s; направление 1: %s; лучшая %s" % (cp["GS"]["form"], *cp["GS"]["form_crit_panel"].values(), cp["GS"].get("used"), t["GS_COMPARISON"]["decision_direction1"].iloc[0], t["_best"])]
    lines += ["G12: %s crit_panel %.6f (порог %.6f), без 2017 %.6f (%.6f), без 1 %% %.6f (%.6f); go %s; EqY %.6f" % (s, v.crit_panel, v.thr1, v.crit_no2017, v.thr2, v.crit_trim1, v.thr3, v.go, v.EqY) for s, v in gs.iterrows()]
    lines.append("G14: w(M) 2022–2025 %.3f [%.3f; %.3f]; w(K8) %.3f [%.3f; %.3f]" % (e.loc["M", "w_hat"], e.loc["M", "w_lo90"], e.loc["M", "w_hi90"], e.loc["K8", "w_hat"], e.loc["K8", "w_lo90"], e.loc["K8", "w_hi90"]))
    lines.append("кэш: %s" % {g: v["cached"] for g, v in r["runs"].items()})
    return lines


def main(a) -> int:
    before = IO.outputs_digest()
    if a.cfg or a.decide_form:
        P, k = PA.build(), IO.frame()
        if a.decide_form:
            print("форма score:", ST.decide(P, k, True))
        else:
            from models_v2.s12 import v5_gs_run as RUN
            v = RUN.get(P, a.cfg, True)
            print(a.cfg, "записан; строк", len(v["main"]), "ключ", v["key"], "из кэша", v["cached"])
        return 0
    r = ST.compute(write=not a.check)
    if not a.check:
        ST.write(r)
        print("стадия GS записана; хэш выходов %s" % IO.outputs_digest())
        return 0
    bad = ST.check(r)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш до %s, после %s" % (before, after))
    print("\n".join(evidence(r)))
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G12–G14: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK", "| бумаг в панели", len(r["P"].tick), "| NaN в F:", int(np.isnan(np.concatenate([v["main"]["F"].to_numpy() for v in r["runs"].values()])).sum()))
    return 1 if bad else 0
