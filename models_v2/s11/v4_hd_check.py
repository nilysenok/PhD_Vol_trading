"""V4-H: проверки гейтов G39–G41 для `run_v4 --stage H --check` — ТОЛЬКО ЧТЕНИЕ (G38): ни файла результатов, ни CHECKPOINT, ни ключа кэша не пишется; пересчёты — в памяти. Возвращают список нарушений."""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_cb_years as Y, v4_cld_brk as BK, v4_h_cache as HC, v4_h_feat as HF, v4_h_run as R, v4_h_stage as ST, v4_hd_report as RP, v4_hd_write as W, v4_io as IO, v4_isolation as ISO, v4_rqc_check as RQ)

DAYS = ("2019-06-03", "2021-09-01")
EXPECT_KEYS = RQ.EXPECT_KEYS + ST.CFGS


def prior_sha_without_stage_i() -> str:
    """Отступление (лист I): то же, что `v4_h_cache.prior_sha`, но файлы этапа I (I1–I3*, I_*) — не «прежние» (сам `v4_h_cache` не правится: он входит в хэш кода H)."""
    import hashlib
    import re
    h = hashlib.sha256()
    for f in sorted(HC.OUT.glob("*.parquet")):
        if not HC.OWN.match(f.name) and not re.match(r"^I(\d|_)", f.name):
            h.update(f.name.encode())
            h.update(IO.sha256(f).encode())
    return h.hexdigest()


def g39_forecasts(o: dict) -> list:
    bad, P = [], R.prepare(o["P"], o["Zh"], ["herf"])[0]
    for cfg in ST.CFGS:
        f, c = o["fc"][cfg], o["cf"][cfg]
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not f["g"].between(-0.5 - 1e-12, 0.5 + 1e-12).all():
            bad.append("G39: %s — прогнозы неполны / не конечны / g вне [−0,5; 0,5]" % cfg)
        if not c["converged"].all():
            bad.append("G39: %s — не все месячные оценки сошлись (%d из %d)" % (cfg, int(c["converged"].sum()), len(c)))
    k8 = np.array([math.sqrt(h * math.sqrt(a * b)) for h, a, b in zip(P["h"], P["k2"], P["k3"])])
    for cfg in ("H1", "H7", "H8"):                       # тождество θ ≡ 0 -> K8 побитно (боевой путь с обнулённым θ) и до 60 пар
        f0 = R.forecast(cfg, o["P"], o["Zh"], zero_theta=True)[0]
        if not np.array_equal(f0["F"].to_numpy(), k8):
            bad.append("G39: %s при θ ≡ 0 не равен K8 побитно" % cfg)
        f = o["fc"][cfg]
        if not np.array_equal(f.loc[f["g"] == 0.0, "F"].to_numpy(), k8[(f["g"] == 0.0).to_numpy()]):
            bad.append("G39: %s — строки с g = 0 не равны K8 побитно" % cfg)
    return bad


def g39_pit(o: dict) -> list:
    """Обрезка боевого пути на реальных данных (подмена RV закрытых после day и признаков после day не меняет F) + BRK + изоляция LOTO + прежние прогнозы."""
    bad, fh = [], HF.firsth_table(IO.T17)
    for cfg in ("H1", "H2", "H7", "H8"):
        for day in DAYS:
            if R.pit_violation(cfg, o["P"], o["Zl"], fh, day):
                bad.append("G39: %s — обрезка нарушена (day %s)" % (cfg, day))
    brk = pd.read_parquet(HC.OUT / "BRK.parquet")
    if not np.array_equal(brk["brk"].to_numpy(), HF.date_level(o["Zl"])["brk"].to_numpy()):
        bad.append("G39: brk (H6) не равен BRK.parquet")
    lt = IO.read_csv("H_LOTO.csv")
    if set(lt["config"]) != {o["best"]} or set(lt["held_out"]) != set(IO.M16 + ["M16"]) or not lt["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G39: H_LOTO неполон или исключённая бумага влияет на панель")
    cp = o["cp"]
    if cp.get("H_prior_sha") != prior_sha_without_stage_i():
        bad.append("G39: прежние прогнозы/Z/BRK изменились после объявления (хэш parquet ≠ записанному)")
    t = dt.datetime.strptime(cp["H_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    early = [f.name for f in HC.OUT.glob("*.parquet") if HC.OWN.match(f.name) and f.stat().st_mtime <= t]
    if early:
        bad.append("G39: файлы H созданы не позже объявления: %s" % early[:5])
    return bad + ["G39: oracle/диагностика DX импортированы путём прогноза: %s" % x for x in ISO.violators()]


def g40_tables(o: dict) -> list:
    W.add_derived(o)
    csv = ["H_COMPARISON.csv", "H_PER_TICKER.csv", "H_TESTS.csv", "H_YEARS.csv", "H_SBER.csv", "H_LOTO.csv", "H_VALUE.csv", "H_COEF.csv"]
    bad = [("G40: нет " + f) for f in csv if not (IO.REP / f).exists()]
    if bad:
        return bad
    comp, tst, cs = IO.read_csv("H_COMPARISON.csv"), IO.read_csv("H_TESTS.csv"), S.crit_panel(o["k"], o["cols"]).set_index("series")["crit_panel"]
    if len(comp) != len(o["cols"]) * 4 or IO.read_csv("H_PER_TICKER.csv")["ticker"].nunique() != 17:
        bad.append("G40: H_COMPARISON / H_PER_TICKER неполны")
    cc = comp[comp["window"] == "all"].set_index("series")["crit_panel"]
    if max(abs(cs[s] - cc[s]) for s in cs.index) > 1e-9:
        bad.append("G40: crit_panel в H_COMPARISON ≠ hp_stats_sel.crit_panel")
    if len(tst) != 8 * 5 * 4 or not (tst["p_holm_wide"].dropna() >= tst["p_holm"].dropna() - 1e-15).all():
        bad.append("G40: H_TESTS — размер или порядок p_holm ≤ p_holm_wide")
    yr = IO.read_csv("H_YEARS.csv")
    if max(abs(Y.crit_from_years(yr)[s] - cs[s]) for s in o["cols"]) > 1e-12 or not {2017, 2018} <= set(yr[yr["series"] == o["best"]]["year"]):
        bad.append("G40: H_YEARS — сумма crit_part ≠ crit_panel или нет 2017/2018")
    sb = IO.read_csv("H_SBER.csv")
    sa = sb[(sb["window"] == "all") & (sb["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G40: H_SBER — не 2271 даты или нет ориентира")
    v, cf = IO.read_csv("H_VALUE.csv"), IO.read_csv("H_COEF.csv")
    if len(v) != 8 * 4 or v["share_rows_out_of_range"].isna().any() or abs(v["d_qlike_k8"] - o["value"]["d_qlike_k8"]).max() > 1e-9:
        bad.append("G40: H_VALUE неполон или Δ к K8 не совпадает с пересчётом")
    if len(cf) != 8 * o["P"]["date"].dt.to_period("M").nunique():
        bad.append("G40: H_COEF — не 8 × число месяцев")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:") and not k.startswith("Ж-222:I")]       # отступление (лист I): строки I1–I3 не считаются
    if sorted(k.split(":")[1] for k in keys) != sorted(EXPECT_KEYS):
        bad.append("G40: ALL_TRIALS — строки Ж-222 не B…RQC + H1–H8 по одной (%d, ожидалось %d)" % (len(keys), len(EXPECT_KEYS)))
    return bad


def g41_report(o: dict) -> list:
    txt = (IO.REP / "H_REPORT.md").read_text()
    bad, cp = [], json.loads((IO.REP / "CHECKPOINT.json").read_text())
    if txt.splitlines()[0] != RP.first_line(o):
        bad.append("G41: первая строка H_REPORT не равна пересчитанной по crit_panel и порогу")
    if len(txt.split()) > 1000:
        bad.append("G41: H_REPORT %d слов > 1000" % len(txt.split()))
    if not all(x in txt for x in ("Условная ценность источников сверх K8", "2017", "2018", "SBER", "LOTO", "Отступления", "вне диапазона", "простое среднее QLIKE 2017–2021")):
        bad.append("G41: в H_REPORT нет обязательных разделов")
    g = o["go"]["cfg"][o["best"]]
    if g["go"] != (g["c1"] and g["c2"] and g["c3"]) or not txt.splitlines()[0].startswith("go" if g["go"] else "stop"):
        bad.append("G41: «go» не равно конъюнкции условий 1–3 / первая строка не по ним")
    if cp.get("H_stop_go") != ("go" if g["go"] else "stop") or cp.get("used_after_H") != 43 or cp.get("H_configs_spent") != 8 or cp.get("H_best") != o["best"]:
        bad.append("G41: CHECKPOINT — H_stop_go / used_after_H / H_configs_spent / H_best")
    return bad


def all_checks(o: dict) -> list:
    return g39_forecasts(o) + g39_pit(o) + g40_tables(o) + g41_report(o)
