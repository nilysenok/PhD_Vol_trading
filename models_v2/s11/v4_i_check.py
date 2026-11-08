"""V4-I: проверки гейтов G42–G44 для `run_v4 --stage I --check` — ТОЛЬКО ЧТЕНИЕ (G38): ни файла результатов, ни CHECKPOINT, ни ключа кэша не пишется; пересчёты — в памяти. Возвращают список нарушений."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_cb_years as Y, v4_hd_check as HK, v4_i_cache as IC_, v4_i_comp as COMP, v4_i_out as OUT, v4_i_run as R, v4_i_stage as ST, v4_i_write as W, v4_io as IO, v4_isolation as ISO)

DAYS = ("2019-06-03", "2021-09-01")
EXPECT_KEYS = HK.EXPECT_KEYS + ST.CFGS
CSV = ["I_COMPARISON.csv", "I_PER_TICKER.csv", "I_TESTS.csv", "I_YEARS.csv", "I_SBER.csv", "I_LOTO.csv", "I_COMP.csv"]


def g42_components(o: dict) -> list:
    bad = ["G42: C + J ≠ RV (сохранённый ряд) у %s" % t for t, ok in COMP.input_identity().items() if not ok]
    t = "SBER"
    d = IO.load_daily(t)
    if COMP.identity_ok(d["rv"], d["bv"], d["j"]):
        bad.append("G42: мутация «C := BV» не поймана тождеством")
    if not COMP.identity_ok(d["rv"], COMP.split(d["rv"], d["j"]), d["j"]):
        bad.append("G42: боевое разложение C = RV − J не проходит тождество")
    P, cmp, co = o["P"], o["cmp"], o["co"]
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    for cfg in ST.CFGS:
        f = o["fc"][cfg]
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all():
            bad.append("G42: %s — прогнозы неполны / не конечны" % cfg)
    if not co.loc[co["fitted"], "jump_ok"].all() or co["fitted"].sum() < 100:
        bad.append("G42: не все месячные оценки модели скачка сошлись / мало оценок")
    k8 = cmp["K8"].to_numpy()
    if not np.array_equal(R.combine("I3", cmp, co[["month", "theta"]], zero_theta=True)["F"].to_numpy(), k8):
        bad.append("G42: I3 при θ ≡ 0 не равен K8 побитно")
    nc = ~np.isfinite((cmp["chat"] + cmp["jhat"]).to_numpy(float))
    if not np.array_equal(o["fc"]["I1"]["F"].to_numpy()[nc], k8[nc]) or not np.array_equal(o["fc"]["I2"]["F"].to_numpy()[nc], k8[nc]):
        bad.append("G42: строки без компонентного прогноза (до 60 пар) не равны K8 побитно")
    if not (np.all(np.isfinite(cmp["chat"].to_numpy(float)[~nc])) and (cmp["jhat"].to_numpy(float)[~nc] > 0).all()):
        bad.append("G42: Ĉ/Ĵ вне области")
    return bad


def g42_pit(o: dict) -> list:
    """Обрезка боевого пути на реальных данных (подмена RV закрытых после day и дневных RV, J после day не меняет F) + прежние прогнозы + порядок объявления + изоляция."""
    bad = []
    for cfg in ("I1", "I3"):
        for day in DAYS:
            if R.pit_violation(cfg, o["P"], o["dl"], day):
                bad.append("G42: %s — обрезка нарушена (day %s)" % (cfg, day))
    lt = IO.read_csv("I_LOTO.csv")
    if set(lt["config"]) != {o["best"]} or set(lt["held_out"]) != set(IO.M16 + ["M16"]) or not lt["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G42: I_LOTO неполон или исключённая бумага влияет на панель")
    cp = o["cp"]
    if cp.get("I_prior_sha") != IC_.prior_sha():
        bad.append("G42: прежние прогнозы изменились после объявления (хэш parquet ≠ записанному)")
    t = dt.datetime.strptime(cp["I_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    early = [f.name for f in IC_.OUT.glob("*.parquet") if IC_.OWN.match(f.name) and f.stat().st_mtime <= t]
    if early:
        bad.append("G42: файлы I созданы не позже объявления: %s" % early[:5])
    return bad + ["G42: oracle/диагностика DX импортированы путём прогноза: %s" % x for x in ISO.violators()]


def g43_tables(o: dict) -> list:
    W.add_derived(o)
    bad = [("G43: нет " + f) for f in CSV if not (IO.REP / f).exists()]
    if bad:
        return bad
    comp, tst, cs = IO.read_csv("I_COMPARISON.csv"), IO.read_csv("I_TESTS.csv"), S.crit_panel(o["k"], o["cols"]).set_index("series")["crit_panel"]
    if len(comp) != len(o["cols"]) * 4 or IO.read_csv("I_PER_TICKER.csv")["ticker"].nunique() != 17:
        bad.append("G43: I_COMPARISON / I_PER_TICKER неполны")
    cc = comp[comp["window"] == "all"].set_index("series")["crit_panel"]
    if max(abs(cs[s] - cc[s]) for s in cs.index) > 1e-9:
        bad.append("G43: crit_panel в I_COMPARISON ≠ hp_stats_sel.crit_panel")
    if len(tst) != 3 * 5 * 4 or not (tst["p_holm_wide"].dropna() >= tst["p_holm"].dropna() - 1e-15).all():
        bad.append("G43: I_TESTS — размер или порядок p_holm ≤ p_holm_wide")
    yr = IO.read_csv("I_YEARS.csv")
    if max(abs(Y.crit_from_years(yr)[s] - cs[s]) for s in o["cols"]) > 1e-12 or not {2017, 2018} <= set(yr[yr["series"] == o["best"]]["year"]):
        bad.append("G43: I_YEARS — сумма crit_part ≠ crit_panel или нет 2017/2018")
    sb = IO.read_csv("I_SBER.csv")
    sa = sb[(sb["window"] == "all") & (sb["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G43: I_SBER — не 2271 даты или нет ориентира")
    ac = IO.read_csv("I_COMP.csv")
    if set(ac["scope"]) != {"M16", "SBER"} or len(ac) != 8 or ac["qlike_c"].isna().any() or abs(ac["qlike_c"].to_numpy() - o["acc"]["qlike_c"].to_numpy()).max() > 1e-9:
        bad.append("G43: I_COMP неполон или не совпадает с пересчётом")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:")]
    if sorted(k.split(":")[1] for k in keys) != sorted(EXPECT_KEYS):
        bad.append("G43: ALL_TRIALS — строки Ж-222 не B…H + I1–I3 по одной (%d, ожидалось %d)" % (len(keys), len(EXPECT_KEYS)))
    return bad


def g44_report(o: dict) -> list:
    txt = (IO.REP / "I_REPORT.md").read_text()
    bad, cp = [], json.loads((IO.REP / "CHECKPOINT.json").read_text())
    if txt.splitlines()[0] != OUT.first_line(o):
        bad.append("G44: первая строка I_REPORT не равна пересчитанной по crit_panel и порогу")
    if len(txt.split()) > 1000:
        bad.append("G44: I_REPORT %d слов > 1000" % len(txt.split()))
    if not all(x in txt for x in ("Точность компонент", "Учитывает ли HAR-J", "2017", "2018", "SBER", "LOTO", "Отступления", "простое среднее QLIKE 2017–2021")):
        bad.append("G44: в I_REPORT нет обязательных разделов")
    g = o["go"]["cfg"][o["best"]]
    if g["go"] != (g["c1"] and g["c2"] and g["c3"]) or not txt.splitlines()[0].startswith("go" if g["go"] else "stop"):
        bad.append("G44: «go» не равно конъюнкции условий 1–3 / первая строка не по ним")
    if cp.get("I_stop_go") != ("go" if g["go"] else "stop") or cp.get("used_after_I") != 46 or cp.get("I_configs_spent") != 3 or cp.get("I_best") != o["best"]:
        bad.append("G44: CHECKPOINT — I_stop_go / used_after_I / I_configs_spent / I_best")
    return bad


def all_checks(o: dict) -> list:
    return g42_components(o) + g42_pit(o) + g43_tables(o) + g44_report(o)
