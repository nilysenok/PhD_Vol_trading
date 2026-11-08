"""V4-RQC: проверки гейтов G35–G37 после записи стадии RQC (`run_v4 --stage RQC --check`). G35 — `v4_rqc_chk35`. Возвращают список нарушений (пусто = OK)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_cb_years as Y, v4_io as IO, v4_rqc_cache as RC, v4_rqc_chk35 as C35, v4_rqc_report as RP, v4_rqc_stage as ST

CSV = ["RQC_COMPARISON.csv", "RQC_PER_TICKER.csv", "RQC_TESTS.csv", "RQC_YEARS.csv", "RQC_SBER.csv", "RQC_LOTO.csv", "RQC_A.csv"]
EXPECT_KEYS = ["B1", "B2", "B3", "B4", "B6"] + ["C%d" % i for i in range(1, 11)] + ST.STAGE_W["E"] + ["CS1", "CS2"] + ["NTP%d" % i for i in range(1, 5)] + ["CLD%d" % i for i in range(1, 5)] + ST.CFGS


def g36_tables(o: dict) -> list:
    bad = [("G36: нет " + f) for f in CSV if not (IO.REP / f).exists()]
    if bad:
        return bad
    comp, tst, cs = IO.read_csv("RQC_COMPARISON.csv"), IO.read_csv("RQC_TESTS.csv"), S.crit_panel(o["k"], o["cols_all"]).set_index("series")["crit_panel"]
    if len(comp) != len(o["cols_all"]) * 4 or len(IO.read_csv("RQC_PER_TICKER.csv")["ticker"].unique()) != 17:
        bad.append("G36: RQC_COMPARISON / RQC_PER_TICKER неполны")
    cc = comp[comp["window"] == "all"].set_index("series")["crit_panel"]
    if max(abs(cs[s] - cc[s]) for s in cs.index) > 1e-9:
        bad.append("G36: crit_panel в RQC_COMPARISON ≠ hp_stats_sel.crit_panel")
    if len(S.common(o["k"], o["cols_main"])) != len(o["c"]):
        bad.append("G36: общие даты основных рядов ≠ общим датам всех рядов (сравнение не на одинаковых датах)")
    if len(tst) != (4 + 3 * 5) * 4 or not (tst["p_holm_wide"].dropna() >= tst["p_holm"].dropna() - 1e-15).all():
        bad.append("G36: RQC_TESTS — размер или порядок p_holm ≤ p_holm_wide")
    yr = IO.read_csv("RQC_YEARS.csv")
    if max(abs(Y.crit_from_years(yr)[s] - cs[s]) for s in o["cols_main"]) > 1e-12 or not {2017, 2018} <= set(yr[yr["series"] == "RQC3"]["year"]):
        bad.append("G36: RQC_YEARS — сумма crit_part ≠ crit_panel или нет 2017/2018")
    sb = IO.read_csv("RQC_SBER.csv")
    sa = sb[(sb["window"] == "all") & (sb["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G36: RQC_SBER — не 2271 даты или нет ориентира")
    lt = IO.read_csv("RQC_LOTO.csv")
    if set(lt["config"]) != {o["best"]} or set(lt["held_out"]) != set(IO.M16 + ["M16"]) or not lt["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G36: RQC_LOTO неполон или исключённая бумага в панели")
    a = IO.read_csv("RQC_A.csv")
    w = a[a["kind"] == "window"]
    if not {"RQC1", "RQC2", "RQC3", "E1", "E5"} <= set(w["config"]) or not {"2018-03..04 (санкции)", "2022-02-24..03-31"} <= set(w["month"]) or not (a["kind"] == "month").any():
        bad.append("G36: RQC_A — нет месяцев/окон 2018 и 2022 или E1–E5")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:") and not k.startswith(("Ж-222:H", "Ж-222:I"))]       # отступление (лист H): строки H1–H8 не считаются
    if sorted(k.split(":")[1] for k in keys) != sorted(EXPECT_KEYS):
        bad.append("G36: ALL_TRIALS — строки Ж-222 не B…CLD + K8, RQC1–RQC3 по одной (%d, ожидалось %d)" % (len(keys), len(EXPECT_KEYS)))
    return bad


def g37_report(o: dict) -> list:
    txt = (IO.REP / "RQC_REPORT.md").read_text()
    bad, cp = [], json.loads((IO.REP / "CHECKPOINT.json").read_text())
    if txt.splitlines()[0] != RP.first_line(o):
        bad.append("G37: первая строка RQC_REPORT не равна пересчитанной по crit_panel и порогу")
    if len(txt.split()) > 1000:
        bad.append("G37: RQC_REPORT %d слов > 1000" % len(txt.split()))
    if not all(x in txt for x in ("K8 по трём условиям", "2017", "2018", "2022", "SBER", "LOTO", "Отступления", "Устойчива ли калибровка после скачков", "простое среднее QLIKE 2017–2021")):
        bad.append("G37: в RQC_REPORT нет обязательных разделов")
    cs, g = o["crit"].set_index("series")["crit_panel"], o["go"]["cfg"][o["best"]]
    if g["go"] != (g["c1"] and g["c2"] and g["c3"]) or g["c1"] != bool(cs[o["best"]] <= cp["go_threshold_B"]) or not txt.splitlines()[0].startswith("go" if g["go"] else "stop"):
        bad.append("G37: «go» не равно конъюнкции условий 1–3 / первая строка не по ним")
    if cp.get("RQC_stop_go") != ("go" if g["go"] else "stop") or cp.get("used_after_RQC") != 35 or cp.get("RQC_configs_spent") != 4:
        bad.append("G37: CHECKPOINT — RQC_stop_go / used_after_RQC / RQC_configs_spent")
    if cp.get("RQC_prior_sha") != RC.prior_sha():
        bad.append("G37: прежние прогнозы изменились после объявления листа (хэш parquet ≠ записанному)")
    t = dt.datetime.strptime(cp["RQC_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    early = [f.name for f in list(RC.OUT.glob("RQC*.parquet")) + list(RC.OUT.glob("K8*.parquet")) if f.stat().st_mtime <= t]
    if early:
        bad.append("G37: файлы K8/RQC созданы не позже объявления: %s" % early)
    return bad


def all_checks(o: dict) -> list:
    return C35.all_g35(o) + g36_tables(o) + g37_report(o)
