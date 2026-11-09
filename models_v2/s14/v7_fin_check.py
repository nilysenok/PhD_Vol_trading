"""V7 лист FIN, G49–G52: проверки (ТОЛЬКО ЧТЕНИЕ) — диск против пересчёта в памяти, числа FIN_* против таблиц листов 2b/3/4a, реестр, CHECKPOINT, документ."""
from __future__ import annotations

import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_evrun as R1, v7_fin_asm as A, v7_fin_data as D, v7_fin_docheck as DC, v7_fin_stage as ST, v7_fin_tabs as TB

TRIALS = IO.REP / "ALL_TRIALS.csv"
LEAF = {"cfg%d" % i: "2" for i in range(1, 8)} | {"cfg8": "3", "cfg10": "4"}


def disk_vs_memory(r: dict) -> list:
    bad = []
    for n in (x for x in D.TABLES if x != "FIN_DEPARTURES"):          # журнал только дополняется: записи сверяет `departures`
        p = IO.REP / (n + ".csv")
        if not p.exists() or not R1.same_frame(r["t"][n], IO.read_csv(n + ".csv")):
            bad.append("G49: %s.csv: нет файла или диск ≠ пересчёту" % n)
    return bad


def vs_sheets(r: dict) -> list:
    """G49: пересчитанные FIN_* = таблицам листов (метрики, MCS, стресс 4; тесты 2b/3/4a по конфигурациям)."""
    t, bad = r["t"], []
    for fin, leaf in (("FIN_ALL", "4_METRICS.csv"), ("FIN_MCS", "4_MCS.csv"), ("FIN_STRESS", "4_STRESS.csv")):
        if not R1.same_frame(t[fin], IO.read_csv(leaf)):
            bad.append("G49: %s ≠ %s" % (fin, leaf))
    for c, st in LEAF.items():
        sh = IO.read_csv(st + "_TESTS.csv")
        sh = sh[sh["config"] == c] if "config" in sh.columns else sh
        if not R1.same_frame(t["FIN_TESTS"][t["FIN_TESTS"]["config"] == c], sh):
            bad.append("G49: FIN_TESTS.%s ≠ %s_TESTS.csv" % (c, st))
    return bad


def departures(r: dict) -> list:
    """Записи журнала в FIN_DEPARTURES дословно содержатся в журнале сейчас (журнал только дополняется)."""
    cur, old = set(D.departures()["text"]), IO.read_csv("FIN_DEPARTURES.csv") if (IO.REP / "FIN_DEPARTURES.csv").exists() else pd.DataFrame(columns=["text"])
    return ["G49: запись журнала [V7] в FIN_DEPARTURES не найдена в журнале: %s" % x[:60] for x in old["text"] if x not in cur]


def registry(r: dict) -> list:
    """G51: ALL_TRIALS — 11 мест без дублей ключей, 9 посчитаны, 9 и 11 «не считается», версия 1 в архиве у 1–7, резерв не использован; ключ FIN = пересчёт; другие ключи на месте; REPRODUCTION со всеми стадиями."""
    reg, bad = r["t"]["FIN_REGISTRY"], []
    tr = pd.read_csv(TRIALS)
    if tr["key"].duplicated().any():
        bad.append("G51: дубли ключей в ALL_TRIALS.csv")
    if tr["key"].str.startswith("Ж-233:12").any() or len(reg) != 11:
        bad.append("G51: резерв использован или мест не 11")
    st = reg.set_index("place")["status"]
    if int((st == "посчитана").sum()) != 9 or not all(str(st[i]).startswith("не считается") for i in (9, 11)):
        bad.append("G51: 9 посчитаны, 9 и 11 «не считается»")
    if not reg[reg["place"] <= 7]["v1_archived"].astype(str).str.lower().eq("true").all():
        bad.append("G51: версия 1 конфигураций 1–7 не в архиве")
    cp = D.read_cp()
    want = ST.fin_entry(cp, len(r["d"]["rv"]))
    if "FIN" not in cp or ST.stable(cp["FIN"]) != ST.stable(want):
        bad.append("G51: ключ FIN на диске ≠ пересчёту")
    bad += ["G51: нет ключа %s" % k for k in ("0", "1a", "1b", "1b_v1", "2", "2_v1", "3", "4", "DX") if k not in cp]
    txt = A.REPRO.read_text(encoding="utf-8") if A.REPRO.exists() else ""
    return bad + ["G51: REPRODUCTION_V7.md без стадии %s" % s for s in A.STAGES if "--stage %s --check" % s not in txt]


def doc(r: dict) -> list:
    t = A.load()
    bad = DC.check_doc(t) + DC.check_static(t)
    if not A.PNG.exists() or A.PNG.stat().st_size < 1000:
        bad.append("G50: нет FIN_WEIGHTS.png")
    if "FIN_WEIGHTS.png" not in A.FINAL.read_text(encoding="utf-8"):
        bad.append("G50: в документе нет ссылки на FIN_WEIGHTS.png")
    return bad


def n_numbers() -> int:
    return DC.count_numbers(A.load())


def all_bad(r: dict) -> list:
    return disk_vs_memory(r) + vs_sheets(r) + departures(r) + registry(r) + doc(r)
