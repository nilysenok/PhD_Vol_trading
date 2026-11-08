"""V4-FIN: проверки G45–G47 для `run_v4 --stage FIN --check` — ТОЛЬКО ЧТЕНИЕ: ни одного файла записи (ни таблиц, ни документа, ни CHECKPOINT). Пересчёт — в памяти (`stage_fin(write=False)`), сверка — с диском. Возвращают список нарушений."""
from __future__ import annotations

import datetime as dt
import glob

import numpy as np
import pandas as pd

from models_v2.s11 import v4_cb_years as Y, v4_fin_data as FD, v4_fin_doc as DOC, v4_fin_stage as ST, v4_fin_write as W, v4_io as IO

METRICS = ("qlike", "rmse", "mae", "r2_rv", "r2_ln", "crit_panel")
TOL = 1e-9
DOC_PARTS = ["Ж-225 п. 9", "Три лучшие архитектуры", "Критерий выбора и находка о весах годов", "Все 46 архитектур", "Математические определения", "Аудит данных", "Oracle bound", "QLIKE по 17 бумагам", "SBER и ориентир 0,2081",
             "Сравнение со всеми базовыми", "Динамика весов", "Эффект режимов", "Причины выигрыша и проигрыша", "Статистические проверки", "Реестр испытаний", "Результаты по годам", "CHECKPOINT и воспроизводимость",
             "ACCEPTED не присваивается", "не оценивалась", "не независимый OOS", "простое среднее", "Примечание о номере", "потолок", "2017", "2022–2025"]


def frames_equal(disk: pd.DataFrame, mem: pd.DataFrame, tol: float = TOL) -> bool:
    if disk.shape != mem.shape or list(disk.columns) != list(mem.columns):
        return False
    for c in disk.columns:
        a, b = disk[c], mem[c]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=0.0, atol=tol, equal_nan=True):
                return False
        elif not a.fillna("").astype(str).str.lower().reset_index(drop=True).equals(b.fillna("").astype(str).str.lower().reset_index(drop=True)):
            return False
    return True


def g45_tables(o: dict) -> list:
    bad = [("G45: нет " + f) for f in ST.OUTS if not (IO.REP / f).exists()]
    if bad:
        return bad
    for f, k in ST.OUTS.items():
        if not frames_equal(IO.read_csv(f), o[k]):
            bad.append("G45: %s не совпадает с пересчётом в памяти" % f)
    a = IO.read_csv("FIN_ALL.csv")
    if set(a["series"]) != set(FD.CFGS + FD.CTRL) or len(FD.CFGS) != 46 or len(a) != 50 * 6 or set(a["window"]) != set(DOC.WN):
        bad.append("G45: FIN_ALL — не 46 конфигураций + 4 контроля × 6 окон")
    bad += cross_stage(a)
    t = IO.read_csv("FIN_TESTS.csv")
    if t.empty or not (t["p_holm_wide"].dropna() >= t["p_holm"].dropna() - 1e-15).all():
        bad.append("G45: FIN_TESTS — пусто или p_holm_wide < p_holm")
    s = IO.read_csv("FIN_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G45: FIN_SBER — не 2271 даты или нет ориентира 0,20814")
    yr, cs = IO.read_csv("FIN_YEARS.csv"), a[a["window"] == "all"].set_index("series")["crit_panel"]
    if max(abs(Y.crit_from_years(yr)[x] - cs[x]) for x in cs.index) > 1e-12 or not {2017, 2018} <= set(yr["year"]) or yr["year"].nunique() != 10:
        bad.append("G45: FIN_YEARS — сумма crit_part ≠ crit_panel или нет годов 2017…2026")
    return bad + keep_checks(o)


def cross_stage(fin: pd.DataFrame) -> list:
    """Числа FIN_ALL = числа таблиц этапов (`*_COMPARISON.csv`), допуск 1e-9; сверено должно быть не меньше 2000 ячеек."""
    f, n, bad = fin.set_index(["series", "window"]), 0, []
    for p in sorted(glob.glob(str(IO.REP / "*_COMPARISON.csv"))):
        st = pd.read_csv(p, comment="#")
        for r in st.itertuples():
            if (r.series, r.window) not in f.index:
                continue
            for c in METRICS:
                if c in st.columns and pd.notna(getattr(r, c)):
                    n += 1
                    if abs(float(getattr(r, c)) - float(f.loc[(r.series, r.window), c])) > TOL:
                        bad.append("G45: %s %s %s: FIN %.9f ≠ этап %.9f" % (p.split("/")[-1], r.series, r.window, f.loc[(r.series, r.window), c], getattr(r, c)))
    return bad[:5] + ([] if n >= 2000 else ["G45: сверено только %d ячеек с таблицами этапов" % n])


def keep_checks(o: dict) -> list:
    """Прежние прогнозы и таблицы этапов побитно прежние; строки реестра — ровно 46; FIN объявлен раньше своих файлов."""
    cp, bad = o["cp"], []
    if cp.get("FIN_prior_parquet_sha") != FD.parquet_sha():
        bad.append("G45: parquet программы изменились после объявления FIN")
    if cp.get("FIN_prior_tables_sha") != FD.tables_sha():
        bad.append("G45: прежние таблицы этапов изменились после объявления FIN")
    keys = sorted(k.split(":")[1] for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:"))
    if keys != sorted(FD.CFGS):
        bad.append("G45: ALL_TRIALS — строки Ж-222 не равны 46 конфигурациям (%d)" % len(keys))
    t0 = dt.datetime.strptime(cp["FIN_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    late = [f for f in list(ST.OUTS) + ["REPRODUCTION_V4.md"] if (IO.REP / f).stat().st_mtime <= t0]
    if late or W.FINAL.stat().st_mtime <= t0:
        bad.append("G45: файлы FIN созданы не позже объявления: %s" % late)
    if cp.get("FIN_configs_spent") != 0:
        bad.append("G45: FIN расходует конфигурации")
    return bad


def g46_doc(o: dict) -> list:
    if not W.FINAL.exists() or not W.REPRO.exists():
        return ["G46: нет документа или REPRODUCTION_V4.md"]
    txt, rep = W.FINAL.read_text(encoding="utf-8"), W.REPRO.read_text(encoding="utf-8")
    bad = ["G46: в документе нет «%s»" % p for p in DOC_PARTS if p not in txt]
    for st in ("audit", "A", "B", "Ca", "Cb", "E", "DX", "CS", "NTP", "CLD", "RQC", "H", "I", "FIN"):
        if "--stage %s --check" % st not in rep:
            bad.append("G46: REPRODUCTION_V4.md — нет команды стадии " + st)
    if not txt.splitlines()[0] or txt.startswith("#"):
        bad.append("G46: первая строка документа — не главный вывод")
    return bad


def g47_numbers(o: dict) -> list:
    """Главная строка и таблицы документа = пересобранные из FIN_*.csv на диске И из таблиц в памяти; чисел в проверенных фрагментах не меньше 30."""
    txt, t = W.FINAL.read_text(encoding="utf-8"), DOC.tables_from_disk()
    mem = {k: o[k] for k in ("all", "tickers", "sber", "tests", "mcs", "years")} | {"cp": o["cp"], "top": o["top"]}
    bad = ["G47: фрагмент документа не равен пересобранному из FIN_*.csv: %s…" % c[:60].replace("\n", " ") for c in DOC.missing_claims(txt, t)]
    if DOC.claims(mem) != DOC.claims(t):
        bad.append("G47: фрагменты по таблицам в памяти и на диске различаются")
    if txt.splitlines()[0] != DOC.first_line(t):
        bad.append("G47: первая строка документа ≠ главная строка по FIN_ALL.csv")
    if sum(DOC.numbers_in(c) for c in DOC.claims(t)) < 30:
        bad.append("G47: сверено меньше 30 чисел")
    if o["top"] != t["top"] or o["cp"].get("FIN_top3") != t["top"]:
        bad.append("G47: три лучшие в памяти / на диске / CHECKPOINT различаются")
    return bad


def all_checks(o: dict) -> list:
    return g45_tables(o) + g46_doc(o) + g47_numbers(o)
