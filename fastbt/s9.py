"""С9: доводка стратегий на разработке 2018–2021 (предварительно, пересчёт на финальном σ_pred).

python3 -m fastbt.s9 sources                 — σ-источники (наивная, seed, RVI) → data/fastbt/s9/
python3 -m fastbt.s9 cache [--workers 2]     — A (основной дизайн, после фильтра оборота) и
                                               C основного дизайна на каждом σ-источнике
python3 -m fastbt.s9 run [--boot-reps 1000]  — надстройки D/B/C, портфели P1–P3, T1–T2 →
                                               docs/tables/strat_dev/*.csv («;», десятичная запятая)

--scope dev (по умолчанию): σ-источники обрезаются по 31.12.2021, оцениваются годы 2018–2021,
валидация 2017…Y−1 (маска _year_bounds(...)[0]). Надстройки применяются к позиции A основного
дизайна того же года (fastbt/designs/final/main.yaml: ансамбль A + фильтр оборота); эталоны — A и
C (u02) основного дизайна на том же σ. Все окна, перцентили, медианы — только прошлые бары (сдвиг
на 1). Сетки — docs/otchety/dev_log_strategies.md, строка «С9 … сетки объявлены до запуска».

--scope test — тот же код на 2022–03.02.2026 (σ до TEST_CUT; окна валидации/калибровки
заморожены на 2018–2021 — на тесте выбора нет; σ_pred h=1 по умолчанию final3_full). Замок тот же, что у fastbt.dev
и fastbt.final_test: без docs/otchety/FINAL_DESIGN_strategies.md — отказ (fastbt.dev.TestLocked)
до чтения любых данных. Выбор B/C/D по С8 не повторяется: ключи берутся из таблиц разработки
(docs/tables/strat_dev/s9_*<суффикс>.csv). C (владелец 26.10, 10:00): основная строка — одна
точка TEST_C = C1 W504 p70 на h=5 (s9_C_osnovnaya), остальная сетка C — s9_C_chuvstvitelnost с
пометкой «чувствительность» (справочно, без выбора и проверки гипотез). Наивная √RV_w и RVI — отдельные файлы *_test.parquet
(sources --scope test), seed-σ в тесте нет. Выход — docs/tables/strat_test/s9_*<суффикс>_test.csv,
кэш и журнал — data/fastbt/s9/*<суффикс>_test.*.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import period_v2, zona
from fastbt.cache import MAIN, code_hash


def numba_cache_dir() -> Path:
    """Д5: каталог numba-кэша зависит от хэша кода fastbt/*.py — правка любого исходника строит кэш заново."""
    return MAIN / "data" / "fastbt" / "numba_cache" / code_hash()[:16]


os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir()))     # до первого импорта numba; явный env побеждает

ROOT =Path(__file__).resolve().parent.parent
REGEN = MAIN / "data" / "regen"
OUT = MAIN / "data" / "fastbt" / "s9"
TABLES = Path(os.environ.get("S9_TABLES", ROOT / "docs" / "tables" / "strat_dev"))
MAIN_YAML = ROOT / "fastbt" / "designs" / "final" / "main.yaml"
YEARS = (2018, 2019, 2020, 2021)
VAL_START = 2017
VAL_END = 0                               # 0 — окно валидации до Y−1 (разработка)
COMM = 0.0005
CUT = pd.Timestamp("2021-12-31")          # тест 2022–2026 закрыт: σ после этой даты не читается
SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
CATS = {"S1": 1, "S2": 1, "S3": 0, "S4": 0, "S5": 2, "S6": 2}   # 0 тренд, 1 контртренд, 2 диапазон
MARGIN = 0.2
SOURCES = {   # метка: (файл, столбец, горизонт)
    "hyb_h1": (REGEN / "sigma_pred_h1_final2.parquet", "sigma_pred_h1", "h1"),
    "hyb_h5": (REGEN / "sigma_pred_h5_final2.parquet", "sigma_pred_h5", "h5"),
    "harj_h1": (REGEN / "sigma_pred_h1_harj_final2.parquet", "sigma_pred_h1", "h1"),
    "naive_h1": (OUT / "sigma_naive_rvw.parquet", "sigma_pred_h1", "h1"),
    "seed42_h1": (OUT / "sigma_seed42_h1.parquet", "sigma_pred_h1", "h1"),
    "seed43_h1": (OUT / "sigma_seed43_h1.parquet", "sigma_pred_h1", "h1"),
    "seed44_h1": (OUT / "sigma_seed44_h1.parquet", "sigma_pred_h1", "h1"),
}
RVI = OUT / "rvi_1850.parquet"
# σ_pred гибрида h=1: final2 — финал второго круга (Д8.12, ночной С9); final3 — финал третьего
# круга D22_h1. Выбор — флаг --sigma-h1; у final3 имена таблиц, кэша и журнала с суффиксом _final3.
# FINAL — тот же D22_h1, собранный как final2 (sigma_pred_h1_v2.py --tag D22_h1 --step10), на весь
# период стратегий; на 2017–2021 побитно равен final3.
SIGMA_H1 = {"final2": REGEN / "sigma_pred_h1_final2.parquet",
            "final3": REGEN / "sigma_pred_h1_final3.parquet",
            "FINAL": REGEN / "sigma_pred_h1_FINAL.parquet",
            # тот же ряд final3, собранный кодом главы 3 на весь период (sigma_pred_h1_final3_full.py);
            # побитно = final3 (SHA 5bed7be9…), таблицы и кэш — с суффиксом _final3
            "final3_full": REGEN / "sigma_pred_h1_final3_full.parquet"}
# σ-набор v2 (раунд 22:20 / решение 22:30, агент СD, «до аудита»): выровненные σ S0
# (data/regen/ch4_dev, SHA — SHA256.txt) — гибрид, HAR-J (б), наивная; h=1 и h=5. Флаг --sigma-set v2.
# Прежние источники (SOURCES выше) не тронуты: без флага таблицы разработки воспроизводятся побайтно.
# В v2: W504 с разгоном 252 → 504 (Ъ владельца, вариант (а)); D1/D2/D4/D7 — плюс W504; p к A —
# Ледуа–Вольф (блок 5), IID справочно; таблицы — docs/tables/strat_dev_v2/s9_*_v2.csv, столбец pometka.
CH4 = REGEN / "ch4_dev"
SOURCES_V2 = {
    "hyb_h1": (CH4 / "sigma_hyb_h1_dev.parquet", "sigma_pred_h1", "h1"),
    "hyb_h5": (CH4 / "sigma_hyb_h5_dev.parquet", "sigma_pred_h5", "h5"),
    "harj_h1": (CH4 / "sigma_harj_b_h1_dev.parquet", "sigma_pred_h1", "h1"),
    "harj_h5": (CH4 / "sigma_harj_b_h5_dev.parquet", "sigma_pred_h5", "h5"),
    "naive_h1": (CH4 / "sigma_naive_dev.parquet", "sigma_pred_h1", "h1"),
    "naive_h5": (CH4 / "sigma_naive_dev.parquet", "sigma_pred_h1", "h5"),   # √RV_w от h не зависит
}
V2 = False
POMETKA = "после аудита (Н1–Н5, H3, ключи кэшей)"     # ЭТ, 26.10: таблицы v2 на коде 9b85bff
# Ъ 23:10, п. 3: проверка RVI на сдвиг (S0, PR #245) — пометка в таблицах выбора D
D_RVI = ("RVI проверен на сдвиг (S0, PR #245): rvi_1850 снят до 18:50 дня d, на закрытии d известен; "
         "D6lag1 берёт RVI_{d−1}; D фиксируется на «да» владельца")
# Ъ 23:10, п. 1: H2 «лучший с лучшим» — отбор по С8 отдельно для каждой σ
H2_GROUPS = (("гибрид", ("hyb_h1", "hyb_h5")), ("HAR-J (б)", ("harj_h1", "harj_h5")),
             ("наивная", ("naive_h1", "naive_h5")))
# Ъ 27.10, 00:00, п. 1: в H2 «лучший с лучшим» — без дизайнов с внутренним выбором по валидации
# (u02 — выбор маски внутри; семьи — выбор точки по валидации на каждый тикер-год)
H2_VNUTR_VYBOR = ("C основного дизайна", " (выбор по валидации)")
# Ъ 23:10, п. 2: устойчивость выбора C — точка первого FINAL и допуск по Шарпу
C_FINAL1 = ("C1", "W504_p70", "hyb_h5")
USTOY_TOL = 0.1
TOL_S = "0,1"
# Ъ 22:45: комиссия v2 — 0,04% за сторону «всё включено»; чувствительность 0,035 / 0,05 / 0,06%
COMM_V2 = 0.0004
T1_COMMS_V2 = (0.00035, 0.0004, 0.0005, 0.0006)
T1_PODPIS = {35: "0,035% — институциональный инвестор",
             40: "0,04% — основная: тариф «Премиум» Т-Банка, биржевой сбор включён",
             50: "0,05% — тариф «Трейдер» Т-Банка", 60: "0,06% — консервативная"}
TABLES_V2 = ROOT / "docs" / "tables" / "strat_dev_v2"
SUFFIX = ""
SCOPE = "dev"
TABLES_DEV = ROOT / "docs" / "tables" / "strat_dev"      # источник выбора по С8 для теста
T2_CUT, T2_LABELS = "2020-03-01", ("2018–2021", "до 1.03.2020", "с 1.03.2020")
# тест: σ_pred h=1/h=5 и HAR-J — до 03.02.2026; 2026 — неполный год (январь)
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
TEST_CUT = pd.Timestamp("2026-02-03")
# владелец 26.10, 10:05, п. 10/13 (толкование дирижёра 10:11): на тесте выбора нет — окна валидации
# и калибровки заморожены на разработке 2018–2021 (2017 — прогрев), пустых окон нет
TEST_VAL_START, TEST_VAL_END = 2018, 2021
TEST_TABLES = Path(os.environ.get("S9_TABLES", ROOT / "docs" / "tables" / "strat_test"))
# владелец 26.10, 10:00: на тест из сетки C — одна точка C1 W504 p70 на h=5 (выбрана по С8 на разработке)
TEST_C = ("C1:W504_p70", "hyb_h5")
# Ъ 26.10, 22:30: новые точки сетки C (C2W — частичная 1→0,5→0, C*1/C*2 — ранг σ_pred/√RV_w),
# fastbt/designs/s9/setka_C.yaml, раздел dop; включаются флагом run --c-dop. None — сетка как была.
C_DOP_YAML = ROOT / "fastbt" / "designs" / "s9" / "setka_C.yaml"
C_DOP = None
# Ъ 27.10, 00:10 (агент К): флаг run --c8v2 — сетка C + p50/p60 (setka_C.yaml: dop.rassh, без семей
# «выбор по валидации»), контроли без прогноза к2 (круговой сдвиг)/к3 (A без S5/S6) в каждой строке C (fastbt.fast_kontrol) и правило
# С8 v2 = С8 И Шарп > Шарп к3 И p к к2 < 0,05 (Ъ 27.10, 01:30); выбор — среди прошедших (rank_c8_v2). False — как было.
C8V2 = False
C8V2_ALPHA = 0.05
POMETKA_C8V2 = "после аудита; сетка и контроли добавлены после просмотра разработки"
# таблицы, которых расширение и выбор C не касаются (пометка прежняя — побайтная сверка)
C8V2_NE_ZATR = ("s9_D_zatvor.csv", "s9_B_targetirovanie.csv", "s9_D_zatvor_harj.csv",
                "s9_P1_vesa_tikerov.csv", "s9_P3_fko.csv")


def set_scope(scope: str, val_start: int | None = None) -> None:
    """Переключает годы, срез σ, источники и папку таблиц. test — только при снятом замке.
    Окно валидации/калибровки в тесте заморожено: TEST_VAL_START…TEST_VAL_END (2018–2021)."""
    global SCOPE, YEARS, VAL_START, VAL_END, CUT, RVI, TABLES, T2_CUT, T2_LABELS
    if scope == "dev":
        return
    from fastbt import dev
    dev.check_scope(scope)                 # без FINAL_DESIGN_strategies.md — TestLocked
    SCOPE, YEARS, CUT = scope, TEST_YEARS, TEST_CUT
    VAL_START, VAL_END = val_start or TEST_VAL_START, TEST_VAL_END
    SOURCES["naive_h1"] = (OUT / "sigma_naive_rvw_test.parquet", "sigma_pred_h1", "h1")
    for s in (42, 43, 44):                 # seed-σ — реализации D12_h1 на 2017–2021
        SOURCES.pop(f"seed{s}_h1", None)
    RVI = OUT / "rvi_1850_test.parquet"
    TABLES = TEST_TABLES
    T2_CUT, T2_LABELS = "2025-01-01", ("2022–2026", "2022–2024", "2025–01.2026")


PERIOD = "train"
# Ж-075 п. 1: период В1 — годы метрик и начало окна валидации (= год прогрева); путь A без σ-источников
PERIODS_S9 = {"train": ((2018, 2019, 2020, 2021), 2017), "train4": ((2016, 2017, 2018, 2019), 2015),
              "v2": (period_v2.YEARS, period_v2.VAL_START),      # Ж-113: 2015–2019, прогрев с 26.08.2014
              "test1": ((2022, 2023, 2024), 2021)}      # test1 — реверс раунд 1 (Ж-117): зона Z2, прогрев 2021


def set_period(period: str) -> None:
    """train — как было (умолчание). train4 — В1: 2016–2019, прогрев 2015; v2 — 2015–2019, прогрев с 26.08.2014;
    test1 — 2022–2024, прогрев 2021 (зона Z2);
    только dev, только путь A / «A только Sk»."""
    global PERIOD, YEARS, VAL_START
    if period not in PERIODS_S9:
        raise ValueError(f"период {period!r}: есть {sorted(PERIODS_S9)}")
    if SCOPE != "dev":
        raise ValueError("период train4/test1 — только в разработке (scope dev)")
    PERIOD = period
    YEARS, VAL_START = PERIODS_S9[period]
    zona.set_zone("test1" if period == "test1" else "dev")


@contextlib.contextmanager
def period_ctx(period: str):
    """Период на время блока; на выходе (в т.ч. при ошибке) возвращает прежние PERIOD/YEARS/VAL_START (#418) и зону."""
    old = PERIOD, YEARS, VAL_START
    set_period(period)
    try:
        yield
    finally:
        _restore_period(*old)


def _restore_period(period, years, val_start) -> None:
    global PERIOD, YEARS, VAL_START
    PERIOD, YEARS, VAL_START = period, years, val_start
    zona.set_zone("test1" if period == "test1" else "dev")


def _latest_name() -> str:
    """Имя указателя на последний кэш; train4 — свой указатель, кэш train не затирается."""
    return f"cache_latest{_csuf()}{'' if PERIOD == 'train' else '_' + PERIOD}.txt"


def need_sigma(what: str) -> None:
    """B/C/D читают σ-источники (SOURCES с 2017-01-03): на train4 — отказ, а не тихий пропуск."""
    if PERIOD != "train":
        raise NotImplementedError(f"{what}: на периоде {PERIOD} σ-источников нет (с 2017-01-03); считается только путь A")


def use_sigma_h1(name: str, suffix: str | None = None) -> None:
    global SUFFIX
    SOURCES["hyb_h1"] = (SIGMA_H1[name], "sigma_pred_h1", "h1")
    tag = name.replace("_full", "")
    SUFFIX = ("" if tag == "final2" else f"_{tag}") if suffix is None else suffix
    if SCOPE != "dev":
        SUFFIX += f"_{SCOPE}"


def _csuf() -> str:
    """Суффикс кэша: в v2 один кэш на все комиссии (сигналы и A от комиссии МСП не зависят)."""
    return "_v2" if V2 else SUFFIX


def use_v2(comm: float | None = None) -> dict:
    """σ-набор v2: входы сверены с data/regen/ch4_dev/SHA256.txt до счёта; только разработка.
    comm — комиссия МСП и выбора на валидации (по умолчанию COMM_V2); не 0,04% — суффикс _k<…>."""
    global V2, TABLES, SUFFIX, COMM
    if SCOPE != "dev":
        raise SystemExit("--sigma-set v2 — только --scope dev (тест закрыт)")
    ref = {}
    for ln in (CH4 / "SHA256.txt").read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            ref[Path(parts[1]).name] = parts[0]
    sha = {}
    for lab, (p, _, _) in SOURCES_V2.items():
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if ref.get(Path(p).name) != h:
            raise SystemExit(f"SHA σ не совпал с SHA256.txt: {Path(p).name}")
        sha[lab] = h
    V2, TABLES = True, TABLES_V2
    COMM = COMM_V2 if comm is None else comm
    SUFFIX = "_v2" if abs(COMM - COMM_V2) < 1e-12 else f"_v2_k{COMM * 100:.3f}".replace(".", "")
    SOURCES.clear()
    SOURCES.update(SOURCES_V2)
    return sha


def _minp(W: int) -> int:
    """v2: W504 с разгоном 252 → 504 (растущее окно); иначе — s9_stats.minp."""
    from fastbt.s9_stats import minp
    return 252 if (V2 and W == 504) else minp(W)


def _past_median(s: np.ndarray, W: int) -> np.ndarray:
    from fastbt.s9_stats import past_median
    if V2 and W == 504:
        return pd.Series(s).shift(1).rolling(W, min_periods=_minp(W)).median().values
    return past_median(s, W)


def _d_windows() -> tuple:
    return (126, 252, 504) if V2 else (126, 252)


def to_csv(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    if V2:
        df = df.copy()
        # --c8v2: таблицы, на которые влияют сетка C и выбор C, — с пометкой Ъ 27.10, 00:10
        df.insert(0, "pometka", POMETKA_C8V2 if (C8V2 and name not in C8V2_NE_ZATR) else POMETKA)
    if SUFFIX:
        name = name.replace(".csv", f"{SUFFIX}.csv")
    p = TABLES / name
    df.to_csv(p, sep=";", decimal=",", index=False, float_format="%.4f",
              encoding="utf-8-sig" if SUFFIX else "utf-8")      # новые таблицы — с BOM
    return p


# ======================= σ-источники =======================

def build_sources() -> None:
    """Наивная sqrt(RV_w) (строка PIT дня t: RV по t−1), seed 42/43/44 финалиста h=1
    (sqrt(rv_pred·F года), val-прогнозы XGB, шаги 2–6 = 2017–2021), RVI (срез 18:50)."""
    need_sigma("build_sources")
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(REGEN / "features_pit_v3.parquet",
                        columns=["date", "ticker", "rv_w", "close_RVI"])
    f["date"] = pd.to_datetime(f["date"])
    f = f[(f["date"] >= "2017-01-03") & (f["date"] <= CUT)]
    nv = f[["date", "ticker"]].copy()
    nv["sigma_pred_h1"] = np.sqrt(f["rv_w"].astype(float)).astype("float32")
    nv.dropna().to_parquet(SOURCES["naive_h1"][0], index=False)
    rvi = f.dropna(subset=["close_RVI"]).groupby("date")["close_RVI"].first().reset_index()
    rvi.to_parquet(RVI, index=False)
    if SCOPE != "dev":
        (OUT / f"sources_README_{SCOPE}.json").write_text(json.dumps(dict(
            naive=f"sqrt(rv_w) из features_pit_v3 (строка дня t, RV дней t−5…t−1), "
                  f"2017-01-03…{CUT.date()}", rvi="close_RVI (срез 18:50) из features_pit_v3",
            rows=dict(naive=int(nv.dropna().shape[0]), rvi=len(rvi))), ensure_ascii=False, indent=1))
        print("σ-источники (тест):", SOURCES["naive_h1"][0].name, RVI.name)
        return
    steps = pd.read_csv(REGEN / "sigma_pred_h1_final2_steps.csv")
    F = dict(zip(steps["test_year"], steps["F"]))
    src = {42: REGEN / "D12_h1_A_d812_xgb_h1_val.parquet",
           43: REGEN / "D12_h1_A_d812_xgb_h1_seeds4344_val.parquet",
           44: REGEN / "D12_h1_A_d812_xgb_h1_seeds4344_val.parquet"}
    info = {}
    for s, p in src.items():
        v = pd.read_parquet(p, columns=["date", "ticker", "year", "seed", "rank", "rv_pred"])
        v = v[(v["seed"] == s) & (v["rank"] == 1) & (v["year"].between(2017, 2021))]
        v["date"] = pd.to_datetime(v["date"])
        v = v[v["date"] <= CUT]
        out = v[["date", "ticker"]].copy()
        out["sigma_pred_h1"] = np.sqrt(v["rv_pred"] * v["year"].map(F)).astype("float32")
        out.to_parquet(SOURCES[f"seed{s}_h1"][0], index=False)
        info[s] = len(out)
    (OUT / "sources_README.json").write_text(json.dumps(dict(
        naive="sqrt(rv_w) из features_pit_v3 (строка дня t, RV дней t−5…t−1), 2017-01-03…2021-12-31",
        seeds="sqrt(rv_pred·F_года) из D12_h1_A_d812_xgb_h1{,_seeds4344}_val (rank 1, шаги 2–6), "
              "F — sigma_pred_h1_final2_steps.csv; читались только date,ticker,year,seed,rank,rv_pred",
        rvi="close_RVI (срез 18:50) из features_pit_v3", rows=info), ensure_ascii=False, indent=1))
    print("σ-источники:", {k: str(v[0].name) for k, v in SOURCES.items()}, info)


def load_source(label: str) -> pd.DataFrame:
    need_sigma(f"load_source({label})")
    p, col, _ = SOURCES[label]
    d = pd.read_parquet(p, columns=["date", "ticker", col])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= CUT].rename(columns={col: "sigma"})
    return d.sort_values(["ticker", "date"])


def load_c_dop(path=None) -> dict:
    """Раздел dop сетки C (Ъ 22:30) → глобальный C_DOP."""
    import yaml
    global C_DOP
    C_DOP = yaml.safe_load(Path(path or C_DOP_YAML).read_text())["dop"]
    return C_DOP


def _c_gate(R: np.ndarray, par) -> np.ndarray:
    """G[3, тикер, бар] по рангу R: par = p — фильтр (как C1), par = (p1, p2) — частичная
    1 → 0,5 → 0 (как C2). Ранг NaN → 1."""
    T, N = R.shape
    nanR = np.isnan(R)
    G = np.ones((3, T, N))
    with np.errstate(invalid="ignore"):
        if not isinstance(par, tuple):
            p = par
            G[0] = np.where(nanR | (R > 1 - p), 1.0, 0.0)
            G[1] = np.where(nanR | (R < p), 1.0, 0.0)
            G[2] = np.where(nanR | ((R >= 1 - p) & (R <= p)), 1.0, 0.0)
            return G
        p1, p2 = par
        G[0] = np.where(R > 1 - p1, 1.0, np.where(R > 1 - p2, 0.5, 0.0))
        G[1] = np.where(R < p1, 1.0, np.where(R < p2, 0.5, 0.0))
        inner = (R >= 1 - p1) & (R <= p1)
        outer = (R >= 1 - p2) & (R <= p2)
        G[2] = np.where(inner, 1.0, np.where(outer, 0.5, 0.0))
    for k in range(3):
        G[k][nanR] = 1.0
    return G


def c_dop_gates(S: np.ndarray, RV, n_t, spec: dict) -> list:
    """Новые точки C: [(семья, [(метка, G[3, тикер, бар])])]. S — σ_pred [тикер, бар],
    RV — √RV_w (None — без C*). Ранг — по окну [d−W, d−1] без текущего дня."""
    from fastbt.s9_stats import past_rank
    out = []
    for fam, f in spec["semi"].items():
        if f["signal"] == "sigma_over_rvw":
            if RV is None:
                continue
            X = S / RV
        else:
            X = S
        pts = []
        for W in f["okna"]:
            R = np.full(X.shape, np.nan)
            for ti in range(X.shape[0]):
                n = n_t[ti]
                R[ti, :n] = past_rank(np.ascontiguousarray(X[ti, :n]), int(W),
                                      int(spec["min_periods"][W]))
            for v in f.get("filtr", []):
                pts.append((f"W{W}_p{v}", _c_gate(R, v / 100)))
            for v1, v2 in f.get("chastichnaya", []):
                pts.append((f"W{W}_p{v1}_{v2}", _c_gate(R, (v1 / 100, v2 / 100))))
        out.append((fam, pts))
    return out


def align(dates: np.ndarray, d: pd.DataFrame) -> np.ndarray:
    """Как fastbt.data.align_sigma: последнее значение с датой ≤ даты бара; после CUT — NaN."""
    arr = np.full(len(dates), np.nan)
    vd = pd.to_datetime(d["date"].values)
    idx = np.searchsorted(vd, pd.to_datetime(dates), side="right") - 1
    ok = (idx >= 0) & (pd.to_datetime(dates) <= CUT)
    arr[ok] = d["sigma"].values.astype(float)[idx[ok]]
    return arr


# ======================= кэш A и C основного дизайна =======================

def period_cfg(cfg):
    """Годы A и начало валидации — из периода. train — cfg как есть (2017–2021, побайтно прежнее)."""
    if PERIOD == "train":
        return cfg
    return dataclasses.replace(cfg, years_a=tuple(range(YEARS[0], YEARS[-1] + 1)), val_start=VAL_START)


def _job(args):
    ticker, srcs = args[:2]
    if len(args) > 2:                      # тест: в процессе spawn глобальные настройки заново
        set_scope(*args[2])
    if len(args) > 3 and args[3] != "train":   # spawn: период заново
        set_period(args[3])
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from fastbt import dev, masks, turnover
    from fastbt import design as dsg
    from fastbt.approach_a import ApproachA
    from fastbt.approach_cd import run_c
    from fastbt.data import load_ticker
    from fastbt.unit import make_unit
    d, _ = dev.load_design(MAIN_YAML)
    cfg = dev.make_config(d, SCOPE)
    cfg5 = dev.make_config({**d, "horizons": ["h5"]}, SCOPE)
    if SCOPE != "dev":
        cfg = dataclasses.replace(cfg, val_start=VAL_START, val_end=VAL_END)
        cfg5 = dataclasses.replace(cfg5, val_start=VAL_START, val_end=VAL_END)
    cfg, cfg5 = period_cfg(cfg), period_cfg(cfg5)
    dsg.apply(cfg.design)
    td = load_ticker(ticker, cfg.corp_actions)
    if PERIOD == "v2":                     # прогрев с 26.08.2014
        td = period_v2.trim_td(td)
    sig = {lab: (h, align(td.dates, df)) for lab, (h, df) in srcs.items()}
    out = {"dates": td.dates, "lr": td.log_ret, "A": {}, "C": {}, "Cmeta": {}}
    for sid in SIDS:
        dsg.apply(cfg.design)
        u = make_unit(ticker, sid, td)
        a = ApproachA(u, cfg).run()
        a_tf = turnover.apply(a, u, cfg, "A")
        out["A"][sid] = {Y: v[1].astype(np.float64) for Y, v in a_tf.items()}
        if PERIOD == "v2":                 # Ж-114 п. 1: невалидный тикер-день — нет сигнала, позиция держится
            out["A"][sid] = period_v2.hold_years(out["A"][sid], ticker, td.dates)
        if SCOPE != "dev":                 # п. 15 (10:05): A без фильтра оборота — рядом с основным
            out.setdefault("A0", {})[sid] = {Y: np.asarray(v[1], np.float64) for Y, v in a.items()}
        for lab, (h, arr) in sig.items():
            c_cfg = cfg if h == "h1" else cfg5
            dsg.apply(c_cfg.design)
            sg = {k: np.full(td.n, np.nan) for k in ("h1", "h5", "h22")}
            sg[h] = arr
            u.td = dataclasses.replace(td, sigma=sg)
            masks._CACHE.clear()               # ключ кэша масок не включает σ
            c = turnover.apply(run_c(u, c_cfg, a), u, c_cfg, "C")
            out["C"].setdefault(lab, {})[sid] = {Y: v[1].astype(np.float64)
                                                 for Y, v in c.items() if Y in YEARS}
            out["Cmeta"].setdefault(lab, {})[sid] = {Y: str(v[0]) for Y, v in c.items()
                                                     if Y in YEARS}
        u.td = td
        masks._CACHE.clear()
    return ticker, out


def cache_key() -> str:
    from fastbt.cache import code_hash
    h = hashlib.sha1(code_hash().encode())
    h.update(MAIN_YAML.read_bytes())
    for lab, (p, _, _) in SOURCES.items():
        h.update(lab.encode() + hashlib.sha1(Path(p).read_bytes()).digest())
    from fastbt import data as fd          # аудит FA2b-Н1: A и C строятся по барам и корп. событиям
    for p in (zona.path("daily", fd.OHLCV), fd.CORP):      # зона Z2: ряд из FASTBT_Z2_DAILY
        h.update(hashlib.sha1(Path(p).read_bytes()).digest())
    from fastbt import dividendy           # Ж-057 п. 1 (Д5): lr и ряд сигналов зависят от таблицы дивидендов и множителя НДФЛ
    h.update(dividendy.digest().encode())
    if SCOPE != "dev":
        h.update(f"{SCOPE}|{VAL_START}|{VAL_END}|{CUT.date()}".encode())
    if PERIOD != "train":                  # Д5: период в ключе только при не-умолчании (ключ train не меняется)
        h.update(f"period|{PERIOD}|{YEARS}|{VAL_START}".encode())
    if PERIOD == "v2":                     # прогрев и правило качества — в ключе (train4 не меняется)
        h.update(f"start|{period_v2.START.date()}|hold_invalid".encode())
        h.update(Path(period_v2.__file__).read_bytes())
    return h.hexdigest()[:10]


def build_cache(workers: int = 2) -> Path:
    from fastbt.data import TICKERS
    if PERIOD == "train":
        os.environ["FASTBT_VPRED"] = str(SOURCES["hyb_h1"][0])
        srcs = {lab: (SOURCES[lab][2], load_source(lab)) for lab in SOURCES}
    else:                                  # train4: только A и «A только Sk», σ-источники не читаются
        srcs = {}
    extra = (((SCOPE, VAL_START) if SCOPE != "dev" else ("dev",)), PERIOD)
    jobs = [(t, {lab: (h, df[df["ticker"] == t]) for lab, (h, df) in srcs.items()}) + extra
            for t in TICKERS]
    t0 = time.time()
    res = {}
    if workers <= 1:
        it = map(_job, jobs)
    else:
        import multiprocessing as mp
        os.environ["NUMBA_NUM_THREADS"] = "1"
        pool = mp.get_context("spawn").Pool(workers)
        it = pool.imap_unordered(_job, jobs)
    for tk, out in it:
        res[tk] = out
        print(f"{tk}: {time.time() - t0:.0f} с", flush=True)
    if workers > 1:
        pool.close()
        pool.join()
    p = OUT / f"cache_{cache_key()}.pkl"
    p.write_bytes(pickle.dumps(res, protocol=4))
    (OUT / _latest_name()).write_text(p.name)
    print(f"кэш → {p} ({time.time() - t0:.0f} с)")
    return p


# ======================= движок надстроек =======================

class Engine:
    def __init__(self, cache: dict):
        from fastbt import msp
        from fastbt.data import TICKERS, year_bounds
        self.tickers = [t for t in TICKERS if t in cache]
        self.cache = cache
        self.nmax = max(len(cache[t]["dates"]) for t in self.tickers)
        self.sig = {}
        rows = dict(tick=[], bar=[], pair=[], year=[], cat=[])
        self.vwin = {}          # (ti, Y) → (vs, ve) окна валидации/калибровки
        self.pairs = []
        alld = []
        for ti, t in enumerate(self.tickers):
            c = cache[t]
            dates = c["dates"]
            for Y in YEARS:
                ye = Y if not VAL_END else min(Y, VAL_END + 1)
                self.vwin[(ti, Y)] = (year_bounds(dates, VAL_START)[0], year_bounds(dates, ye)[0])
                if self.vwin[(ti, Y)][1] <= self.vwin[(ti, Y)][0]:
                    raise ValueError(f"пустое окно валидации {VAL_START}…{ye - 1} для {t} {Y}")
            for sid in SIDS:
                pid = len(self.pairs)
                self.pairs.append((t, sid))
                for Y in YEARS:
                    if Y not in c["A"][sid]:
                        continue
                    ys, ye = year_bounds(dates, Y)[0], year_bounds(dates, Y + 1)[0]
                    b = np.arange(ys, ye)
                    rows["tick"].append(np.full(len(b), ti))
                    rows["bar"].append(b)
                    rows["pair"].append(np.full(len(b), pid))
                    rows["year"].append(np.full(len(b), Y))
                    rows["cat"].append(np.full(len(b), CATS[sid]))
                    alld.append(pd.to_datetime(dates[ys:ye]))
        for k in rows:
            rows[k] = np.concatenate(rows[k])
        self.r = rows
        dts = pd.DatetimeIndex(np.concatenate([d.values for d in alld]))
        self.dates, self.r["date"] = np.unique(dts.values, return_inverse=True)
        self.dates = pd.DatetimeIndex(self.dates)
        self.first = np.r_[True, self.r["pair"][1:] != self.r["pair"][:-1]]
        self.lr = np.zeros((len(self.tickers), self.nmax))
        for ti, t in enumerate(self.tickers):
            self.lr[ti, :len(cache[t]["lr"])] = cache[t]["lr"]
        self.lr_row = self.lr[self.r["tick"], self.r["bar"]]
        self.cnt = np.bincount(self.r["date"], minlength=len(self.dates)).astype(float)
        rate = msp.rate_cbr()
        # Ъ ~23:20: режим «back» (дни (d_prev, d], rate_restored) — только для печатей v1;
        # в разработке и тесте v2 — ставка ЦБ и дни [d, d_next)
        assert rate.attrs.get("cal") != "back", "С9: ставка с днями (d_prev, d] запрещена"
        kr = rate.reindex(rate.index.union(self.dates)).sort_index().ffill().bfill()
        self.kr = kr.reindex(self.dates).values
        self.rf = self.kr / 365.0
        self.cal = msp.cal_days(self.dates)      # [d, d_next): период позиции d→d+1 (аудит FA, Н2)
        self.year_of = self.dates.year.values
        self.A_row = self.rows_from({(t, s): self.cache[t]["A"][s] for t, s in self.pairs})
        self.n_t = np.array([len(cache[t]["dates"]) for t in self.tickers])

    # ---- σ и позиции ----
    def sigma(self, label: str) -> np.ndarray:
        """[тикер, бар] σ-источника, выровненная по барам тикера (NaN вне ряда и после CUT)."""
        if label not in self.sig:
            df = load_source(label)
            S = np.full((len(self.tickers), self.nmax), np.nan)
            for ti, t in enumerate(self.tickers):
                S[ti, :self.n_t[ti]] = align(self.cache[t]["dates"], df[df["ticker"] == t])
            self.sig[label] = S
        return self.sig[label]

    def rvw(self) -> np.ndarray:
        """√RV_w [тикер, бар] для C* (файл из C_DOP['rvw']; дата ≤ бара, после CUT — NaN)."""
        if "__rvw__" not in self.sig:
            spec = C_DOP["rvw"]
            p = Path(spec["file"])
            p = p if p.is_absolute() else MAIN / p
            d = pd.read_parquet(p, columns=["date", "ticker", spec["column"]])
            d["date"] = pd.to_datetime(d["date"])
            d = d[d["date"] <= CUT].rename(columns={spec["column"]: "sigma"})
            d = d.sort_values(["ticker", "date"])
            R = np.full((len(self.tickers), self.nmax), np.nan)
            for ti, t in enumerate(self.tickers):
                R[ti, :self.n_t[ti]] = align(self.cache[t]["dates"], d[d["ticker"] == t])
            self.sig["__rvw__"] = R
        return self.sig["__rvw__"]

    def rvi(self, lag: int = 0) -> np.ndarray:
        d = pd.read_parquet(RVI)
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] <= CUT].rename(columns={"close_RVI": "sigma"})
        R = np.full((len(self.tickers), self.nmax), np.nan)
        for ti, t in enumerate(self.tickers):
            a = align(self.cache[t]["dates"], d)
            if lag:
                a = np.r_[np.full(lag, np.nan), a[:-lag]]
            R[ti, :self.n_t[ti]] = a
        return R

    def rows_from(self, by_pair: dict) -> np.ndarray:
        """{(тикер, sid): {Y: полный ряд}} → вектор строк (годы 2018–2021)."""
        out = np.zeros(len(self.r["bar"]))
        pos = 0
        for t, sid in self.pairs:
            d = by_pair[(t, sid)]
            for Y in YEARS:
                if Y not in self.cache[t]["A"][sid]:
                    continue
                ys = self._slice(t, Y)
                seg = d[Y][ys[0]:ys[1]] if Y in d else np.zeros(ys[1] - ys[0])
                out[pos:pos + len(seg)] = seg
                pos += len(seg)
        assert pos == len(out)
        return out

    def _slice(self, t, Y):
        from fastbt.data import year_bounds
        dates = self.cache[t]["dates"]
        return year_bounds(dates, Y)[0], year_bounds(dates, Y + 1)[0]

    def g_row(self, G: np.ndarray) -> np.ndarray:
        """Затвор/вес [тикер, бар] или [категория, тикер, бар] → вектор строк."""
        if G.ndim == 2:
            return G[self.r["tick"], self.r["bar"]]
        return G[self.r["cat"], self.r["tick"], self.r["bar"]]

    # ---- МСП ----
    def port(self, pos, comm=None, cap=1.0, wgt=None):
        comm = COMM if comm is None else comm      # v2: COMM задаётся --comm (Ъ 22:45: 0,04%)
        dp = np.diff(pos, prepend=0.0)
        dp[self.first] = pos[self.first]
        # Ъ 26.10, 23:50 (аудит FA2a, Н5): позиция, открытая на последнем баре окна пары, закрыта
        # по close первого бара после окна (доходность последней строки) — платит комиссию выхода
        last = np.r_[self.first[1:], True]
        di = self.r["date"]
        nd = len(self.dates)
        if wgt is None:
            sides = np.abs(dp) + np.where(last, np.abs(pos), 0.0)
            net = pos * self.lr_row - sides * comm
            r = np.bincount(di, net, nd) / self.cnt
            gross = np.bincount(di, pos * self.lr_row, nd) / self.cnt
            exp_ = np.bincount(di, np.abs(pos), nd) / self.cnt
            turn = np.bincount(di, sides, nd) / self.cnt
        else:
            eff = pos * wgt
            de = np.diff(eff, prepend=0.0)
            de[self.first] = eff[self.first]
            sides = np.abs(de) + np.where(last, np.abs(eff), 0.0)
            r = np.bincount(di, eff * self.lr_row - sides * comm, nd)
            gross = np.bincount(di, eff * self.lr_row, nd)
            exp_ = np.bincount(di, np.abs(eff), nd)
            turn = np.bincount(di, sides, nd)
        return self._finish(r, gross, exp_, turn, cap)

    def port_rows(self, g, sides, expo, comm=None, cap=1.0, noch=None):
        """Ж-047 (приёмка #345): портфель по готовым строкам — g = доходность×вес, sides = стороны×вес (с Н5 на
        границах окна), expo = вес×доля баров в рынке. Арифметика `port` (wgt=None): pos·lr → g, |dp|+last·|pos|
        → sides, |pos| → expo; деление на cnt и `_finish` те же. `port` не менялся.
        noch (Ж-065 п. 3) — занятость на 18:50 по строкам (fdr.noch_iz_*), только для ставки на свободный капитал;
        None — старый путь побайтно, `exposure` в выходе от noch не зависит."""
        comm = COMM if comm is None else comm
        di, nd = self.r["date"], len(self.dates)
        net = g - sides * comm
        r = np.bincount(di, net, nd) / self.cnt
        gross = np.bincount(di, g, nd) / self.cnt
        exp_ = np.bincount(di, expo, nd) / self.cnt
        turn = np.bincount(di, sides, nd) / self.cnt
        noch_ = None if noch is None else np.bincount(di, noch, nd) / self.cnt
        return self._finish(r, gross, exp_, turn, cap, noch_)

    def _finish(self, r, gross, exp_, turn, cap, noch_=None):
        """noch_ — уже сведённая по датам занятость на 18:50 (None — ставка по exp_, как раньше). Плечо mg — по exp_."""
        used = cap * exp_
        used_fdr = used if noch_ is None else cap * noch_
        fdr = np.maximum(0.0, 1.0 - used_fdr) * self.rf * self.cal
        # плата за плечо — за те же календарные дни [d, d_next), что и ставка (аудит FA2a, Н4)
        mg = np.maximum(0.0, used - 1.0) * (self.kr + 0.02) / 365.0 * self.cal
        total = r + fdr - mg
        p = pd.DataFrame(dict(r=r, gross=gross, exposure=exp_, turnover=turn, total=total,
                              excess=total - self.rf * self.cal,
                              excess_gross=gross + fdr - mg - self.rf * self.cal),
                         index=self.dates)
        return p

    def mix(self, ports, cap=1.0):
        """Равные доли капитала: средние r, gross, экспозиция, оборот по портфелям."""
        k = len(ports)
        f = lambda c: sum(p[c].values for p in ports) / k
        return self._finish(f("r"), f("gross"), f("exposure"), f("turnover"), cap)

    # ---- выбор по валидации ----
    def select(self, fam: np.ndarray, kind="mask", screen=None, cat=False):
        """fam: [K, тикер, бар] (или [K, кат, тикер, бар]) — затворы (1/0) или веса.
        Выбор на пару-год: первый лучший Шарп A·g на валидации 2017…Y−1 (комиссия 0,05%);
        screen=(lo, hi) — только точки с долей закрытых баров окна в [lo, hi] (нет — ближайшая).
        Возвращает (вектор строк затвора, {k: число пар-лет})."""
        from fastbt.approach_cd import _eval, _first_best, scaled_eval
        K = fam.shape[0]
        choice = {}
        cnt = {}
        for pid, (t, sid) in enumerate(self.pairs):
            ti = self.tickers.index(t)
            c = self.cache[t]
            n = self.n_t[ti]
            rows = fam[:, CATS[sid], ti, :n] if cat else fam[:, ti, :n]
            for Y in YEARS:
                if Y not in c["A"][sid]:
                    continue
                vs, ve = self.vwin[(ti, Y)]
                a = c["A"][sid][Y]
                if kind == "mask":
                    m = np.ascontiguousarray(rows > 0.5)
                    sh, _, _ = _eval(a, m, c["lr"], vs, ve, COMM)
                else:
                    sh = np.zeros(K)
                    scaled_eval(a, np.ascontiguousarray(rows), c["lr"], vs, ve, COMM, sh)
                ok = None
                if screen is not None:
                    cf = 1.0 - rows[:, vs:ve].mean(axis=1)
                    ok = (cf >= screen[0]) & (cf <= screen[1])
                    if not ok.any():
                        dist = np.maximum(screen[0] - cf, cf - screen[1])
                        ok = dist <= dist.min() + 1e-12
                k = _first_best(sh, ok)
                k = 0 if k < 0 else k
                choice[(pid, Y)] = k
                cnt[k] = cnt.get(k, 0) + 1
        g = np.ones(len(self.r["bar"]))
        pos = 0
        for pid, (t, sid) in enumerate(self.pairs):
            ti = self.tickers.index(t)
            for Y in YEARS:
                if (pid, Y) not in choice:
                    continue
                ys, ye = self._slice(t, Y)
                k = choice[(pid, Y)]
                src = fam[k, CATS[sid], ti] if cat else fam[k, ti]
                g[pos:pos + ye - ys] = src[ys:ye]
                pos += ye - ys
        return g, cnt

    def calib_closed(self, G: np.ndarray) -> float:
        """Средняя по тикеро-годам (по парам-годам для масок по категориям) доля закрытых
        баров калибровочного окна 2017…Y−1."""
        vals = []
        for ti in range(len(self.tickers)):
            for Y in YEARS:
                vs, ve = self.vwin[(ti, Y)]
                if G.ndim == 2:
                    vals.append(1.0 - G[ti, vs:ve].mean())
                else:
                    for sid in SIDS:
                        vals.append(1.0 - G[CATS[sid], ti, vs:ve].mean())
        return float(np.mean(vals))

    def gate_stats(self, g: np.ndarray) -> dict:
        """Доля закрытых баров 2018–2021 (от всех пар-дней, closed_dev), отсев от позиций A
        (closed_A — доля позиций-дней A ≠ 0, закрытых затвором: 1 − средний вес по этим дням;
        аудит FA, Н3) и переключений в год (в среднем по парам)."""
        closed = 1.0 - g
        ch = np.r_[0.0, np.abs(np.diff(g))] > 1e-9
        ch[self.first] = False
        pid = self.r["pair"]
        npair = pid.max() + 1
        cl = np.bincount(pid, closed, npair) / np.maximum(np.bincount(pid, None, npair), 1)
        sw = np.bincount(pid, ch.astype(float), npair) / len(YEARS)
        inA = self.A_row != 0
        closed_A = float(closed[inA].mean()) if inA.any() else float("nan")
        return dict(closed_dev=float(cl.mean()), closed_A=closed_A, switches_y=float(sw.mean()),
                    mean_weight=float(g.mean()))


# ======================= метрики =======================

def sharpe(x) -> float:
    from fastbt.msp import sharpe as s
    return s(x)


def metrics(p: pd.DataFrame, col="excess") -> dict:
    ex = p[col].values
    tot = p["total"].values
    neg = tot[tot < 0]
    ds = np.std(neg, ddof=1) if len(neg) > 1 else 1e-12
    cum = np.cumsum(ex)
    mdd = float(np.max(np.maximum.accumulate(cum) - cum))
    ann = float(np.mean(ex) * 252)
    out = dict(sharpe=sharpe(ex), sortino=float(np.mean(ex) * np.sqrt(252) / ds),
               calmar=ann / mdd if mdd > 1e-12 else np.nan, ann_excess_pct=ann * 100,
               mdd_excess_pct=mdd * 100, exposure_pct=float(p["exposure"].mean() * 100),
               turnover_y=float(p["turnover"].mean() * 252),
               comm_pct_y=float((p["gross"] - p["r"]).mean() * 252 * 100))
    yr = p.index.year
    for y in YEARS:
        out[str(y)] = sharpe(ex[yr == y])
    out["worst"] = min(out[str(y)] for y in YEARS)
    return out


def c8(m: dict, a: dict) -> dict:
    win = sum(m[str(y)] > a[str(y)] for y in YEARS)
    cy, cw, cp = win >= len(YEARS) - 1, m["worst"] >= a["worst"], m["sharpe"] > a["sharpe"]
    return dict(win=f"{win}/{len(YEARS)}", c_years=cy, c_worst=cw, c_period=cp,
                passed=bool(cy and cw and cp), positive=m["sharpe"] > 0,
                n_met=int(cy) + int(cw) + int(cp),
                mean_delta=float(np.mean([m[str(y)] - a[str(y)] for y in YEARS])))


def rank_c8(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["passed", "positive", "n_met", "mean_delta"],
                          ascending=[False, False, False, False])


def rank_c8_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Ъ 27.10, 00:10: сначала прошедшие С8 v2, среди них — прежний порядок rank_c8."""
    return df.sort_values(["passed_v2", "passed", "positive", "n_met", "mean_delta"],
                          ascending=[False, False, False, False, False])


def set_c8v2(scope: str = "dev", sigma_set: str = "v2", c_dop=True) -> None:
    """Включает --c8v2 (Ъ 27.10, 00:10): только разработка, σ v2 и новые точки C (--c-dop)."""
    global C8V2
    if scope != "dev" or sigma_set != "v2" or not c_dop:
        raise SystemExit("--c8v2 — только --scope dev --sigma-set v2 --c-dop")
    C8V2 = True


def _rank_c(df: pd.DataFrame) -> pd.DataFrame:
    return rank_c8_v2(df) if C8V2 else rank_c8(df)


# ======================= эксперименты =======================

class S9:
    def __init__(self, eng: Engine, reps: int):
        self.e, self.reps = eng, reps
        self.A = eng.port(eng.A_row)
        self.Am = eng.port(eng.A_row, cap=MARGIN)
        self.mA = metrics(self.A)
        self.mAm = metrics(self.Am)
        self.C_rows = {lab: eng.rows_from({(t, s): eng.cache[t]["C"][lab][s]
                                           for t, s in eng.pairs}) for lab in SOURCES}
        self.C = {lab: eng.port(r) for lab, r in self.C_rows.items()}
        self.Cm = {lab: eng.port(r, cap=MARGIN) for lab, r in self.C_rows.items()}

    def boot(self, p1, p0):
        from fastbt.report import boot_delta
        d, pv, _, _ = boot_delta(p1["excess"].values, p0["excess"].values, self.reps)
        return d, pv

    def delta(self, r: dict, key: str, p1, p0) -> None:
        """r[d_key], r[p_key]. dev — парный IID (как прежде). test (владелец 26.10, 10:05, п. 12):
        основной p — Ледуа–Вольф, круговой блочный, блок 5; IID — справочно в p_key_iid."""
        d, pv = self.boot(p1, p0)
        r[f"d_{key}"] = d
        if SCOPE == "dev" and not V2:
            r[f"p_{key}"] = pv
            return
        from fastbt.s9_stats import lw_sharpe_diff
        r[f"p_{key}"] = lw_sharpe_diff(p1["excess"].values, p0["excess"].values, self.reps, 5)[2]
        r[f"p_{key}_iid"] = pv

    def row(self, pos, family, point, h, src, extra=None, g=None, c_ref=None):
        e = self.e
        p, pm = e.port(pos), e.port(pos, cap=MARGIN)
        m = metrics(p)
        r = dict(family=family, point=point, h=h, sigma=src)
        if extra:
            r.update(extra)
        if g is not None:
            r.update(e.gate_stats(g))
        r.update(m)
        r["sharpe_margin"] = metrics(pm)["sharpe"]
        r["sharpe_A"], r["sharpe_A_margin"] = self.mA["sharpe"], self.mAm["sharpe"]
        self.delta(r, "vs_A", p, self.A)
        if c_ref is not None:
            r["sharpe_C"] = metrics(self.C[c_ref])["sharpe"]
            self.delta(r, "vs_C", p, self.C[c_ref])
        r.update(c8(m, self.mA))
        return r, p

    # ---------- D ----------
    def d_points(self, src: str):
        """{семейство: [(метка, G[тикер, бар], вид)]} для σ-источника src."""
        from fastbt.masks import hyst_gate
        from fastbt.s9_stats import band_hold, ema, minp, past_median, past_rank
        e = self.e
        S = e.sigma(src)
        T, N = S.shape
        fams = {k: [] for k in ("D1", "D2", "D4", "D5", "D6", "D6lag1", "D7")}

        def per_t(fn):
            G = np.ones((T, N))
            for ti in range(T):
                n = e.n_t[ti]
                G[ti, :n] = fn(ti, n)
            return G
        rk = {}
        for W in _d_windows():
            rk[W] = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                rk[W][ti, :n] = past_rank(S[ti, :n].copy(), W, _minp(W))
        for W in _d_windows():
            for p in (0.70, 0.80, 0.90):
                R = rk[W]
                fams["D1"].append((f"W{W}_p{int(p*100)}", np.where(R > p, 0.0, 1.0), "mask"))
                for gp in (0.10, 0.20):
                    G = per_t(lambda ti, n: hyst_gate(np.ascontiguousarray(R[ti, :n]), p - gp, p)
                              .astype(float))
                    fams["D2"].append((f"W{W}_p{int(p*100)}_g{int(gp*100)}", G, "mask"))
        for span in (3, 5, 10):
            Se = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                Se[ti, :n] = ema(S[ti, :n], span)
            for W in _d_windows():
                R = np.full((T, N), np.nan)
                for ti in range(T):
                    n = e.n_t[ti]
                    R[ti, :n] = past_rank(Se[ti, :n].copy(), W, _minp(W))
                for p in (0.70, 0.80, 0.90):
                    fams["D4"].append((f"ema{span}_W{W}_p{int(p*100)}",
                                       np.where(R > p, 0.0, 1.0), "mask"))
        chg = np.full((T, N), np.nan)
        chg[:, 5:] = S[:, 5:] / S[:, :-5] - 1.0
        for x in (0.20, 0.30, 0.50):
            fams["D5"].append((f"x{int(x*100)}", np.where(chg > x, 0.0, 1.0), "mask"))
        for lag, key in ((0, "D6"), (1, "D6lag1")):
            RV = e.rvi(lag)
            ratio = S * np.sqrt(252) * 100 / RV
            R = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                R[ti, :n] = past_rank(np.ascontiguousarray(ratio[ti, :n]), 252, minp(252))
            for p in (0.70, 0.80, 0.90):
                fams[key].append((f"W252_p{int(p*100)}", np.where(R > p, 0.0, 1.0), "mask"))
        for W in _d_windows():
            med = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                med[ti, :n] = _past_median(S[ti, :n], W)
            for gm in (1.0, 2.0):
                w = np.minimum(1.0, (med / S) ** gm)
                fams["D7"].append((f"W{W}_g{int(gm)}", np.where(np.isnan(w), 1.0, w), "scale"))
        return fams

    def run_d(self, srcs=("hyb_h1", "hyb_h5"), screen=(0.10, 0.50), tag="D"):
        e = self.e
        rows, fam_rows, ports = [], [], {}
        for src in srcs:
            h = SOURCES[src][2]
            cref = "hyb_h5" if h == "h5" else "hyb_h1"
            cref = cref if src.startswith("hyb") else src
            fams = self.d_points(src)
            allmask = []
            for fam, pts in fams.items():
                keep = []
                for lab, G, kind in pts:
                    cc = e.calib_closed(G)
                    ok = True if kind == "scale" else (screen[0] <= cc <= screen[1])
                    extra = dict(calib_closed=cc, screened_in=ok)
                    if not ok:
                        rows.append(dict(family=fam, point=lab, h=h, sigma=src, **extra))
                        continue
                    g = e.g_row(G)
                    r, _ = self.row(e.A_row * g, fam, lab, h, src, extra, g, cref)
                    rows.append(r)
                    keep.append((lab, G, kind))
                    if kind == "mask" and fam != "D6lag1":
                        allmask.append((f"{fam}:{lab}", G, kind))
                if keep:
                    fam_rows.append((fam, keep))
            fam_rows.append(("D1–D6 все точки", allmask))
            for fam, keep in fam_rows:
                if not keep:
                    continue
                kind = keep[0][2]
                stack = np.stack([G for _, G, _ in keep])
                g, cnt = e.select(stack, kind, screen if kind == "mask" else None)
                ch = ", ".join(f"{keep[k][0]}×{v}" for k, v in
                               sorted(cnt.items(), key=lambda kv: -kv[1])[:4])
                r, p = self.row(e.A_row * g, f"{fam} (выбор по валидации)", "семья", h, src,
                                dict(choice=ch, n_points=len(keep)), g, cref)
                rows.append(r)
                ports[(fam, src)] = (p, e.A_row * g)
            fam_rows = []
        df = pd.DataFrame(rows)
        return df, ports

    # ---------- B ----------
    def b_weights(self, src, cap, band):
        from fastbt.s9_stats import band_hold, past_median
        e = self.e
        S = e.sigma(src)
        W = np.ones_like(S)
        for ti in range(S.shape[0]):
            n = e.n_t[ti]
            med = past_median(S[ti, :n], 252)
            w = np.minimum(cap, med / S[ti, :n])
            W[ti, :n] = band_hold(np.ascontiguousarray(w), band)
        return W

    def run_b(self, srcs=("hyb_h1", "hyb_h5")):
        e = self.e
        rows, ports = [], {}
        base = dict(family="A (эталон)", point="—", h="—", sigma="—")
        for nm, pos in (("A", e.A_row),):
            for cap, lab in ((1.0, "спот"), (MARGIN, "маржа 20%")):
                p, pg = e.port(pos, cap=cap), e.port(pos, comm=0.0, cap=cap)
                rows.append(dict(base, scenario=lab, **self._bcols(p, pg)))
        for src in srcs:
            h = SOURCES[src][2]
            for cap in (1.0, 1.5, 2.0):
                for band in (0.0, 0.10, 0.20):
                    G = self.b_weights(src, cap, band)
                    g = e.g_row(G)
                    pos = e.A_row * g
                    lab = f"cap{cap:g}_band{int(band*100)}"
                    for sc, cs in (("спот", 1.0), ("маржа 20%", MARGIN)):
                        p, pg = e.port(pos, cap=cs), e.port(pos, comm=0.0, cap=cs)
                        ref = self.A if cs == 1.0 else self.Am
                        r = dict(family="B", point=lab, h=h, sigma=src, scenario=sc,
                                 **self._bcols(p, pg))
                        r["mean_weight"], r["max_weight"] = float(g.mean()), float(g.max())
                        self.delta(r, "vs_A", p, ref)
                        r.update(c8(metrics(p), metrics(ref)))
                        rows.append(r)
                        if sc == "спот":
                            ports[(lab, src)] = (p, pos)
        return pd.DataFrame(rows), ports

    @staticmethod
    def _bcols(p, pg):
        m, mg = metrics(p), metrics(pg)
        return dict(turnover_y=m["turnover_y"], comm_pct_y=m["comm_pct_y"],
                    exposure_pct=m["exposure_pct"], sharpe_gross=mg["sharpe"],
                    sharpe_net=m["sharpe"], ann_excess_net_pct=m["ann_excess_pct"],
                    **{str(y): m[str(y)] for y in YEARS}, worst=m["worst"])

    # ---------- C ----------
    def c_points(self, src):
        from fastbt.s9_stats import minp, past_rank
        e = self.e
        S = e.sigma(src)
        T, N = S.shape
        rk = {}
        for W in (126, 252, 504):
            rk[W] = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                rk[W][ti, :n] = past_rank(S[ti, :n].copy(), W, _minp(W))
        pts1 = []
        for W in (126, 252, 504):
            R = rk[W]
            nanR = np.isnan(R)
            for p in (0.70, 0.80, 0.90):
                G = np.ones((3, T, N))
                G[0] = np.where(nanR | (R > 1 - p), 1.0, 0.0)
                G[1] = np.where(nanR | (R < p), 1.0, 0.0)
                G[2] = np.where(nanR | ((R >= 1 - p) & (R <= p)), 1.0, 0.0)
                pts1.append((f"W{W}_p{int(p*100)}", G))
        pts2 = []
        R = rk[252]
        nanR = np.isnan(R)
        for p1, p2 in ((0.70, 0.90), (0.80, 0.90), (0.70, 0.80)):
            G = np.ones((3, T, N))
            G[0] = np.where(R > 1 - p1, 1.0, np.where(R > 1 - p2, 0.5, 0.0))
            G[1] = np.where(R < p1, 1.0, np.where(R < p2, 0.5, 0.0))
            inner = (R >= 1 - p1) & (R <= p1)
            outer = (R >= 1 - p2) & (R <= p2)
            G[2] = np.where(inner, 1.0, np.where(outer, 0.5, 0.0))
            for k in range(3):
                G[k][nanR] = 1.0
            pts2.append((f"W252_p{int(p1*100)}_{int(p2*100)}", G))
        return pts1, pts2

    def run_c(self, srcs=("hyb_h1", "hyb_h5")):
        e = self.e
        rows, ports = [], {}
        for src in srcs:
            h = SOURCES[src][2]
            cref = src
            m = metrics(self.C[src])
            r = dict(family="C основного дизайна (u02)", point="как в коде", h=h, sigma=src,
                     **m, sharpe_margin=metrics(self.Cm[src])["sharpe"])
            self.delta(r, "vs_A", self.C[src], self.A)
            r.update(c8(m, self.mA))
            if C8V2:
                self.kontrol(r, self.C_rows[src])
            rows.append(r)
            ports[("Cref", src)] = (self.C[src], self.C_rows[src])
            pts1, pts2 = self.c_points(src)
            fams = [("C1", pts1), ("C2", pts2)]
            if C_DOP is not None:          # Ъ 22:30: новые точки; C* — не на самой наивной σ
                rv = None if src.startswith("naive") else e.rvw()
                fams += c_dop_gates(e.sigma(src), rv, e.n_t, C_DOP)
            if C8V2:                       # Ъ 27.10, 00:10: p50/p60 — точки без семьи (rassh)
                fams += [(f, pts, "rassh") for f, pts in
                         c_dop_gates(e.sigma(src), None, e.n_t,
                                     dict(min_periods=C_DOP["min_periods"], semi=C_DOP["rassh"]))]
            for item in fams:
                fam, pts = item[0], item[1]
                for lab, G in pts:
                    g = e.g_row(G)
                    r, p = self.row(e.A_row * g, fam, lab, h, src,
                                    dict(calib_closed=e.calib_closed(G)), g, cref)
                    if C8V2:
                        self.kontrol(r, e.A_row * g, g)
                    rows.append(r)
                    ports[(f"{fam}:{lab}", src)] = (p, e.A_row * g)
                if len(item) > 2:          # расширение: семья «выбор по валидации» не строится
                    continue
                stack = np.stack([G for _, G in pts])
                kind = "mask" if fam in ("C1", "C*1") else "scale"
                g, cnt = e.select(stack, kind, None, cat=True)
                ch = ", ".join(f"{pts[k][0]}×{v}" for k, v in
                               sorted(cnt.items(), key=lambda kv: -kv[1])[:4])
                r, p = self.row(e.A_row * g, f"{fam} (выбор по валидации)", "семья", h, src,
                                dict(choice=ch), g, cref)
                if C8V2:
                    self.kontrol(r, e.A_row * g, g)
                rows.append(r)
                ports[(f"{fam} семья", src)] = (p, e.A_row * g)
        return pd.DataFrame(rows), ports

    def kontrol(self, r: dict, pos: np.ndarray, g: np.ndarray | None = None) -> None:
        """Ъ 27.10, 01:30: контроли без прогноза (fastbt.fast_kontrol) и С8 v2 для строки C:
        к2 — круговой сдвиг маски точки g по тикеру (v3 §0.4), к3 — A без S5/S6; к1 снят."""
        from fastbt import fast_kontrol as fk
        if getattr(self, "_k", None) is None or self._k.comm != COMM:
            self._k = fk.Kontrol(self.e, COMM)
            self._p3 = self.e.port(self._k.k3_pos())
            self._m3 = metrics(self._p3)
        k = self._k
        r["sharpe_k3"] = self._m3["sharpe"]
        self.delta(r, "vs_k3", self.e.port(pos), self._p3)       # d_vs_k3, p_vs_k3 (Ледуа–Вольф, блок 5)
        if g is None:
            # C u02: позиция — не A × затвор (фильтр оборота после маски), маски нет — к2 не определён,
            # С8 v2 точка не проходит
            r["sharpe_k2_med"] = r["sharpe_k2_q95"] = r["p_vs_k2"] = np.nan
            r["n_shift_k2"] = 0
            r["k2_primechanie"] = "к2 не определён: позиция ≠ A × затвор (u02); С8 v2 не проходит"
            r["passed_v2"] = False
            return
        if not np.array_equal(pos, self.e.A_row * g):
            raise AssertionError("к2: позиция точки ≠ A × затвор g")
        k.check(g)
        sh = k.k2_sharpes(g)
        r["sharpe_k2_med"] = float(np.median(sh))
        r["sharpe_k2_q95"] = float(np.quantile(sh, 0.95))
        r["p_vs_k2"] = fk.p_perm(r["sharpe"], sh)
        r["n_shift_k2"] = len(sh)
        r["passed_v2"] = bool(r["passed"] and r["sharpe"] > r["sharpe_k3"]
                              and r["p_vs_k2"] < C8V2_ALPHA)

    # ---------- C3: точность прогноза → результат ----------
    def run_c3(self, c_ports, srcs=("hyb_h1", "harj_h1", "naive_h1")):
        """v2: srcs — тройка одного h (h1 или h5), в списке и точки C2."""
        from fastbt.s9_stats import lw_sharpe_diff
        rows = []
        if V2:                             # все точки и семьи C (в т. ч. --c-dop)
            kinds = sorted({k for k, s in c_ports if s == srcs[0]})
        else:
            kinds = ["Cref", "C1 семья", "C2 семья"] + sorted({k for k, s in c_ports
                                                            if k.startswith("C1:") and s == srcs[0]})
        for kind in kinds:
            ps = {s: c_ports[(kind, s)][0] for s in srcs if (kind, s) in c_ports}
            if (len(ps) < 3) if not V2 else not (srcs[0] in ps and srcs[1] in ps):
                continue                       # v2: C* на наивной не строится — без неё
            r = dict(design=kind)
            if V2:
                r["h"] = SOURCES[srcs[0]][2]
            for s in srcs:
                if V2 and s not in ps:
                    continue
                if V2:                         # одна таблица на h=1 и h=5: столбцы без h
                    m = metrics(ps[s])
                    tag = s.split("_")[0]
                    r[f"sharpe_{tag}"] = m["sharpe"]
                    for y in YEARS:
                        r[f"{y}_{tag}"] = m[str(y)]
                    r[f"d_vs_A_{tag}"] = m["sharpe"] - self.mA["sharpe"]
                else:
                    r[f"sharpe_{s}"] = metrics(ps[s])["sharpe"]
            hy, hj, nv = srcs
            for a, b in ((hy, hj), (hy, nv), (hj, nv)):
                if a not in ps or b not in ps:
                    continue
                d, se, p = lw_sharpe_diff(ps[a]["excess"].values, ps[b]["excess"].values,
                                          self.reps, 5)
                _, pb = self.boot(ps[a], ps[b])
                k = f"{a.split('_')[0]}−{b.split('_')[0]}"
                r[f"d_{k}"], r[f"se_lw_{k}"], r[f"p_lw_{k}"], r[f"p_iid_{k}"] = d, se, p, pb
            rows.append(r)
        return pd.DataFrame(rows)

    # ---------- P1 ----------
    def run_p1(self):
        e = self.e
        rows = []
        bases = {"A": e.A_row, "C (u02, гибрид h1)": self.C_rows["hyb_h1"]}
        inv = {"равные": np.ones(len(e.r["bar"])),
               "1/σ_pred (гибрид h1)": 1.0 / e.sigma("hyb_h1")[e.r["tick"], e.r["bar"]],
               "1/RV_w (наивная)": 1.0 / e.sigma("naive_h1")[e.r["tick"], e.r["bar"]]}
        refs = {}
        for bn, pos in bases.items():
            for wn, iv in inv.items():
                iv = np.where(np.isfinite(iv), iv, np.nan)
                # нет σ на дату — равный вес тикера (среднее 1/σ по строкам даты)
                di = e.r["date"]
                mean_iv = np.bincount(di, np.nan_to_num(iv), len(e.dates)) / np.maximum(
                    np.bincount(di, np.isfinite(iv).astype(float), len(e.dates)), 1)
                iv = np.where(np.isfinite(iv), iv, mean_iv[di])
                wgt = iv / np.bincount(di, iv, len(e.dates))[di]
                for sc, cs in (("спот", 1.0), ("маржа 20%", MARGIN)):
                    p = e.port(pos, cap=cs, wgt=wgt)
                    m = metrics(p)
                    r = dict(base=bn, weights=wn, scenario=sc, **m)
                    if wn == "равные":
                        refs[(bn, sc)] = p
                    else:
                        self.delta(r, "vs_equal", p, refs[(bn, sc)])
                    rows.append(r)
        return pd.DataFrame(rows)

    # ---------- P2 ----------
    def run_p2(self, sel: dict):
        """sel: {'A','B','C','D': (портфель, позиции)} — выбранные одиночные."""
        e = self.e
        rows = []
        singles = {k: v[0] for k, v in sel.items()}
        for k, p in singles.items():
            rows.append(dict(portfolio=f"одиночная {k}", **metrics(p)))
        eq = e.mix(list(singles.values()))
        avg = e.port(sum(v[1] for v in sel.values()) / len(sel))
        best_k = max(singles, key=lambda k: metrics(singles[k])["sharpe"])
        for nm, p in (("равные доли A/B/C/D", eq), ("среднее позиций A/B/C/D", avg)):
            r = dict(portfolio=nm, **metrics(p))
            r["best_single"] = best_k
            self.delta(r, "vs_best", p, singles[best_k])
            self.delta(r, "vs_A", p, self.A)
            rows.append(r)
        eqm = e.mix([e.port(v[1], cap=MARGIN) for v in sel.values()], cap=MARGIN)
        avgm = e.port(sum(v[1] for v in sel.values()) / len(sel), cap=MARGIN)
        rows.append(dict(portfolio="равные доли A/B/C/D, маржа 20%", **metrics(eqm)))
        rows.append(dict(portfolio="среднее позиций A/B/C/D, маржа 20%", **metrics(avgm)))
        return pd.DataFrame(rows), {"eq": (eq, None), "avg": (avg, sum(v[1] for v in sel.values())
                                                              / len(sel))}

    # ---------- P3 ----------
    def run_p3(self, pairs: dict):
        """pairs: {название: (портфель на гибриде, портфель на HAR-J)}."""
        from fastbt.final_test import ce_boot
        from fastbt.s9_stats import fko_fee
        rows = []
        for nm, (ph, pj) in pairs.items():
            for g in (3, 5, 10):
                d_ce, p_ce, lo, hi = ce_boot(ph["total"].values, pj["total"].values, g, self.reps)
                rows.append(dict(design=nm, gamma=g,
                                 fko_fee_pct_y=fko_fee(ph["total"].values, pj["total"].values, g),
                                 d_ce_pct_y=d_ce, p_ce=p_ce, ci_lo=lo, ci_hi=hi,
                                 sharpe_hyb=metrics(ph)["sharpe"], sharpe_harj=metrics(pj)["sharpe"]))
        return pd.DataFrame(rows)

    # ---------- T1, T2 ----------
    def run_t1(self, sel: dict):
        e = self.e
        rows = []
        comms, slips = (T1_COMMS_V2, (0.0,)) if V2 else ((0.0005, 0.0010, 0.0020), (0.0, 0.0005))
        for nm, pos in sel.items():
            for c in comms:
                for sl in slips:
                    for sc, cs in (("спот", 1.0), ("маржа 20%", MARGIN)):
                        m = metrics(e.port(pos, comm=c + sl, cap=cs))
                        if V2:                 # Ъ 22:55: подписи ставок, «всё включено»
                            rows.append(dict(podpis=T1_PODPIS[round(c * 1e5)]))
                        else:
                            rows.append({})
                        rows[-1].update(dict(design=nm, comm_pct=c * 100, slippage_pct=sl * 100,
                                         scenario=sc, sharpe=m["sharpe"],
                                         ann_excess_pct=m["ann_excess_pct"],
                                         comm_pct_y=m["comm_pct_y"], worst=m["worst"],
                                         **{str(y): m[str(y)] for y in YEARS}))
        return pd.DataFrame(rows)

    def run_t2(self, sel_ports: dict):
        rows = []
        cut = pd.Timestamp(T2_CUT)
        for nm, p in sel_ports.items():
            for per, mask in ((T2_LABELS[0], slice(None)),
                              (T2_LABELS[1], p.index < cut), (T2_LABELS[2], p.index >= cut)):
                q = p.loc[mask] if not isinstance(mask, slice) else p
                rows.append(dict(design=nm, period=per, sharpe=sharpe(q["excess"]),
                                 ann_excess_pct=float(q["excess"].mean() * 252 * 100),
                                 exposure_pct=float(q["exposure"].mean() * 100), days=len(q)))
            for y in YEARS:
                q = p[p.index.year == y]
                rows.append(dict(design=nm, period=str(y), sharpe=sharpe(q["excess"]),
                                 ann_excess_pct=float(q["excess"].mean() * 252 * 100),
                                 exposure_pct=float(q["exposure"].mean() * 100), days=len(q)))
        return pd.DataFrame(rows)


def load_cache() -> dict:
    name = (OUT / _latest_name()).read_text().strip()
    return pickle.loads((OUT / name).read_bytes())


def check_against_pipeline(eng: Engine, s9: "S9") -> str:
    """Сверка движка с fastbt.msp.portfolio на позициях пробного прогона основного дизайна."""
    from fastbt import msp
    p = MAIN / "data" / "fastbt" / f"final_{SCOPE}_main.parquet"
    if not p.exists():
        return f"нет {p.name} — сверка пропущена"
    pos = pd.read_parquet(p)
    pos = pos[pos["test_year"].isin(YEARS)]
    out = []
    for ap, mine in (("A", s9.A), ("C", s9.C["hyb_h1"])):
        ref = msp.portfolio(pos[pos["approach"] == ap], msp.rate_cbr(), COMM)
        j = ref.index.intersection(mine.index)
        out.append(f"{ap}: Шарп пайплайн {msp.sharpe(ref['excess']):.4f} / движок "
                   f"{sharpe(mine['excess']):.4f}, max|Δexcess| "
                   f"{np.abs(ref.loc[j, 'excess'] - mine.loc[j, 'excess']).max():.2e}")
    return "; ".join(out)


def run_all_experiments(reps: int) -> None:
    t0 = time.time()
    eng = Engine(load_cache())
    s9 = S9(eng, reps)
    log = [f"сверка: {check_against_pipeline(eng, s9)}"]
    print(log[-1], flush=True)
    # D
    dfd, dports = s9.run_d(("hyb_h1", "hyb_h5"))
    to_csv(dfd, "s9_D_zatvor.csv")
    print(f"D: {len(dfd)} строк, {time.time() - t0:.0f} с", flush=True)
    # B
    dfb, bports = s9.run_b(("hyb_h1", "hyb_h5"))
    to_csv(dfb, "s9_B_targetirovanie.csv")
    print(f"B: {len(dfb)} строк, {time.time() - t0:.0f} с", flush=True)
    # C (+ HAR-J, наивная для C3)
    dfc, cports = s9.run_c(("hyb_h1", "hyb_h5", "harj_h1", "harj_h5", "naive_h1", "naive_h5") if V2
                           else ("hyb_h1", "hyb_h5", "harj_h1", "naive_h1"))
    if SCOPE == "dev":
        to_csv(dfc, "s9_C_rezhim.csv")
    else:                                  # тест: одна заданная точка C + сетка «чувствительность»
        kt = frozen_keys()
        main_c, sens_c = split_c_test(dfc, kt["C"])
        to_csv(main_c, "s9_C_osnovnaya.csv")
        to_csv(sens_c, "s9_C_chuvstvitelnost.csv")
    dfc3 = (pd.concat([s9.run_c3(cports, (f"hyb_{h}", f"harj_{h}", f"naive_{h}")) for h in ("h1", "h5")],
                      ignore_index=True) if V2 else s9.run_c3(cports))
    to_csv(dfc3, "s9_C3_tochnost_v_rezultat.csv")
    print(f"C: {len(dfc)} строк, C3 {len(dfc3)}, {time.time() - t0:.0f} с", flush=True)
    # выбор лучших по С8 (в тесте — не повторяется: ключи из таблиц разработки)
    k = select_keys(dfd, dfb, dfc) if SCOPE == "dev" else kt
    dkey, bkey, ckey, dk1, b1 = k["D"], k["B"], k["C"], k["D1"], k["B1"]
    sel = {"A": (s9.A, eng.A_row), "B": bports[bkey], "C": cports[ckey], "D": dports[dkey]}
    choice = dict(D=f"{dkey[0]} на {dkey[1]}", B=f"{bkey[0]} на {bkey[1]}",
                  C=f"{ckey[0]} на {ckey[1]}")
    log.append(f"выбор по С8{'' if SCOPE == 'dev' else ' (из разработки)'}: {choice}")
    if SCOPE != "dev":
        to_csv(main_rows(s9, eng, sel, choice, k["passed"]), "s9_osnovnye.csv")
    if V2:
        vb = vybor_v2(s9, dfd, dfb, dfc)
        to_csv(vb, "s9_vybor_C8.csv")
        alt = TABLES / "s9_vybor_C8_v2_k0050.csv"      # прогон --comm 0.0005 (тот же кэш), если был
        if SUFFIX == "_v2" and alt.exists():
            to_csv(sravnenie_komissiy(vb, pd.read_csv(alt, sep=";", decimal=",", encoding="utf-8-sig")),
                   "s9_vybor_C8_komissiya.csv")
        h2 = dfc3[(dfc3["design"] == k["C"][0]) & (dfc3["h"] == SOURCES[k["C"][1]][2])]
        to_csv(h2.assign(rol="H2 внутри выбранной точки C: без выбора"), "s9_H2_vybrannaya_C.csv")
        if SUFFIX == "_v2":                # Ъ 23:10, п. 1–2: только основная комиссия 0,04%
            to_csv(h2_luchshiy_v2(s9, dfc, cports, k["C"]), "s9_H2_luchshiy_s_luchshim.csv")
            to_csv(ustoychivost_c8v2(dfc) if C8V2 else ustoychivost_v2(dfc), "s9_C_ustoychivost.csv")
    print(log[-1], flush=True)
    dfp2, p2 = s9.run_p2(sel)
    to_csv(dfp2, "s9_P2_multistrategiya.csv")
    dfp1 = s9.run_p1()
    to_csv(dfp1, "s9_P1_vesa_tikerov.csv")
    # P3: гибрид против HAR-J (h=1) — C основного дизайна, C1/C2 семьи, лучшие D/B на h=1
    dports_j, _ = None, None
    dj, dpj = s9.run_d(("harj_h1",))
    to_csv(dj, "s9_D_zatvor_harj.csv")
    bj, bpj = s9.run_b(("harj_h1",))
    pairs = {"C основного дизайна (u02)": (s9.C["hyb_h1"], s9.C["harj_h1"]),
             "C1 семья": (cports[("C1 семья", "hyb_h1")][0], cports[("C1 семья", "harj_h1")][0]),
             "C2 семья": (cports[("C2 семья", "hyb_h1")][0], cports[("C2 семья", "harj_h1")][0]),
             f"D лучший на h=1: {dk1}": (dports[(dk1, "hyb_h1")][0], dpj[(dk1, "harj_h1")][0]),
             f"B лучший на h=1: {b1}": (bports[(b1, "hyb_h1")][0], bpj[(b1, "harj_h1")][0])}
    to_csv(s9.run_p3(pairs), "s9_P3_fko.csv")
    # T1/T2
    t_sel = {"A": eng.A_row, f"B {choice['B']}": sel["B"][1], f"C {choice['C']}": sel["C"][1],
             f"D {choice['D']}": sel["D"][1], "P2 среднее позиций": p2["avg"][1]}
    to_csv(s9.run_t1(t_sel), "s9_T1_komissii.csv")
    tp = {"A": s9.A, f"B {choice['B']}": sel["B"][0], f"C {choice['C']}": sel["C"][0],
          f"D {choice['D']}": sel["D"][0], "C основного дизайна (u02)": s9.C["hyb_h1"],
          "P2 равные доли": p2["eq"][0], "P2 среднее позиций": p2["avg"][0]}
    to_csv(s9.run_t2(tp), "s9_T2_podperiody.csv")
    # seed σ: C основного дизайна, C1/C2 семьи, лучший D и B на h=1
    seed_rows = []
    # seed 42/43/44 — реализации финалиста второго круга (D12_h1); к σ_pred final3 (D22_h1)
    # они не относятся, поэтому на final3 таблица seed не считается
    for sd in (("hyb_h1", "seed42_h1", "seed43_h1", "seed44_h1") if not SUFFIX else ()):
        dd, dpp = (dfd, dports) if sd == "hyb_h1" else s9.run_d((sd,))
        cc_, cpp = (dfc, cports) if sd == "hyb_h1" else s9.run_c((sd,))
        bb, bpp = (dfb, bports) if sd == "hyb_h1" else s9.run_b((sd,))
        items = {"C основного дизайна (u02)": s9.C[sd], "C1 семья": cpp[("C1 семья", sd)][0],
                 "C2 семья": cpp[("C2 семья", sd)][0], f"D {dk1}": dpp[(dk1, sd)][0],
                 f"B {b1}": bpp[(b1, sd)][0]}
        for nm, p in items.items():
            m = metrics(p)
            r = dict(sigma=sd, design=nm, **m)
            s9.delta(r, "vs_A", p, s9.A)
            r.update(c8(m, s9.mA))
            seed_rows.append(r)
    if seed_rows:
        to_csv(pd.DataFrame(seed_rows), "s9_seed_sigma.csv")
    log.append(f"готово за {time.time() - t0:.0f} с")
    (OUT / f"run_log{SUFFIX}.json").write_text(json.dumps(dict(log=log, choice=choice, reps=reps,
                                                          sigma_h1=str(SOURCES["hyb_h1"][0])),
                                                 ensure_ascii=False, indent=1))
    print("\n".join(log))


def main_rows(s9: "S9", eng: Engine, sel: dict, choice: dict, passed: dict) -> pd.DataFrame:
    """Тест (владелец 26.10, 10:05, п. 13, 15): по одному дизайну на подход, выбранному по С8 на
    разработке; основной A — с фильтром оборота, A без фильтра — рядом (справочно); Δ к A —
    p Ледуа–Вольфа (блок 5) основной, IID справочно; маржа — отдельная таблица, здесь только спот."""
    rows = []
    a0 = eng.rows_from({(t, s): eng.cache[t]["A0"][s] for t, s in eng.pairs})
    items = [("A", "A основного дизайна (с фильтром оборота) — база С8 и H3", s9.A, "")]
    items.append(("A без фильтра", "A без фильтра оборота — справочно", eng.port(a0), ""))
    for ap in ("B", "C", "D"):
        note = "" if passed[ap] else "лучшая по С8 на разработке, С8 не прошла"
        items.append((ap, choice[ap], sel[ap][0], note))
    for ap, design, p, note in items:
        r = dict(approach=ap, design=design, note=note, **metrics(p))
        if ap != "A":
            s9.delta(r, "vs_A", p, s9.A)
        rows.append(r)
    return pd.DataFrame(rows)


def _key_of(row, sem=" семья"):
    return row["family"].replace(" (выбор по валидации)", sem)


def select_keys(dfd: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame) -> dict:
    """Лучшие по С8: D (семья), B (спот), C (h=1/h=5 гибрида); D1, B1 — лучшие на h=1 (для P3)."""
    fam_d = dfd[dfd["point"] == "семья"].copy()
    best_d = rank_c8(fam_d).iloc[0]
    best_d1 = rank_c8(fam_d[fam_d["h"] == "h1"]).iloc[0]
    bsp = dfb[(dfb["family"] == "B") & (dfb["scenario"] == "спот")]
    best_b = rank_c8(bsp).iloc[0]
    best_b1 = rank_c8(bsp[bsp["h"] == "h1"]).iloc[0]
    cc = dfc[dfc["sigma"].isin(["hyb_h1", "hyb_h5"])]
    best_c = _rank_c(cc).iloc[0]           # --c8v2: правило С8 v2 (Ъ 27.10, 00:10)
    ckey_name = ("Cref" if best_c["family"].startswith("C основного") else
                 (_key_of(best_c) if best_c["point"] == "семья"
                  else f"{best_c['family']}:{best_c['point']}"))
    return dict(D=(_key_of(best_d, ""), best_d["sigma"]), B=(best_b["point"], best_b["sigma"]),
                C=(ckey_name, best_c["sigma"]), D1=_key_of(best_d1, ""), B1=best_b1["point"],
                passed={k: bool(v["passed"]) for k, v in (("B", best_b), ("C", best_c), ("D", best_d))})


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def vybor_v2(s9: "S9", dfd: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame) -> pd.DataFrame:
    """v2: один дизайн на подход по С8 (rank_c8, кандидаты — как в select_keys); p к A —
    Ледуа–Вольф (блок 5); поправка на число кандидатов: Бонферрони и Холм по p_vs_A кандидатов."""
    cand = {"D": dfd[dfd["point"] == "семья"],
            "B": dfb[(dfb["family"] == "B") & (dfb["scenario"] == "спот")],
            "C": dfc[dfc["sigma"].isin(["hyb_h1", "hyb_h5"])]}
    grid = {"D": int(dfd["sharpe"].notna().sum()), "B": len(cand["B"]), "C": len(cand["C"])}
    rows = [dict(approach="A", design="main.yaml:A (эталон С8)", sigma="—", **s9.mA)]
    items = list(cand.items())
    if C8V2:                               # Ъ 27.10, 00:10: прежний выбор C — справочно, рядом
        items.append(("C (прежнее правило С8, без к2/к3 — справочно)", cand["C"]))
    for ap, c in items:
        c = c.copy()
        c["p_holm"] = _holm(c["p_vs_A"].values)
        c["p_bonf"] = np.minimum(1.0, c["p_vs_A"] * len(c))
        v2 = C8V2 and ap == "C"
        best = (rank_c8_v2(c) if v2 else rank_c8(c)).iloc[0]
        r = dict(approach=ap, design=f"{best['family']}:{best['point']}", sigma=best["sigma"],
                 h=best["h"], n_kandidatov=len(c), n_tochek_setki=grid[ap if ap in grid else "C"],
                 n_proshli_C8=int(c["passed"].sum()))
        if C8V2 and ap.startswith("C"):
            r["n_proshli_C8_v2"] = int(c["passed_v2"].astype(bool).sum())
        for col in ("sharpe", "sortino", "calmar", "mdd_excess_pct", "exposure_pct", "turnover_y",
                    "sharpe_net", "turnover_y", *[str(y) for y in YEARS], "worst", "win", "c_years",
                    "c_worst", "c_period", "passed", "positive", "n_met", "mean_delta", "d_vs_A",
                    "p_vs_A", "p_vs_A_iid", "p_holm", "p_bonf", *K_COLS):
            if col in best.index:
                r[col] = best[col]
        if ap == "B":
            r["sharpe"] = best["sharpe_net"]
        r["fakt"] = ("прошёл С8" if best["passed"] else "лучший по С8, С8 не прошёл") + \
                    ("; положительный период" if best["positive"] else "; положительного периода нет")
        if v2:
            r["fakt"] = ("прошёл С8 v2 (С8, Шарп > к3, p к к2 < 0,05); выбор среди прошедших"
                         if best["passed_v2"] else "С8 v2 не прошла ни одна точка; лучший по С8") + \
                        ("; положительный период" if best["positive"] else "; положительного периода нет")
        if ap == "D":
            r["proverka_RVI"] = D_RVI
        rows.append(r)
    return pd.DataFrame(rows)


# Ъ 27.10, 00:10: столбцы контролей без прогноза (S9.kontrol)
K_COLS = ("sharpe_k2_med", "sharpe_k2_q95", "p_vs_k2", "n_shift_k2", "sharpe_k3", "d_vs_k3", "p_vs_k3",
          "passed_v2")


def _ckey(row) -> str:
    """Ключ портфеля C (как в run_c) по строке таблицы C."""
    if row["family"].startswith("C основного"):
        return "Cref"
    return _key_of(row) if row["point"] == "семья" else f"{row['family']}:{row['point']}"


_C8_COLS = ("sharpe", "sortino", "calmar", "ann_excess_pct", "mdd_excess_pct", "exposure_pct",
            "turnover_y", "comm_pct_y", *[str(y) for y in YEARS], "worst", "win", "c_years",
            "c_worst", "c_period", "passed", "positive", "n_met", "mean_delta", "d_vs_A", "p_vs_A",
            "p_vs_A_iid")


def h2_luchshiy_v2(s9: "S9", dfc: pd.DataFrame, cports: dict, ckey: tuple) -> pd.DataFrame:
    """Ъ 23:10, п. 1: H2 «лучший с лучшим» — для каждой σ (гибрид, HAR-J (б), наивная) свой отбор
    по С8 (rank_c8) по той же сетке C, включая --c-dop (у наивной C* нет: σ/√RV_w ≡ 1), кандидаты —
    h=1 и h=5 этой σ; затем выбранные точки попарно: ΔШарп, p Ледуа–Вольфа (блок 5), IID справочно.
    Первая таблица H2 (внутри точки гибрида) — s9_H2_vybrannaya_C, не меняется.
    Ъ 27.10, 00:00, п. 1: кандидаты — только фиксированные точки без выбора внутри: исключены
    C основного дизайна (u02) и семьи «(выбор по валидации)» (H2_VNUTR_VYBOR)."""
    from fastbt.s9_stats import lw_sharpe_diff
    rows, sel = [], {}
    fixed = ~(dfc["family"].str.startswith(H2_VNUTR_VYBOR[0])
              | dfc["family"].str.endswith(H2_VNUTR_VYBOR[1]) | (dfc["point"] == "семья"))
    for nm, srcs in H2_GROUPS:
        c = dfc[dfc["sigma"].isin(srcs) & fixed].copy()
        c["p_holm"] = _holm(c["p_vs_A"].values)
        best = _rank_c(c).iloc[0]          # --c8v2: С8 v2 (Ъ 27.10, 00:10)
        key = (_ckey(best), best["sigma"])
        sel[nm] = (f"{best['family']}:{best['point']} на {best['sigma']}", cports[key][0])
        r = dict(rol="выбор по С8 отдельно для σ (только фиксированные точки)", sigma_nabor=nm,
                 design=f"{best['family']}:{best['point']}", sigma=best["sigma"], h=best["h"],
                 n_kandidatov=len(c), n_proshli_C8=int(c["passed"].sum()),
                 sovpal_s_vyborom_C8=(key == tuple(ckey)) if nm == "гибрид" else None,
                 isklyucheny="C основного дизайна (u02); C1/C2/C2W/C*1/C*2 (выбор по валидации): семья")
        r.update({col: best[col] for col in _C8_COLS if col in best.index})
        r["p_holm"] = best["p_holm"]
        if C8V2:
            r.update({col: best[col] for col in K_COLS if col in best.index})
            r["n_proshli_C8_v2"] = int(c["passed_v2"].astype(bool).sum())
        r["fakt"] = ("прошёл С8" if best["passed"] else "лучший по С8, С8 не прошёл") + \
                    ("; положительный период" if best["positive"] else "; положительного периода нет")
        if nm == "наивная":
            r["primechanie"] = ("naive_h1 и naive_h5 — один ряд √RV_w (C1/C2/C2W и семьи совпадают, "
                                "при равенстве — первая по таблице, h1); C* не строится")
        rows.append(r)
    names = [nm for nm, _ in H2_GROUPS]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = sel[a][1], sel[b][1]
            d, se, p = lw_sharpe_diff(pa["excess"].values, pb["excess"].values, s9.reps, 5)
            _, pi = s9.boot(pa, pb)
            rows.append(dict(rol="попарно: выбранная − выбранная", sigma_nabor=f"{a} − {b}",
                             design=f"{sel[a][0]} − {sel[b][0]}", sharpe=metrics(pa)["sharpe"],
                             sharpe_2=metrics(pb)["sharpe"], d_sharpe=d, se_lw=se, p_lw=p,
                             p_iid=pi))
    return pd.DataFrame(rows)


def _okno_porog(point: str) -> tuple:
    import re
    m = re.fullmatch(r"W(\d+)_p(\d+)(?:_(\d+))?", str(point))
    if not m:
        return "—", "—"
    return int(m.group(1)), (m.group(2) if m.group(3) is None else f"{m.group(2)}/{m.group(3)}")


def ustoychivost_v2(dfc: pd.DataFrame) -> pd.DataFrame:
    """Ъ 23:10, п. 2: устойчивость выбора C на гибриде (кандидаты отбора — hyb_h1, hyb_h5).
    (а) C1 W504 p70 h=5 (точка первого FINAL) на новых σ, 0,04% — все метрики, Δ к A, p, С8;
    (б) точки, прошедшие С8, с |Шарп − Шарп выбранной| ≤ 0,1 (выбранная — первой строкой);
    (в) вывод: соседи выбранной по окну (те же p, h, семья) и по порогу (те же W, h, семья).
    Правило (объявлено до счёта): плато — оба ближайших соседа (следующее окно и следующий
    порог) в пределах 0,1 и прошли С8; пик — ни одного из соседей; иначе — частичное плато."""
    c = dfc[dfc["sigma"].isin(["hyb_h1", "hyb_h5"])].copy()
    c["p_holm"] = _holm(c["p_vs_A"].values)
    best = rank_c8(c).iloc[0]
    sh0 = float(best["sharpe"])
    rows = []
    fam, pt, sg = C_FINAL1
    a = c[(c["family"] == fam) & (c["point"] == pt) & (c["sigma"] == sg)]
    if len(a) != 1:
        raise SystemExit(f"устойчивость: строк {C_FINAL1} — {len(a)}")
    r = dict(rol="(а) точка первого FINAL: C1 W504 p70 h=5 на гибриде (новые σ, 0,04%)")
    r.update({k: v for k, v in a.iloc[0].items() if k != "pometka"})
    r["p_holm"] = a.iloc[0]["p_holm"]
    r["d_k_vybrannoy"] = float(a.iloc[0]["sharpe"]) - sh0
    r["okno"], r["porog"] = _okno_porog(pt)
    rows.append(r)
    near = c[c["passed"].astype(bool) & ((c["sharpe"] - sh0).abs() <= USTOY_TOL + 1e-12)].copy()
    near["_sel"] = (near["family"] == best["family"]) & (near["point"] == best["point"]) & \
                   (near["sigma"] == best["sigma"])
    near = near.sort_values(["_sel", "sharpe"], ascending=[False, False])
    for _, x in near.iterrows():
        w, p = _okno_porog(x["point"])
        rows.append(dict(rol="(б) выбранная" if x["_sel"] else
                         f"(б) прошла С8, Шарп в пределах {TOL_S} от выбранной",
                         family=x["family"], point=x["point"], sigma=x["sigma"], h=x["h"],
                         okno=w, porog=p, sharpe=x["sharpe"], d_k_vybrannoy=x["sharpe"] - sh0,
                         win=x["win"], passed=x["passed"], d_vs_A=x["d_vs_A"], p_vs_A=x["p_vs_A"],
                         p_vs_A_iid=x["p_vs_A_iid"], p_holm=x["p_holm"]))
    w0, p0 = _okno_porog(best["point"])
    same = c[(c["family"] == best["family"]) & (c["sigma"] == best["sigma"])].copy()
    same["okno"], same["porog"] = zip(*same["point"].map(_okno_porog))
    by_w = same[(same["porog"] == p0) & (same["okno"] != w0)].sort_values("okno")
    by_p = same[(same["okno"] == w0) & (same["porog"] != p0)].sort_values("porog")
    nb = pd.concat([by_w.assign(os_="окно"), by_p.assign(os_="порог")])
    ok = nb["passed"].astype(bool) & ((nb["sharpe"] - sh0).abs() <= USTOY_TOL + 1e-12)
    for (_, x), o in zip(nb.iterrows(), ok):
        rows.append(dict(rol=f"(в) сосед по оси «{x['os_']}»", family=x["family"], point=x["point"],
                         sigma=x["sigma"], h=x["h"], okno=x["okno"], porog=x["porog"],
                         sharpe=x["sharpe"], d_k_vybrannoy=x["sharpe"] - sh0, win=x["win"],
                         passed=x["passed"], d_vs_A=x["d_vs_A"], p_vs_A=x["p_vs_A"],
                         p_vs_A_iid=x["p_vs_A_iid"], p_holm=x["p_holm"], v_predelah_0_1=bool(o)))
    nxt_w = by_w[by_w["okno"] > w0]["okno"].min() if len(by_w) else None
    nxt_p = by_p["porog"].iloc[0] if len(by_p) else None
    imm = nb[((nb["os_"] == "окно") & (nb["okno"] == nxt_w)) | ((nb["os_"] == "порог") &
                                                               (nb["porog"] == nxt_p))]
    imm_ok = ok.loc[imm.index]
    n_w, n_p = int(ok[nb["os_"] == "окно"].sum()), int(ok[nb["os_"] == "порог"].sum())
    verdict = ("плато" if len(imm) and bool(imm_ok.all()) else
               "пик" if int(ok.sum()) == 0 else "частичное плато")
    rows.append(dict(rol="(в) вывод", family=best["family"], point=best["point"], sigma=best["sigma"],
                     h=best["h"], okno=w0, porog=p0, sharpe=sh0,
                     n_C8_v_predelah=int((~near["_sel"]).sum()), n_kandidatov=len(c) - 1,
                     vyvod=(f"{verdict}: соседей {best['family']} {best['point']} {best['h']} в пределах "
                            f"{TOL_S} и с С8 — по окну {n_w} из {len(by_w)}, по порогу {n_p} из "
                            f"{len(by_p)}; всего точек С8 в пределах {TOL_S} — "
                            f"{int((~near['_sel']).sum())} из {len(c) - 1}"
                            + "".join(f"; {x['family']} {x['point']} {x['h']} ({x['sharpe']:.3f})"
                                      for _, x in near[~near["_sel"]].iterrows()))))
    return pd.DataFrame(rows)


def _porog_key(p) -> tuple:
    return tuple(int(x) for x in str(p).split("/")) if p != "—" else (999,)


def ustoychivost_c8v2(dfc: pd.DataFrame) -> pd.DataFrame:
    """Ъ 27.10, 00:10 (--c8v2): устойчивость выбора C по правилу С8 v2 на сетке с p50/p60.
    (а) C1 W504 p70 h=5 (точка первого FINAL); (б) точки, прошедшие С8 v2, с |Шарп − Шарп выбранной|
    ≤ 0,1 (выбранная — первой); (в) соседи выбранной по окну (тот же порог) и по порогу (то же окно)
    той же семьи и σ — все, с к2/к3. Вывод (правило объявлено до счёта, обобщение прежнего на
    двусторонних соседей): плато — все ближайшие соседи (по одному с каждой стороны на каждой оси)
    в пределах 0,1 и прошли С8 v2; пик — ни один сосед; иначе — частичное плато."""
    c = dfc[dfc["sigma"].isin(["hyb_h1", "hyb_h5"])].copy()
    c["p_holm"] = _holm(c["p_vs_A"].values)
    best = rank_c8_v2(c).iloc[0]
    sh0 = float(best["sharpe"])
    pv2 = c["passed_v2"].astype(bool)
    cols = ("win", "passed", "passed_v2", "d_vs_A", "p_vs_A", "p_vs_A_iid", "p_holm", *K_COLS)
    rows = []
    fam, pt, sg = C_FINAL1
    a = c[(c["family"] == fam) & (c["point"] == pt) & (c["sigma"] == sg)]
    if len(a) != 1:
        raise SystemExit(f"устойчивость: строк {C_FINAL1} — {len(a)}")
    r = dict(rol="(а) точка первого FINAL: C1 W504 p70 h=5 на гибриде (новые σ, 0,04%)")
    r.update({k: v for k, v in a.iloc[0].items() if k != "pometka"})
    r["p_holm"] = a.iloc[0]["p_holm"]
    r["d_k_vybrannoy"] = float(a.iloc[0]["sharpe"]) - sh0
    r["okno"], r["porog"] = _okno_porog(pt)
    rows.append(r)

    def short(x, rol, **kw):
        w, p = _okno_porog(x["point"])
        d = dict(rol=rol, family=x["family"], point=x["point"], sigma=x["sigma"], h=x["h"], okno=w,
                 porog=p, sharpe=x["sharpe"], d_k_vybrannoy=x["sharpe"] - sh0)
        d.update({k: x[k] for k in cols if k in x.index})
        d.update(kw)
        return d
    near = c[pv2 & ((c["sharpe"] - sh0).abs() <= USTOY_TOL + 1e-12)].copy()
    near["_sel"] = (near["family"] == best["family"]) & (near["point"] == best["point"]) & \
                   (near["sigma"] == best["sigma"])
    near = near.sort_values(["_sel", "sharpe"], ascending=[False, False])
    for _, x in near.iterrows():
        rows.append(short(x, "(б) выбранная" if x["_sel"] else
                          f"(б) прошла С8 v2, Шарп в пределах {TOL_S} от выбранной"))
    near_c8 = int((c["passed"].astype(bool) & ((c["sharpe"] - sh0).abs() <= USTOY_TOL + 1e-12)).sum()) - \
        int(bool(best["passed"]))
    w0, p0 = _okno_porog(best["point"])
    same = c[(c["family"] == best["family"]) & (c["sigma"] == best["sigma"])].copy()
    same["okno"], same["porog"] = zip(*same["point"].map(_okno_porog))
    by_w = same[(same["porog"] == p0) & (same["okno"] != w0)].sort_values("okno")
    by_p = same[(same["okno"] == w0) & (same["porog"] != p0)].copy()
    keys = [_porog_key(v) for v in by_p["porog"]]
    by_p = by_p.iloc[sorted(range(len(keys)), key=lambda i: keys[i])]
    imm = set()
    lo_w, hi_w = by_w[by_w["okno"] < w0], by_w[by_w["okno"] > w0]
    if len(lo_w):
        imm.add(lo_w.index[-1])
    if len(hi_w):
        imm.add(hi_w.index[0])
    k0 = _porog_key(p0)
    lo_p = [i for i, v in zip(by_p.index, by_p["porog"]) if _porog_key(v) < k0]
    hi_p = [i for i, v in zip(by_p.index, by_p["porog"]) if _porog_key(v) > k0]
    if lo_p:
        imm.add(lo_p[-1])
    if hi_p:
        imm.add(hi_p[0])
    nb = pd.concat([by_w.assign(os_="окно"), by_p.assign(os_="порог")])
    ok = nb["passed_v2"].astype(bool) & ((nb["sharpe"] - sh0).abs() <= USTOY_TOL + 1e-12)
    for (i, x), o in zip(nb.iterrows(), ok):
        rows.append(short(x, f"(в) сосед по оси «{x['os_']}»" + (" — ближайший" if i in imm else ""),
                          v_predelah_0_1=bool(abs(x["sharpe"] - sh0) <= USTOY_TOL + 1e-12),
                          v_predelah_0_1_i_C8_v2=bool(o)))
    imm_ok = [bool(o) for (i, _), o in zip(nb.iterrows(), ok) if i in imm]
    n_w, n_p = int(ok[nb["os_"] == "окно"].sum()), int(ok[nb["os_"] == "порог"].sum())
    verdict = ("плато" if imm_ok and all(imm_ok) else "пик" if int(ok.sum()) == 0 else "частичное плато")
    n_near = int((~near["_sel"]).sum())
    rows.append(dict(rol="(в) вывод", family=best["family"], point=best["point"], sigma=best["sigma"],
                     h=best["h"], okno=w0, porog=p0, sharpe=sh0, passed_v2=bool(best["passed_v2"]),
                     n_C8_v2_v_predelah=n_near, n_C8_v_predelah=near_c8, n_kandidatov=len(c) - 1,
                     n_proshli_C8_v2=int(pv2.sum()), n_proshli_C8=int(c["passed"].astype(bool).sum()),
                     vyvod=(f"{verdict}: соседей {best['family']} {best['point']} {best['h']} в пределах "
                            f"{TOL_S} и с С8 v2 — по окну {n_w} из {len(by_w)}, по порогу {n_p} из "
                            f"{len(by_p)}; ближайших в пределах и с С8 v2 — {sum(imm_ok)} из {len(imm_ok)}; "
                            f"всего точек С8 v2 в пределах {TOL_S} — {n_near} из {len(c) - 1}"
                            + "".join(f"; {x['family']} {x['point']} {x['h']} ({x['sharpe']:.3f})"
                                      for _, x in near[~near["_sel"]].iterrows()))))
    return pd.DataFrame(rows)


def sravnenie_komissiy(v04: pd.DataFrame, v05: pd.DataFrame) -> pd.DataFrame:
    """Ъ 22:45/22:55: совпал бы выбор С8 на 0,05% с выбором на 0,04% (по подходам); отбор не меняет."""
    rows = []
    for ap in ("B", "C", "D"):
        a, b = v04[v04["approach"] == ap].iloc[0], v05[v05["approach"] == ap].iloc[0]
        same = (a["design"], a["sigma"]) == (b["design"], b["sigma"])
        rows.append(dict(approach=ap, vybor_0_04=f"{a['design']} на {a['sigma']}",
                         sharpe_0_04=a["sharpe"], passed_0_04=a["passed"],
                         vybor_0_05=f"{b['design']} на {b['sigma']}", sharpe_0_05=b["sharpe"],
                         passed_0_05=b["passed"], sovpal="да" if same else "нет",
                         primechanie="справочно: отбор — на 0,04%, по этой таблице не меняется"))
        if ap == "D":
            rows[-1]["proverka_RVI"] = D_RVI
    return pd.DataFrame(rows)


def frozen_keys() -> dict:
    """Тест: выбор по С8 из таблиц разработки того же σ (docs/tables/strat_dev), без перевыбора."""
    suf = SUFFIX[:-len(f"_{SCOPE}")] if SCOPE != "dev" else SUFFIX

    def rd(name):
        p = TABLES_DEV / name.replace(".csv", f"{suf}.csv")
        d = pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig" if suf else "utf-8")
        for c in ("passed", "positive"):
            if c in d:
                d[c] = d[c].map({True: True, False: False, "True": True, "False": False})
        return d
    k = select_keys(rd("s9_D_zatvor.csv"), rd("s9_B_targetirovanie.csv"), rd("s9_C_rezhim.csv"))
    if k["C"] != TEST_C:
        raise SystemExit(f"C в тест задан владельцем ({TEST_C}), а таблицы разработки дают {k['C']}")
    return k


def split_c_test(dfc: pd.DataFrame, ckey: tuple) -> tuple:
    """Тест (владелец 26.10, 10:00): основная строка C — одна заданная точка; остальная сетка C —
    отдельно, «чувствительность»: справочно, без выбора и без проверки гипотез."""
    name, sig = ckey
    if name == "Cref":
        m = dfc["family"].str.startswith("C основного")
    elif name.endswith(" семья"):
        m = (dfc["family"] == name.replace(" семья", " (выбор по валидации)")) & (dfc["point"] == "семья")
    else:
        fam, pt = name.split(":", 1)
        m = (dfc["family"] == fam) & (dfc["point"] == pt)
    m = m & (dfc["sigma"] == sig)
    if m.sum() != 1:
        raise SystemExit(f"основная точка C {ckey}: найдено строк {int(m.sum())}, нужна одна")
    main = dfc[m].assign(rol="основная: одна точка, выбрана по С8 на разработке; на тесте выбора нет")
    sens = dfc[~m].assign(rol="чувствительность: справочно, без выбора и без проверки гипотез")
    return main, sens


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sources", "cache", "check", "run"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--scope", choices=["dev", "test"], default="dev")
    p.add_argument("--sigma-h1", choices=sorted(SIGMA_H1), default=None,
                   help="σ_pred гибрида h=1 (final3 — финал третьего круга D22_h1); "
                        "по умолчанию dev — final2, test — final3")
    p.add_argument("--suffix", default=None, help="суффикс таблиц/кэша (по умолчанию от --sigma-h1)")
    p.add_argument("--sigma-set", choices=["old", "v2"], default="old",
                   help="v2 — выровненные σ S0 (data/regen/ch4_dev), только dev")
    p.add_argument("--comm", type=float, default=None,
                   help="v2: комиссия МСП/выбора за сторону (по умолчанию 0,0004 — Ъ 22:45)")
    p.add_argument("--val-start", type=int, default=None,
                   help="начало замороженного окна валидации в тесте (по умолчанию 2018; конец — "
                        "2021); в разработке всегда 2017…Y−1")
    p.add_argument("--c-dop", nargs="?", const=str(C_DOP_YAML), default=None,
                   help="новые точки сетки C (Ъ 22:30) из yaml, раздел dop; без флага — как было")
    p.add_argument("--c8v2", action="store_true",
                   help="Ъ 27.10, 00:10: сетка C + p50/p60 (setka_C.yaml: dop.rassh), контроли к2/к3, "
                        "правило С8 v2; только с --sigma-set v2 и --c-dop")
    a = p.parse_args(argv)
    from fastbt import dev
    dev.check_scope(a.scope)               # замок: test без FINAL_DESIGN_strategies.md — отказ
    if a.c8v2:
        set_c8v2(a.scope, a.sigma_set, a.c_dop)
    if a.c_dop:
        if a.scope != "dev":               # √RV_w из ch4_dev — только ≤ 2021: на тесте C* был бы = A
            raise SystemExit("--c-dop только для --scope dev (нет √RV_w на тест)")
        load_c_dop(a.c_dop)
    set_scope(a.scope, a.val_start)
    if a.sigma_set == "v2":
        print("σ v2 (SHA сверены):", {k: v[:12] for k, v in use_v2(a.comm).items()},
              f"комиссия {COMM * 100:.3f}%, суффикс {SUFFIX}", flush=True)
        if a.stage == "sources":           # σ — готовые файлы S0; RVI — прежний rvi_1850 (не σ_pred)
            print("v2: sources не собираются — σ из data/regen/ch4_dev; RVI:", RVI.name)
            return 0
    else:
        use_sigma_h1(a.sigma_h1 or ("final2" if a.scope == "dev" else "final3_full"), a.suffix)
    if a.stage == "sources":
        build_sources()
    elif a.stage == "cache":
        build_cache(a.workers)
    elif a.stage == "check":
        eng = Engine(load_cache())
        x = S9(eng, 200)
        print(check_against_pipeline(eng, x))
        for nm, p in [("A", x.A)] + [(f"C {lab}", x.C[lab]) for lab in SOURCES]:
            m = metrics(p)
            print(nm, f"{m['sharpe']:.3f}", " ".join(f"{m[str(y)]:.2f}" for y in YEARS))
    else:
        run_all_experiments(a.boot_reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
