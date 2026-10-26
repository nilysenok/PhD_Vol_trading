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

from fastbt.cache import MAIN

ROOT = Path(__file__).resolve().parent.parent
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


def use_sigma_h1(name: str, suffix: str | None = None) -> None:
    global SUFFIX
    SOURCES["hyb_h1"] = (SIGMA_H1[name], "sigma_pred_h1", "h1")
    tag = name.replace("_full", "")
    SUFFIX = ("" if tag == "final2" else f"_{tag}") if suffix is None else suffix
    if SCOPE != "dev":
        SUFFIX += f"_{SCOPE}"


def to_csv(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
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
    p, col, _ = SOURCES[label]
    d = pd.read_parquet(p, columns=["date", "ticker", col])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= CUT].rename(columns={col: "sigma"})
    return d.sort_values(["ticker", "date"])


def align(dates: np.ndarray, d: pd.DataFrame) -> np.ndarray:
    """Как fastbt.data.align_sigma: последнее значение с датой ≤ даты бара; после CUT — NaN."""
    arr = np.full(len(dates), np.nan)
    vd = pd.to_datetime(d["date"].values)
    idx = np.searchsorted(vd, pd.to_datetime(dates), side="right") - 1
    ok = (idx >= 0) & (pd.to_datetime(dates) <= CUT)
    arr[ok] = d["sigma"].values.astype(float)[idx[ok]]
    return arr


# ======================= кэш A и C основного дизайна =======================

def _job(args):
    ticker, srcs = args[:2]
    if len(args) > 2:                      # тест: в процессе spawn глобальные настройки заново
        set_scope(*args[2])
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
    dsg.apply(cfg.design)
    td = load_ticker(ticker, cfg.corp_actions)
    sig = {lab: (h, align(td.dates, df)) for lab, (h, df) in srcs.items()}
    out = {"dates": td.dates, "lr": td.log_ret, "A": {}, "C": {}, "Cmeta": {}}
    for sid in SIDS:
        dsg.apply(cfg.design)
        u = make_unit(ticker, sid, td)
        a = ApproachA(u, cfg).run()
        a_tf = turnover.apply(a, u, cfg, "A")
        out["A"][sid] = {Y: v[1].astype(np.float64) for Y, v in a_tf.items()}
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
    if SCOPE != "dev":
        h.update(f"{SCOPE}|{VAL_START}|{VAL_END}|{CUT.date()}".encode())
    return h.hexdigest()[:10]


def build_cache(workers: int = 2) -> Path:
    from fastbt.data import TICKERS
    os.environ["FASTBT_VPRED"] = str(SOURCES["hyb_h1"][0])
    srcs = {lab: (SOURCES[lab][2], load_source(lab)) for lab in SOURCES}
    extra = () if SCOPE == "dev" else ((SCOPE, VAL_START),)
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
    (OUT / f"cache_latest{SUFFIX}.txt").write_text(p.name)
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
        kr = rate.reindex(rate.index.union(self.dates)).sort_index().ffill().bfill()
        self.kr = kr.reindex(self.dates).values
        self.rf = self.kr / 365.0
        cal = np.ones(len(self.dates))
        cal[1:] = np.diff(self.dates.values).astype("timedelta64[D]").astype(float)
        self.cal = cal
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
    def port(self, pos, comm=COMM, cap=1.0, wgt=None):
        dp = np.diff(pos, prepend=0.0)
        dp[self.first] = pos[self.first]
        di = self.r["date"]
        nd = len(self.dates)
        if wgt is None:
            net = pos * self.lr_row - np.abs(dp) * comm
            r = np.bincount(di, net, nd) / self.cnt
            gross = np.bincount(di, pos * self.lr_row, nd) / self.cnt
            exp_ = np.bincount(di, np.abs(pos), nd) / self.cnt
            turn = np.bincount(di, np.abs(dp), nd) / self.cnt
        else:
            eff = pos * wgt
            de = np.diff(eff, prepend=0.0)
            de[self.first] = eff[self.first]
            r = np.bincount(di, eff * self.lr_row - np.abs(de) * comm, nd)
            gross = np.bincount(di, eff * self.lr_row, nd)
            exp_ = np.bincount(di, np.abs(eff), nd)
            turn = np.bincount(di, np.abs(de), nd)
        return self._finish(r, gross, exp_, turn, cap)

    def _finish(self, r, gross, exp_, turn, cap):
        used = cap * exp_
        fdr = np.maximum(0.0, 1.0 - used) * self.rf * self.cal
        mg = np.maximum(0.0, used - 1.0) * (self.kr + 0.02) / 365.0
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
        """Доля закрытых баров 2018–2021 и переключений в год (в среднем по парам)."""
        closed = 1.0 - g
        ch = np.r_[0.0, np.abs(np.diff(g))] > 1e-9
        ch[self.first] = False
        pid = self.r["pair"]
        npair = pid.max() + 1
        cl = np.bincount(pid, closed, npair) / np.maximum(np.bincount(pid, None, npair), 1)
        sw = np.bincount(pid, ch.astype(float), npair) / len(YEARS)
        return dict(closed_dev=float(cl.mean()), switches_y=float(sw.mean()),
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
        if SCOPE == "dev":
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
        for W in (126, 252):
            rk[W] = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                rk[W][ti, :n] = past_rank(S[ti, :n].copy(), W, minp(W))
        for W in (126, 252):
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
            for W in (126, 252):
                R = np.full((T, N), np.nan)
                for ti in range(T):
                    n = e.n_t[ti]
                    R[ti, :n] = past_rank(Se[ti, :n].copy(), W, minp(W))
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
        for W in (126, 252):
            med = np.full((T, N), np.nan)
            for ti in range(T):
                n = e.n_t[ti]
                med[ti, :n] = past_median(S[ti, :n], W)
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
                rk[W][ti, :n] = past_rank(S[ti, :n].copy(), W, minp(W))
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
            rows.append(r)
            ports[("Cref", src)] = (self.C[src], self.C_rows[src])
            pts1, pts2 = self.c_points(src)
            for fam, pts in (("C1", pts1), ("C2", pts2)):
                for lab, G in pts:
                    g = e.g_row(G)
                    r, p = self.row(e.A_row * g, fam, lab, h, src,
                                    dict(calib_closed=e.calib_closed(G)), g, cref)
                    rows.append(r)
                    ports[(f"{fam}:{lab}", src)] = (p, e.A_row * g)
                stack = np.stack([G for _, G in pts])
                kind = "mask" if fam == "C1" else "scale"
                g, cnt = e.select(stack, kind, None, cat=True)
                ch = ", ".join(f"{pts[k][0]}×{v}" for k, v in
                               sorted(cnt.items(), key=lambda kv: -kv[1])[:4])
                r, p = self.row(e.A_row * g, f"{fam} (выбор по валидации)", "семья", h, src,
                                dict(choice=ch), g, cref)
                rows.append(r)
                ports[(f"{fam} семья", src)] = (p, e.A_row * g)
        return pd.DataFrame(rows), ports

    # ---------- C3: точность прогноза → результат ----------
    def run_c3(self, c_ports):
        from fastbt.s9_stats import lw_sharpe_diff
        rows = []
        srcs = ("hyb_h1", "harj_h1", "naive_h1")
        kinds = ["Cref", "C1 семья", "C2 семья"] + sorted({k for k, s in c_ports
                                                        if k.startswith("C1:") and s == "hyb_h1"})
        for kind in kinds:
            ps = {s: c_ports[(kind, s)][0] for s in srcs if (kind, s) in c_ports}
            if len(ps) < 3:
                continue
            r = dict(design=kind)
            for s in srcs:
                r[f"sharpe_{s}"] = metrics(ps[s])["sharpe"]
            for a, b in (("hyb_h1", "harj_h1"), ("hyb_h1", "naive_h1"), ("harj_h1", "naive_h1")):
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
        for nm, pos in sel.items():
            for c in (0.0005, 0.0010, 0.0020):
                for sl in (0.0, 0.0005):
                    for sc, cs in (("спот", 1.0), ("маржа 20%", MARGIN)):
                        m = metrics(e.port(pos, comm=c + sl, cap=cs))
                        rows.append(dict(design=nm, comm_pct=c * 100, slippage_pct=sl * 100,
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
    name = (OUT / f"cache_latest{SUFFIX}.txt").read_text().strip()
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
    dfc, cports = s9.run_c(("hyb_h1", "hyb_h5", "harj_h1", "naive_h1"))
    if SCOPE == "dev":
        to_csv(dfc, "s9_C_rezhim.csv")
    else:                                  # тест: одна заданная точка C + сетка «чувствительность»
        kt = frozen_keys()
        main_c, sens_c = split_c_test(dfc, kt["C"])
        to_csv(main_c, "s9_C_osnovnaya.csv")
        to_csv(sens_c, "s9_C_chuvstvitelnost.csv")
    dfc3 = s9.run_c3(cports)
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
    best_c = rank_c8(cc).iloc[0]
    ckey_name = ("Cref" if best_c["family"].startswith("C основного") else
                 (_key_of(best_c) if best_c["point"] == "семья"
                  else f"{best_c['family']}:{best_c['point']}"))
    return dict(D=(_key_of(best_d, ""), best_d["sigma"]), B=(best_b["point"], best_b["sigma"]),
                C=(ckey_name, best_c["sigma"]), D1=_key_of(best_d1, ""), B1=best_b1["point"],
                passed={k: bool(v["passed"]) for k, v in (("B", best_b), ("C", best_c), ("D", best_d))})


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
    p.add_argument("--val-start", type=int, default=None,
                   help="начало замороженного окна валидации в тесте (по умолчанию 2018; конец — "
                        "2021); в разработке всегда 2017…Y−1")
    a = p.parse_args(argv)
    from fastbt import dev
    dev.check_scope(a.scope)               # замок: test без FINAL_DESIGN_strategies.md — отказ
    set_scope(a.scope, a.val_start)
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
