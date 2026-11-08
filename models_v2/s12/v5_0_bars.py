"""V5-0, G4: цель RV и разложение Парсеваля. По коду проекта (`models_v2/daily.py`: `main_session`, `intraday_returns`, `build_daily`):
RV дня = 1e4 · Σ r_i², r_1 = ln(close_1/open_1), r_i = ln(close_i/close_{i-1}); бары основной сессии begin ∈ [10:00; 18:40] (10-минутные); ночная доходность (close_{d-1} → open_d) в RV НЕ входит, масштаб %², фильтры дня: ≥ 30 баров,
остановка 28.02–24.03.2022, с 01.03.2025 — только дни сессии IMOEX. Поэтому разложение точное: ортонормальное DCT-II дневного вектора r длины n даёт Σ c_k² = Σ r_i² = RV/1e4. Полосы объявлены до счёта: low 3k < n, mid n ≤ 3k < 2n, high 3k ≥ 2n (k = 0…n−1).
Ночного элемента в цели нет, отдельная полоса — нулевая (определение, не допущение)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P
from models_v2.s12 import v5_0_io as IO
from models_v2.s9 import hv2_m10 as M

TOL_ID = 1e-10
TOL_PARSEVAL = 1e-12


def band_energy(x) -> tuple:
    """Дневной вектор доходностей -> (Σ r², Σ c² low, mid, high) без масштаба; c = ортонормальное DCT-II."""
    x = np.asarray(x, float)
    c = dct(x, type=2, norm="ortho") ** 2
    k, n = np.arange(len(x)), len(x)
    return float((x * x).sum()), float(c[3 * k < n].sum()), float(c[(3 * k >= n) & (3 * k < 2 * n)].sum()), float(c[3 * k >= 2 * n].sum())


def day_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Бары -> по дням: n, rv_bars (1e4·Σr²), e_low/e_mid/e_high (1e4·Σc² полосы), e_tot, herf (Σq²/(Σq)², q = r²). Боевой путь: те же доходности, что `hv2_m10.bar_returns`."""
    r = M.bar_returns(bars)
    rows = {}
    for d, x in r.groupby("date")["r"]:
        v = x.to_numpy(float)
        s, lo, mi, hi = band_energy(v)
        q = v * v
        rows[d] = (len(v), s * D.SCALE, lo * D.SCALE, mi * D.SCALE, hi * D.SCALE, float((q * q).sum() / s ** 2) if s > 0 else np.nan)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["n", "rv_bars", "e_low", "e_mid", "e_high", "herf"]).rename_axis("date")


def rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.abs(a - b) / np.abs(b)).max())


def ticker_bands(t: str) -> pd.DataFrame:
    """Таблица дня бумаги на канонических датах `daily_<T>.parquet` + RV канона, j, окна, rv_close и признак sched_ok."""
    d = pd.read_parquet(P.OUT / ("daily_%s.parquet" % t))
    d.index = pd.to_datetime(d.index)
    b = day_table(pd.read_parquet(P.bars_path(t))).reindex(d.index)
    b["rv"] = d["rv"]
    for c in ("j", "rv_w", "rv_m", "rv_close", "sched_ok", "n_bars"):
        b[c] = d[c]
    return b


def all_bands() -> pd.DataFrame:
    fr = []
    for t in IO.T17:
        x = ticker_bands(t)
        x.insert(0, "ticker", t)
        fr.append(x.reset_index())
    return pd.concat(fr, ignore_index=True)


def cache_key() -> str:
    """Хэш входов (бары 17 бумаг и прогнозы) + хэш кода, который строит энергии полос (этот файл)."""
    return IO.inputs_hash() + ":" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def cached_bands(write: bool = False) -> pd.DataFrame:
    """Кэш `data/regen/harj_hybrid_v5/bands.parquet`; ключ = хэш входов и кода (Д5) в имени файла. `write=False` (проверка) кэш не создаёт — пересчёт в памяти."""
    p = IO.OUT / ("bands_%s.parquet" % cache_key().replace(":", "_"))
    if p.exists():
        return pd.read_parquet(p)
    b = all_bands()
    if write:
        IO.OUT.mkdir(parents=True, exist_ok=True)
        b.to_parquet(p)
    return b


def identity_table(b: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """PASS_RV: по бумаге — число бумаго-дней, max отн. разн. Σr² против RV канона и против цели `rv_tplus` (по t⁺), Парсеваль Σc² против Σr², распределение n."""
    ok = b.dropna(subset=["rv"])
    tgt = k.merge(ok[["ticker", "date", "rv_bars"]].rename(columns={"date": "t_plus"}), on=["ticker", "t_plus"], how="left")
    rows = []
    for t, g in ok.groupby("ticker"):
        tg = tgt[tgt["ticker"] == t]
        e = g[["e_low", "e_mid", "e_high"]].sum(axis=1)
        mode = int(g["n"].mode().iloc[0])
        rows.append(dict(section="identity", ticker=t, n_days=len(g), n_target_days=int(tg["rv_bars"].notna().sum()), max_rel_bars_vs_daily=rel(g["rv_bars"], g["rv"]),
                         max_rel_bars_vs_target=rel(tg["rv_bars"].dropna(), tg.loc[tg["rv_bars"].notna(), "rv_tplus"]), max_rel_parseval=rel(e, g["rv_bars"]),
                         n_modal=mode, share_n_not_modal=float((g["n"] != mode).mean()), share_sched_not_ok=float((~g["sched_ok"].astype(bool)).mean())))
    return pd.DataFrame(rows)


def n_distribution(b: pd.DataFrame) -> pd.DataFrame:
    """Распределение числа M10-доходностей дня n по бумаге × году."""
    g = b.dropna(subset=["rv"]).assign(year=lambda x: x["date"].dt.year).groupby(["ticker", "year"])["n"]
    t = g.agg(n_days="size", n_min="min", n_median="median", n_max="max").reset_index()
    t["section"] = "n_by_year"
    return t


def definition_rows() -> pd.DataFrame:
    txt = [("bars", "основная сессия: begin ∈ [10:00; 18:40], 10-минутные бары (daily.main_session)"), ("first_bar", "r_1 = ln(close_1/open_1); далее close-к-close внутри дня"), ("overnight", "ночная доходность в цель НЕ входит; полоса «ночь» = 0"),
           ("scale", "RV = 1e4·Σ r² (%²); порог дня: ≥ 30 баров; остановка 28.02–24.03.2022 исключена; с 01.03.2025 дни без сессии IMOEX исключены"),
           ("dct", "ортонормальное DCT-II; low 3k<n, mid n≤3k<2n, high 3k≥2n"), ("exact", "Σc²=Σr²=RV/1e4: точное разложение; расхождений нет")]
    return pd.DataFrame([dict(section="definition", item=a, note=b) for a, b in txt])
