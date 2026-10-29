"""Фазы рынка и сектора (Ж-099 пп. 2–3). Правила зашиты, ничего не обучается.

Фаза на день d — по закрытиям до d−1 включительно (решение к 18:50 d, `_LAG = 1`).
  тренд: close > MA200 и MA50 > MA200 → рост; close < MA200 и MA50 < MA200 → падение; иначе боковик;
  волатильность: σ (std лог-доходностей, ddof=1) за 22 дня; ранг в окне 252 значений σ
    (доля значений окна строго меньше текущего) < 0,70 → спокойно, иначе стресс;
  итог — 6 фаз «тренд_волатильность».
Определена с позиции 274 ряда (22 доходности для σ + 252 значения σ → 273-е закрытие, +1 лаг).
До неё `faza = NaN` → вес 1 у потребителя. Данные ≥ 2020 не читаются (фильтр на чтении).

Сектор: индекс = равновзвешенная доходность тикеров сектора из 17 (отраслевых индексов за 2015–2019 нет).
Сектор с одним тикером или LSRG → фаза рынка. Относительная фаза: доходность сектора минус рынка
за 60 дней > 0 → «сильнее», иначе «слабее»; для запасных секторов (= рынок) — NaN.
Запуск: `FASTBT_MAIN=<копия> python3 -m fastbt.fazy` → docs/tables/msp/fazy_doli.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

OKNO_MA_D, OKNO_MA_K, OKNO_SIGMA, OKNO_RANG, OKNO_OTN = 200, 50, 22, 252, 60
POROG = 0.70
_LAG = 1
S, E = pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-01")   # прогрев 2015; ≥ 2020 не читать
ROOT = Path(__file__).resolve().parents[1]
FAZY = [f"{t}_{v}" for t in ("рост", "падение", "боковик") for v in ("спокойно", "стресс")]


def _ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rang(sigma: pd.Series, n: int = OKNO_RANG) -> pd.Series:
    """Доля значений окна из n последних σ, строго меньших текущей; NaN, если в окне есть NaN."""
    a = sigma.to_numpy(float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        r = (w < w[:, -1:]).mean(axis=1)
        r[np.isnan(w).any(axis=1)] = np.nan
        out[n - 1:] = r
    return pd.Series(out, index=sigma.index)


def _indeks_sektora(ret: pd.DataFrame) -> pd.Series:
    """Уровень равновзвешенного индекса по доходностям тикеров (NaN-тикеры в день пропускаются)."""
    return (1 + ret.mean(axis=1, skipna=True).fillna(0)).cumprod()


def _priznaki(close: pd.Series) -> pd.DataFrame:
    ma200, ma50 = _ma(close, OKNO_MA_D), _ma(close, OKNO_MA_K)
    rost = (close > ma200) & (ma50 > ma200)
    pad = (close < ma200) & (ma50 < ma200)
    trend = pd.Series(np.where(rost, "рост", np.where(pad, "падение", "боковик")), index=close.index)
    rang = _rang(np.log(close).diff().rolling(OKNO_SIGMA).std())
    vol = pd.Series(np.where(rang < POROG, "спокойно", "стресс"), index=close.index)
    trend, vol = trend.where(ma200.notna()), vol.where(rang.notna())
    return pd.DataFrame({"trend": trend, "vol": vol, "faza": trend.str.cat(vol, sep="_")}).shift(_LAG)


def zagruzit(koren: Path | None = None, tikery_dir: Path | None = None) -> dict:
    """{'imoex': Series, 'akcii': wide close (≤ 2019), 'karta': тикер→сектор}."""
    koren = koren or ROOT
    im = pd.read_csv(koren / "data/iss_mix_2015_2019/IMOEX.csv", usecols=["TRADEDATE", "CLOSE"])
    im["TRADEDATE"] = pd.to_datetime(im["TRADEDATE"])
    im = im[(im.TRADEDATE >= S) & (im.TRADEDATE < E)].drop_duplicates("TRADEDATE", keep="last")
    imoex = im.set_index("TRADEDATE")["CLOSE"].sort_index()
    import pyarrow.parquet as pq   # фильтр на чтении: строки ≥ 2020 в память не попадают
    main = Path(os.environ.get("FASTBT_MAIN", ROOT))
    t = pq.read_table(main / "data/fastbt/daily/ohlcv_daily_1850.parquet", columns=["ticker", "date", "close"],
                      filters=[("date", "<", E.to_pydatetime()), ("date", ">=", S.to_pydatetime())])
    akcii = t.to_pandas().pivot(index="date", columns="ticker", values="close").reindex(imoex.index)
    sk = pd.read_csv(ROOT / "docs/issledovaniya/2026-10-27_sektory.csv")
    karta = {r.ticker: r.sektor_moex.split(" (")[0].strip() for r in sk.itertuples()}
    return {"imoex": imoex, "akcii": akcii, "karta": karta}


def _daty(daty) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(daty, name="d")


def faza_rynka(daty, dannye: dict | None = None) -> pd.DataFrame:
    d = dannye or zagruzit()
    return _priznaki(d["imoex"]).reindex(_daty(daty))


def _tikery(sektor: str, d: dict) -> list:
    ts = [t for t, s in d["karta"].items() if s == sektor and t in d["akcii"].columns]
    if not any(s == sektor for s in d["karta"].values()):
        raise ValueError(f"неизвестный сектор: {sektor}")
    return ts


def _uroven(sektor: str, d: dict):
    """Уровень индекса сектора или None (один тикер / LSRG → запасной вариант: рынок)."""
    ts = _tikery(sektor, d)
    if len(ts) < 2 or "LSRG" in ts:
        return None
    ret = pd.DataFrame({t: d["akcii"][t].dropna().pct_change() for t in ts}).reindex(d["imoex"].index)
    return _indeks_sektora(ret)


def faza_sektora(sektor: str, daty, dannye: dict | None = None) -> pd.DataFrame:
    d = dannye or zagruzit()
    u = _uroven(sektor, d)
    return (faza_rynka(daty, d) if u is None else _priznaki(u).reindex(_daty(daty)))


def faza_tikera(tiker: str, daty, dannye: dict | None = None) -> pd.DataFrame:
    """Фаза сектора тикера: общая для всего сектора, тикер из своего сектора не исключается (Ж-101)."""
    d = dannye or zagruzit()
    return faza_sektora(d["karta"][tiker], daty, d)


def otn_faza(sektor: str, daty, dannye: dict | None = None) -> pd.Series:
    d = dannye or zagruzit()
    u = _uroven(sektor, d)
    if u is None:
        return pd.Series(np.nan, index=_daty(daty), name="otn", dtype=object)
    m = d["imoex"]
    raz = (u / u.shift(OKNO_OTN) - 1) - (m / m.shift(OKNO_OTN) - 1)
    raz = raz.shift(_LAG)
    lab = pd.Series(np.where(raz > 0, "сильнее", "слабее"), index=raz.index).where(raz.notna())
    return lab.reindex(_daty(daty)).rename("otn")


def doli_po_godam(f: pd.DataFrame, gody=(2016, 2017, 2018, 2019), status: str = "разработка 2016–2019, до аудита") -> pd.DataFrame:
    rows = []
    for g in gody:
        x = f.faza[f.index.year == g]
        for k in FAZY + ["nan"]:
            n = int(x.isna().sum()) if k == "nan" else int((x == k).sum())
            rows.append({"god": g, "faza": k, "dni": n, "dolya": round(n / len(x), 4), "dni_goda": len(x),
                         "status": status})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    im = zagruzit()["imoex"]
    tab = doli_po_godam(faza_rynka(im.index))
    out = ROOT / "docs/tables/msp/fazy_doli.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(tab.pivot(index="faza", columns="god", values="dolya").to_string())
