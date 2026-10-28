"""Д12.1: дневные RV/BV/J/RSV±/RQ 28 тикеров со срезом 18:50 из ISS-баров (крейт moex_iss).

Вход: data/iss_check/bars_10m_full/{T}.parquet (все бары ISS TQBR, begin МСК).
Карантин: та же функция пайплайна moex_discovery/src/pipeline_d001_d003.py:201-246 (detect_quarantine_flags,
импортируется из файла как есть) на полном ряду баров тикера — одинаково для всех 28.
Окно: begin в [09:50; 18:50), бары карантина выброшены. Формула — moex_iss/src/rv.rs::daily
(r = ln(c_i/c_{i-1}) по всему отфильтрованному ряду, первая доходность дня — от последнего бара прошлого дня;
BV = π/2·Σ|r_i||r_{i-1}| с лагом внутри дня; J = max(RV−BV, 0); RSV± = Σr² при r>0 / r<0);
добавлено RQ = (M/3)·Σr⁴, M — число доходностей дня.
Выход: data/iss_check/rv_1850_noquar_28.parquet (+ rv_1850_noquar_28_isskar.parquet — карантин из хранимых
файлов пайплайна для сверки), сводки в docs/otchety/d121/.
"""
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba
import numpy as np
import pandas as pd

MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
ISS = MAIN / "data/iss_check"
ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "docs/otchety/d121"
MAIN17 = ["SBER", "VTBR", "MOEX", "LKOH", "TATN", "NVTK", "ALRS", "MTLR", "HYDR", "IRAO", "OGKB", "RTKM",
          "MTSS", "MGNT", "AFLT", "PHOR", "LSRG"]
EXTRA11 = ["TGKA", "MSNG", "TRMK", "UTAR", "RGSS", "MSRS", "GCHE", "SELG", "MRKC", "PIKK", "MRKP"]
T28 = MAIN17 + EXTRA11
W0, W1 = 9 * 60 + 50, 18 * 60 + 50  # окно begin в [09:50; 18:50), минуты от полуночи
FIRST_FEAT = "2014-06-09"  # первый день хранимых признаков/матрицы
Q_FROM = "2014-06-09"  # DATE_FROM пайплайна
STORED_Q = MAIN / "moex_discovery/data/quarantine/candles_10m"


def _pipeline_quarantine():
    p = ROOT / "moex_discovery/src/pipeline_d001_d003.py"
    spec = importlib.util.spec_from_file_location("pipeline_d001_d003", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.detect_quarantine_flags


def load_full(t):
    b = pd.read_parquet(ISS / f"bars_10m_full/{t}.parquet")
    b = b.rename(columns={"begin": "timestamp"})
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    return b.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def quarantine_ts(full):
    """Метки времени баров карантина — функцией пайплайна на полном ряду (как D-002 на сыром файле)."""
    f = _pipeline_quarantine()(full[["timestamp", "open", "high", "low", "close", "volume"]])
    return set(f.loc[f["is_quarantine"], "timestamp"].values.astype("datetime64[ms]").astype(np.int64))


@numba.njit(cache=True)
def _daily_core(c, day):
    """Тот же цикл, что moex_iss/src/rv.rs::daily (порядок сложений сохранён -> побитовое совпадение), + Σr⁴."""
    n = len(c)
    out_day = np.empty(n, np.int64)
    acc = np.zeros((n, 6))  # rv, bv, rsv_pos, rsv_neg, r4, n
    k = -1
    prev_abs = 0.0
    has_prev = False
    for i in range(1, n):
        p, b = c[i - 1], c[i]
        if p <= 0.0 or b <= 0.0:
            continue
        r = np.log(b / p)
        if k < 0 or out_day[k] != day[i]:
            k += 1
            out_day[k] = day[i]
            has_prev = False
        if has_prev:
            acc[k, 1] += abs(r) * prev_abs
        acc[k, 0] += r * r
        if r > 0.0:
            acc[k, 2] += r * r
        elif r < 0.0:
            acc[k, 3] += r * r
        acc[k, 4] += r * r * r * r
        acc[k, 5] += 1.0
        prev_abs = abs(r)
        has_prev = True
    return out_day[:k + 1], acc[:k + 1]


def daily(bars):
    """Порт moex_iss/src/rv.rs::daily + RQ = (M/3)·Σr⁴. bars — отсортированы, в окне, без карантина."""
    ts = bars["timestamp"].values.astype("datetime64[ms]").astype(np.int64)
    d, a = _daily_core(bars["close"].to_numpy(np.float64), ts // 86_400_000)
    out = pd.DataFrame({"rv": a[:, 0], "bv": a[:, 1] * (np.pi / 2), "rsv_pos": a[:, 2], "rsv_neg": a[:, 3],
                        "n_bars": a[:, 5].astype(np.int64)},
                       index=pd.to_datetime(d * 86_400_000, unit="ms"))
    out["j"] = np.maximum(out.rv - out.bv, 0.0)
    out["rq"] = out.n_bars / 3.0 * a[:, 4]
    return out


def in_window(ts):
    m = ts.dt.hour * 60 + ts.dt.minute
    return (m >= W0) & (m < W1)


def one(t):
    full = load_full(t)
    # как rv1850.rs: бары с дня (первый день признаков 2014-06-09) − 5 (влияет только на первую доходность ряда)
    full_rv = full[full.timestamp >= pd.Timestamp(FIRST_FEAT) - pd.Timedelta(days=5)]
    # карантин — на ряду с даты начала выгрузки пайплайна (DATE_FROM, pipeline_d001_d003.py:35), как D-002
    q_new = quarantine_ts(full[full.timestamp >= pd.Timestamp(Q_FROM)].reset_index(drop=True))
    q_old = set()
    sq = STORED_Q / f"{t}.parquet"
    if sq.exists():
        q_old = set(pd.to_datetime(pd.read_parquet(sq)["timestamp"]).values.astype("datetime64[ms]").astype(np.int64))
    ts_ms = full_rv.timestamp.values.astype("datetime64[ms]").astype(np.int64)
    w = in_window(full_rv.timestamp).to_numpy()
    res = {}
    for name, q in (("new", q_new), ("old", q_old)):
        isq = np.isin(ts_ms, np.fromiter(q, np.int64, len(q))) if q else np.zeros(len(ts_ms), bool)
        d = daily(full_rv[w & ~isq])
        nq = pd.Series(isq[w]).groupby(full_rv.timestamp[w].dt.normalize().values).sum()
        nw = pd.Series(np.ones(w.sum())).groupby(full_rv.timestamp[w].dt.normalize().values).sum()
        d["n_quar"] = nq.reindex(d.index).fillna(0).astype(int)
        d["n_win_raw"] = nw.reindex(d.index).fillna(0).astype(int)
        d = d.reset_index().rename(columns={"index": "date"})
        d.insert(1, "ticker", t)
        res[name] = d
    # сверка карантина: общий период хранимого сырого файла пайплайна
    raw = pd.read_parquet(MAIN / f"moex_discovery/data/raw/candles_10m/{t}.parquet")
    lo, hi = pd.to_datetime(raw.timestamp).min(), pd.to_datetime(raw.timestamp).max()
    lo_ms, hi_ms = (np.datetime64(x, "ms").astype(np.int64) for x in (lo, hi))
    a = {x for x in q_new if lo_ms <= x <= hi_ms}
    b = {x for x in q_old if lo_ms <= x <= hi_ms}
    allw = ts_ms[(ts_ms >= lo_ms) & (ts_ms <= hi_ms)]
    info = {"ticker": t, "bars_full": int(len(full)), "first_bar": str(full.timestamp.min()),
            "last_bar": str(full.timestamp.max()), "quar_new": len(q_new), "quar_old": len(q_old),
            "quar_cmp_both": len(a & b), "quar_cmp_only_new": len(a - b), "quar_cmp_only_old": len(b - a),
            "quar_share_full": round(len(q_new) / max(len(full), 1), 6), "bars_cmp_period": int(len(allw))}
    return res["new"], res["old"], info


def main():
    old17 = pd.read_parquet(ISS / "rv_1850_noquar.parquet")
    old17["date"] = pd.to_datetime(old17["date"])
    with ProcessPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(one, T28))
    new = pd.concat([o[0] for o in out], ignore_index=True)
    old = pd.concat([o[1] for o in out], ignore_index=True)
    info = pd.DataFrame([o[2] for o in out])
    cols = ["date", "ticker", "rv", "bv", "j", "rsv_pos", "rsv_neg", "rq", "n_bars", "n_quar", "n_win_raw"]
    for d in (new, old):
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d["is_main17"] = d.ticker.isin(MAIN17)
    new[cols + ["is_main17"]].to_parquet(ISS / "rv_1850_noquar_28.parquet", index=False)
    old[cols + ["is_main17"]].to_parquet(ISS / "rv_1850_noquar_28_isskar.parquet", index=False)
    OUTD.mkdir(parents=True, exist_ok=True)
    info.to_csv(OUTD / "karantin_sverka.csv", index=False)
    # сверка 17 основных с существующим rv_1850_noquar.parquet
    rows = []
    for lab, d in (("карантин_новый", new), ("карантин_хранимый", old)):
        x = d[d.is_main17].copy()
        x["date"] = pd.to_datetime(x["date"])
        m = old17.merge(x, on=["date", "ticker"], how="outer", suffixes=("_e", ""), indicator=True)
        both = m[m._merge == "both"]
        for c in ["rv", "bv", "j", "rsv_pos", "rsv_neg", "n_bars", "n_quar"]:
            eq = np.isclose(both[c].astype(float), both[c + "_e"].astype(float), rtol=1e-12, atol=0)
            ex = (both[c].astype(float) == both[c + "_e"].astype(float))
            rows.append({"вариант": lab, "колонка": c, "общих": len(both), "только_сущ": int((m._merge == "left_only").sum()),
                         "только_новый": int((m._merge == "right_only").sum()), "совп_1e-12_%": round(100 * eq.mean(), 4),
                         "точно_%": round(100 * ex.mean(), 4)})
    sv = pd.DataFrame(rows)
    sv.to_csv(OUTD / "sverka17_rv.csv", index=False)
    print(sv.to_string())
    print(info.to_string())


if __name__ == "__main__":
    main()
