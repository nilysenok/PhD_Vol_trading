"""Лист Т-1: голые идеи на ТФ — центр S1–S6 × {10m, 30m, 1h, 2h} × train4/В2 и импульс (❓3), замер с/точку.

Центр семьи — средний элемент каждой оси сетки v5 (индекс len//2; ↺ определение центра дирижёру не задано,
берём номинальный по осям · откат: другая точка в `tsentr`, 29.10). Исполнение ячеек — Ж-111 (fastbt.tf_ispolnenie_v5),
комиссия 0,04 % на сторону; в каждой строке — строка издержек (брутто, издержки, нетто, %/год).
Импульс (умолчание ❓3, класс A, ответ — пакет 18:00): пробой канала 20 полных баров на 2ч (бар закончился ≤ 18:30;
Ж-107) на последнем полном баре дня d; вход по open дня d+1 (Д2), выход по close дня d+τ, τ ∈ {1, 2, 3};
сделки не пересекаются. Чтение 10-минуток строго < 2020-01-01. Статус: разработка, до аудита; потолок.

Запуск (сервер): python3 scripts/server/golaya_tf_v5.py --period train4|v2 --out DIR [--zamer N]
На Маке счёт не запускать (Ж-135); --zamer 5 — малая проверка, вывод на экран.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zamer_tf_os_v5 as Z  # noqa: E402

from fastbt import bars_agg as ba  # noqa: E402
from fastbt import period_v2  # noqa: E402
from fastbt import tf_ispolnenie_v5 as X  # noqa: E402
from fastbt import tf_os_v5 as T  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402

GRANITSA = pd.Timestamp("2020-01-01")
PERIODY = {"train4": dict(a="2016-01-01", b="2020-01-01", warm="2015-01-01"),
           "v2": dict(a="2015-01-01", b="2020-01-01", warm=str(period_v2.START.date()))}
TFS = ("10m", "30m", "1h", "2h")
SIDS = tuple(Z.OSI)
KANAL, TAU, LAG = 20, (1, 2, 3), 1                # LAG: вход через LAG торговых дней после сигнала (Д2: ≥ 1)
POLNYH = (18 * 60 + 30 - ba.OPEN_MIN) // (ba.STEP * ba.K["2h"])   # полных 2ч-баров до 18:30 = 4


def tsentr(osi: dict) -> dict:
    """Центр сетки семьи: средний элемент каждой оси."""
    return {k: (v[len(v) // 2].item() if hasattr(v[len(v) // 2], "item") else v[len(v) // 2])
            for k, v in sorted(osi.items())}


def chtenie(ticker: str, end: str, main: Path | None = None) -> pd.DataFrame:
    """10-минутки тикера строго до end; end позже 2020-01-01 — отказ до чтения."""
    if pd.Timestamp(end) > GRANITSA:
        raise ValueError(f"чтение 10-минуток до {end}: разрешено только < 2020-01-01")
    return ba.load_10m(ticker, end, main)


def zagruzit(period: str, main: Path | None = None, tickers=TICKERS) -> dict:
    p = PERIODY[period]
    out = {}
    for t in tickers:
        b = chtenie(t, p["b"], main)
        out[t] = b[b["date"] >= pd.Timestamp(p["warm"])]
    return out


def bary_tf(b10s: dict, tf: str) -> dict:
    return {t: T.ticker_data(T.bary(b, tf), t) for t, b in b10s.items()}


def yacheyki_tf(tds: dict, sid: str, tf: str, period: str, grid: list | None = None) -> list:
    """Строки издержек точек grid (по умолчанию — центр) семьи sid на ТФ tf за период."""
    p = PERIODY[period]
    grid = grid or [tsentr(Z.OSI[sid])]
    return [dict(r, period=period) for r in X.progon(sid, tf, tds, grid, p["a"], p["b"])]


def impuls_signal(b10: pd.DataFrame) -> pd.Series:
    """Сигнал дня d (±1/0) = пробой канала KANAL полных 2ч-баров на последнем полном баре дня (закрыт ≤ 18:30)."""
    f = ba.resample(b10, ba.K["2h"])
    f = f[f["slot"] < POLNYH].sort_values(["date", "slot"]).reset_index(drop=True)
    hi = f["high"].shift(1).rolling(KANAL).max()
    lo = f["low"].shift(1).rolling(KANAL).min()
    s = np.where(f["close"] > hi, 1, np.where(f["close"] < lo, -1, 0))
    s = pd.Series(s, index=f["date"].to_numpy())
    return s[f["slot"].to_numpy() == POLNYH - 1]


def proverka_d2(sig_i: int, vhod_i: int) -> None:
    if vhod_i <= sig_i:
        raise ValueError("Д2: вход в день сигнала или раньше")


def sdelki_impulsa(d: pd.DataFrame, sig: pd.Series, tau: int, a: str, b: str) -> list:
    """[(сторона, open входа, close выхода)] : сигнал дня i, вход open дня i+LAG, выход close дня i+tau; не пересекаются."""
    dt, o, c = d["date"].to_numpy(), d["open"].to_numpy(float), d["close"].to_numpy(float)
    s = sig.reindex(d["date"]).fillna(0).to_numpy()
    lo, hi = np.datetime64(pd.Timestamp(a)), np.datetime64(pd.Timestamp(b))
    out, i = [], 0
    while i < len(dt):
        if s[i] != 0 and lo <= dt[i] < hi and i + tau < len(dt):
            proverka_d2(i, i + LAG)
            out.append((int(s[i]), o[i + LAG], c[i + tau]))
            i += tau
        else:
            i += 1
    return out


def yacheyki_impuls(b10s: dict, period: str, komissiya: float = T.KOMISSIYA) -> list:
    """Строки издержек импульса (sid = IMP, tf = 2h) по τ; сумма доходностей сделок по тикерам, как у ячеек S1–S6."""
    p = PERIODY[period]
    years = (pd.Timestamp(p["b"]) - pd.Timestamp(p["a"])).days / 365.25
    sig = {t: impuls_signal(b) for t, b in b10s.items()}
    dd = {t: ba.daily(b) for t, b in b10s.items()}
    rows = []
    for tau in TAU:
        chasti = []
        for t, d in dd.items():
            tr = sdelki_impulsa(d, sig[t], tau, p["a"], p["b"])
            n_d = int(((d["date"] >= pd.Timestamp(p["a"])) & (d["date"] < pd.Timestamp(p["b"]))).sum())
            br = sum(s * (x / e - 1.0) for s, e, x in tr)
            chasti.append((br, 2 * komissiya * len(tr), len(tr), n_d))
        rows.append(dict(T.stroka_izderzhek("IMP", "2h", dict(kanal=KANAL, tau=tau), chasti, years, komissiya), period=period))
    return rows


def zamer(b10s: dict, period: str, n: int) -> None:
    """Малая проверка скорости: n случайных точек каждой семьи на каждом ТФ (с/точку); в файлы ничего не пишется."""
    rng = np.random.default_rng(0)
    for tf in TFS:
        t0 = time.perf_counter()
        tds = bary_tf(b10s, tf)
        print(f"{period} {tf}: бары {time.perf_counter() - t0:.1f} с", flush=True)
        for sid in SIDS:
            grid = [{k: (x.item() if hasattr(x, "item") else x) for k, x in p.items()} for p in Z.tochki(Z.OSI[sid], n, rng)]
            t1 = time.perf_counter()
            yacheyki_tf(tds, sid, tf, period, grid)
            print(f"{period} {sid} {tf}: {n} точек, {(time.perf_counter() - t1) / n:.2f} с/точку", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--period", choices=sorted(PERIODY), required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--zamer", type=int, default=0, help="N точек на семью и ТФ: только замер")
    ap.add_argument("--main", type=Path, default=Z.MAIN)
    a = ap.parse_args()
    b10s = zagruzit(a.period, a.main)
    if a.zamer:
        return zamer(b10s, a.period, a.zamer)
    if a.out is None:
        ap.error("--out обязателен без --zamer")
    rows = []
    for tf in TFS:
        tds = bary_tf(b10s, tf)
        for sid in SIDS:
            rows += yacheyki_tf(tds, sid, tf, a.period)
    rows += yacheyki_impuls(b10s, a.period)
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"golaya_tf_{a.period}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"{f}: {len(rows)} строк; {T.STATUS}")


if __name__ == "__main__":
    main()
