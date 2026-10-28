"""Входы зоны Z2 «тест-1» (2022-01-01…2024-12-31, прогрев — всё до 2022) для листов реверса: интерфейс fastbt/zona.py.

Выход (в --out, умолчание data/regen/test1_2022_2024/; существующие файлы не перезаписываются):
  daily/ohlcv_daily_1850.parquet — FASTBT_Z2_DAILY: ряд 18:50 тем же кодом (fastbt.daily_1850.build) и из того же источника
                                    (moex_discovery/data/raw/candles_10m/<T>.parquet), что dev-ряд; даты < 2025-01-01;
  imoex/IMOEX.parquet            — FASTBT_Z2_IMOEX: 10-минутки IMOEX (схема begin/end/open/close/high/low/value/volume,
                                    как final_v6/candles_10m/external/IMOEX.parquet — её читает rezhim_e1.imoex_daily), begin < 2025;
  div/                           — FASTBT_Z2_DIV: ABANDON (в проекте нет источника дивидендов 2022–2024, сеть запрещена);
  SHA256SUMS, export.txt, sdvig_d1.csv (проверка сдвига Д1).
Гейт: строки < 2022 нового ряда побайтно равны dev-ряду (data/fastbt/daily/ohlcv_daily_1850.parquet основной копии).
Запуск: python3 scripts/dannye/z2_vkhody.py [--main ПУТЬ_К_ОСНОВНОЙ_КОПИИ] [--out ПАПКА]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

END = pd.Timestamp("2025-01-01")        # исключительно; граница отсечки при чтении
HARD = pd.Timestamp("2025-01-01")       # независимая проверка результата: ни одной даты ≥ 2025 (не зависит от END)
GATE_END = pd.Timestamp("2022-01-01")   # гейт равенства с dev: строки < 2022
SHIFTS = (-2, -1, 0, 1)


def assert_cut(values, what: str) -> None:
    mx = pd.to_datetime(pd.Series(values)).max()
    if pd.notna(mx) and mx >= HARD:
        raise ValueError(f"{what}: есть даты ≥ {HARD.date()} ({mx})")


def build_daily(main: Path, tickers: list[str]) -> pd.DataFrame:
    """Ряд 18:50 за всю историю до 2025 тем же кодом, что dev (зона test1 меняет только границу чтения)."""
    from fastbt import daily_1850, zona
    with zona.zone_ctx("test1"):
        d = daily_1850.build(main, tickers)
    assert_cut(d["date"], "дневной ряд")
    return d


def gate_dev(new: pd.DataFrame, dev: pd.DataFrame) -> int:
    """Строки < 2022 нового ряда равны dev побайтно; возвращает число сверенных строк."""
    a = new[new["date"] < GATE_END].sort_values(["ticker", "date"]).reset_index(drop=True)
    b = dev[dev["date"] < GATE_END].sort_values(["ticker", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    return len(a)


def build_imoex(src: Path) -> pd.DataFrame:
    """10-минутки IMOEX, begin < 2025-01-01: отсечка на чтении (pyarrow-фильтр), строки ≥ 2025 не материализуются."""
    t = pq.read_table(src, filters=[("begin", "<", str(END.date()))]).to_pandas()
    assert_cut(t["begin"], "IMOEX")
    return t


def imoex_seam(im: pd.DataFrame, daily_ref: pd.DataFrame) -> dict:
    """Стык 2021/2022: последняя цена 2021 и первая 2022; расхождение close 18:50 (10-мин) с дневным close индекса (index_d)."""
    b = im.assign(ts=pd.to_datetime(im["begin"])).sort_values("ts")
    last21 = b[b["ts"] < GATE_END].iloc[-1]
    first22 = b[b["ts"] >= GATE_END].iloc[0]
    m = (b["ts"].dt.hour * 60 + b["ts"].dt.minute) < 18 * 60 + 50
    c = b[m].groupby(b["ts"].dt.normalize())["close"].last()
    r = daily_ref.assign(date=pd.to_datetime(daily_ref["date"])).set_index("date")["close"]
    j = pd.concat([c.rename("c1850"), r.rename("c_day")], axis=1).dropna()
    j = j[(j.index >= GATE_END) & (j.index < END)]
    rel = (j["c1850"] / j["c_day"] - 1).abs()
    return {"last_2021": (str(last21["ts"]), float(last21["close"])), "first_2022": (str(first22["ts"]), float(first22["close"])),
            "dni_2022_24": len(j), "max_otkl_c1850_k_dnevnomu": float(rel.max()), "med_otkl": float(rel.median())}


def sdvig(daily: pd.DataFrame, rv: pd.DataFrame) -> pd.DataFrame:
    """Д1: корреляция log (ln H/L)² дня d с log RV(d+s), s = −2…+1, 2022–2024; максимум должен быть при s = 0."""
    d = daily[(daily["date"] >= GATE_END) & (daily["date"] < END)]
    rv = rv.assign(date=pd.to_datetime(rv["date"]))
    out = {}
    for s in SHIFTS:
        xs, ys = [], []
        for t, g in d.groupby("ticker"):
            full = rv[rv["ticker"] == t].set_index("date")["rv"].sort_index()
            gg = g.sort_values("date")
            x = np.log(np.maximum(np.log(gg["high"] / gg["low"]) ** 2, 1e-12)).to_numpy()
            y = np.log(np.maximum(full.shift(-s).reindex(gg["date"]).to_numpy(), 1e-12))
            ok = np.isfinite(x) & np.isfinite(y) & (np.exp(y) > 1e-12)
            xs.append(x[ok] - x[ok].mean())
            ys.append(y[ok] - y[ok].mean())
        out[s] = float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
    return pd.DataFrame({"shift": list(out), "corr": list(out.values())})


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_once(path: Path, writer) -> bool:
    if path.exists():
        print(f"существует, не перезаписываю: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(path)
    return True


def main(argv=None) -> int:
    from fastbt import daily_1850
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=daily_1850.default_main())
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "regen" / "test1_2022_2024")
    a = ap.parse_args(argv)
    dev = pd.read_parquet(a.main / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet")
    new = build_daily(a.main, sorted(dev["ticker"].unique()))
    print("гейт 2021: строк < 2022 равны dev побайтно:", gate_dev(new, dev))
    rv = pq.read_table(a.main / "data/iss_check/rv_1850_noquar.parquet", filters=[("date", "<", END.to_pydatetime())]).to_pandas()
    sd = sdvig(new, rv)
    print("Д1 сдвиг (corr по s=-2..+1), максимум при s =", int(sd.loc[sd["corr"].idxmax(), "shift"]), "\n" + sd.to_string(index=False))
    im = build_imoex(a.main / "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet")
    ref = pq.read_table(a.main / "data/iss_check/index_d/IMOEX.parquet", filters=[("date", "<", END.date())]).to_pandas()
    print("стык IMOEX:", imoex_seam(im, ref))
    dp, ip = a.out / "daily" / "ohlcv_daily_1850.parquet", a.out / "imoex" / "IMOEX.parquet"
    _write_once(dp, lambda p: new.to_parquet(p, index=False))
    _write_once(ip, lambda p: im.to_parquet(p, index=False))
    _write_once(a.out / "sdvig_d1.csv", lambda p: sd.to_csv(p, index=False))
    (a.out / "SHA256SUMS").write_text("".join(f"{_sha(p)}  {p.relative_to(a.out)}\n" for p in (dp, ip)))
    exp = f"FASTBT_Z2_DAILY={dp.resolve()} FASTBT_Z2_IMOEX={ip.resolve()}   # FASTBT_Z2_DIV: ABANDON (нет источника 2022–2024)"
    (a.out / "export.txt").write_text(exp + "\n")
    print(exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
