"""RVI 18:50 зоны Z2 «тест-1» (2017…2024-12-31) для D6lag1 (лист R-а реверса): интерфейс fastbt/s9.py (FASTBT_Z2_RVI).

Источник — 10-минутки RVI (moex_discovery/data/final_v6/candles_10m/external/RVI.parquet, тот же файл, что в
fastbt/vneshnie_1850.py); значение дня d — close последнего бара с началом < 18:50 (daily_1850.aggregate).
Отсечка begin < 2025-01-01 — НА ЧТЕНИИ (pyarrow-фильтр), строки ≥ 2025 не материализуются.
Выход (--out, умолчание data/regen/test1_2022_2024/; существующее не перезаписывается):
  s9/rvi_1850.parquet — колонки date, close_RVI (как data/fastbt/s9/rvi_1850.parquet), SHA256SUMS, export.txt, sdvig_d1_rvi.csv.
Гейты: 2021 (и раньше) побайтно (≤ 1e-12) = dev-ряд основной копии; нет дат ≥ 2025; Д1.
Запуск: python3 scripts/dannye/z2_rvi.py [--main ПУТЬ_К_ОСНОВНОЙ_КОПИИ] [--out ПАПКА]
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
HARD = pd.Timestamp("2025-01-01")       # независимая проверка результата
GATE_END = pd.Timestamp("2022-01-01")   # гейт равенства с dev: даты < 2022
SHIFTS = (-2, -1, 0, 1)
SRC = "moex_discovery/data/final_v6/candles_10m/external/"


def assert_cut(values, what: str) -> None:
    mx = pd.to_datetime(pd.Series(values)).max()
    if pd.notna(mx) and mx >= HARD:
        raise ValueError(f"{what}: есть даты ≥ {HARD.date()} ({mx})")


MIN_BARS = 10    # день с < 10 баров до 18:50 в источнике «разреженный» (в 2017–2021 их 12 %, значение dev там не из этого файла)


def build_rvi(src: Path) -> tuple[pd.DataFrame, pd.Series]:
    """RVI 18:50 по дням до 2025 и число баров дня до 18:50: чтение с фильтром begin < END, агрегат — daily_1850.aggregate."""
    from fastbt import vneshnie_1850 as v
    bars = v.read_bars(src, END)
    s = v.close_1850(bars, "RVI")
    mod = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    nb = bars[mod < v.CUT_MIN].groupby(bars["timestamp"].dt.normalize()).size()
    out = pd.DataFrame({"date": pd.to_datetime(s.index), "close_RVI": s.to_numpy(dtype=float)})
    out = out.dropna().reset_index(drop=True)
    assert_cut(out["date"], "RVI")
    return out, nb.reindex(out["date"]).fillna(0)


def gate_dev(new: pd.DataFrame, nb: pd.Series, dev: pd.DataFrame, tol: float = 1e-12) -> dict:
    """Даты < 2022 из dev: на полных днях (баров ≥ MIN_BARS) значения равны (|Δ| ≤ tol); разреженные — считаем отдельно."""
    b = dev.assign(date=pd.to_datetime(dev["date"]))
    b = b[b["date"] < GATE_END].set_index("date")["close_RVI"].astype(float)
    a = new[new["date"] < GATE_END].set_index("date")["close_RVI"]
    full = nb.to_numpy() >= MIN_BARS
    full = pd.Series(full, index=pd.DatetimeIndex(new["date"])).reindex(a.index)
    j = pd.concat([a.rename("n"), b.rename("d"), full.rename("full")], axis=1, join="inner")
    miss = b.index.difference(a.index[full.to_numpy()])
    diff = (j["n"] - j["d"]).abs()
    bad = diff[j["full"].astype(bool) & ~(diff <= tol)]
    if len(bad):
        raise AssertionError(f"RVI < 2022 расходится с dev на {len(bad)} полных днях, max|Δ| = {bad.max()}")
    nf = int(j["full"].sum())
    sp = ~j["full"].astype(bool)
    return dict(full_checked=nf, sparse_in_dev=int(sp.sum()), sparse_differ=int((diff[sp] > tol).sum()),
                dev_days_without_full_day=len(miss))


def sdvig(rvi: pd.DataFrame, imoex: pd.Series, rv: pd.Series) -> pd.DataFrame:
    """Д1, 2022–2024: (а) ΔRVI_d против доходности IMOEX 18:50 (связанная величина, синхронная связь);
    (б) ln RVI_d против ln RV рынка (среднее по тикерам) в день d+s — уровни, RVI смотрит вперёд."""
    r = rvi.set_index("date")["close_RVI"]
    sel = lambda s: s[(s.index >= GATE_END) & (s.index < END)]
    dr, ret = sel(r.diff()), sel(np.log(imoex).diff())
    lr, lrv = sel(np.log(r)), sel(np.log(rv.clip(lower=1e-12)))
    rows = []
    for s in SHIFTS:
        rows.append(("dRVI~ret_IMOEX", s, float(dr.corr(ret.shift(-s).reindex(dr.index)))))
        rows.append(("lnRVI~lnRV_rynka", s, float(lr.corr(lrv.shift(-s).reindex(lr.index)))))
    return pd.DataFrame(rows, columns=["cel", "sdvig", "corr"])


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    from fastbt import daily_1850, vneshnie_1850 as v
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=daily_1850.default_main())
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "regen" / "test1_2022_2024")
    a = ap.parse_args(argv)
    new, nb = build_rvi(a.main / SRC / "RVI.parquet")
    dev = pd.read_parquet(a.main / "data" / "fastbt" / "s9" / "rvi_1850.parquet")
    print("гейт dev (< 2022, ≤ 1e-12):", gate_dev(new, nb, dev), "; последняя дата:", new["date"].max().date())
    z2 = nb[(nb.index >= GATE_END)]
    print(f"Z2 2022–2024: дней {len(z2)}, разреженных (< {MIN_BARS} баров) {int((z2 < MIN_BARS).sum())}")
    im = v.close_1850(v.read_bars(a.main / SRC / "IMOEX.parquet", END), "IMOEX")
    rv = pq.read_table(a.main / "data/iss_check/rv_1850_noquar.parquet", filters=[("date", "<", END.to_pydatetime())]).to_pandas()
    rv = rv.assign(date=pd.to_datetime(rv["date"])).groupby("date")["rv"].mean()
    sd = sdvig(new, im, rv)
    print("Д1:\n" + sd.to_string(index=False))
    p = a.out / "s9" / "rvi_1850.parquet"
    if p.exists():
        print("существует, не перезаписываю:", p)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(p, index=False)
        sd.to_csv(a.out / "sdvig_d1_rvi.csv", index=False)
    (a.out / "SHA256SUMS_rvi").write_text(f"{_sha(p)}  s9/rvi_1850.parquet\n")
    exp = f"FASTBT_Z2_RVI={p.resolve()}"
    (a.out / "export_rvi.txt").write_text(exp + "\n")
    print(exp, "sha256", _sha(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
