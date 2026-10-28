"""σ̂ главы 4 для зоны Z2 «тест-1» (2021 прогрев + 2022–2024) — каталог в формате data/regen/ch4_dev/ для листа R-ж
(scripts/revers/r1_zh.py --period test1 --sigma-dir <выход>; имена — cd_s3_sigma.FILES и sigma_naive_dev, плюс h5 как в ch4_dev).

Источники (от корня основной копии --main):
  гибрид, HAR-J (б), h1 и h5:  ≤ 2021 — строки data/regen/ch4_dev/sigma_*_dev.parquet (их построил scripts/analysis/ch4_dev_sigma.py из
                               test_locked/R2/final ≤ 2021; test_locked этот скрипт не читает);
                               2022–2024 — data/regen/test1_2022_2024/regen/R2/final/{final,etalon}_d0_s2emb_h{1,5}__t1.parquet
                               (put_z2 от test_locked/R2/final/*_test.parquet); σ = H.sigma_ch3 (√ряда, float32, сортировка date, ticker).
  наивная √RV:                 H.naive_from_rv (√ среднего RV дней d−4…d) из data/iss_check/rv_1850_noquar.parquet (дата < 2025;
                               rv совпадает с features_pit_v3.rv на всех строках < 2022: 32 426 из 32 426, макс. разность 0) — вся история
                               пересчитывается, так что ≤ 2021 сверяется с ch4_dev по-настоящему.
Гейты (отказ — код 2, файлы не пишутся): (1) ≤ 2021 побайтно = ch4_dev; (2) нет дат ≥ 2025; (3) нет дыр в датах от последней строки
≤ 2021 до конца ряда по data/calendar/torgovye_dni_moex.csv; (4) Д1: Спирмен ln σ̂² (2022–2024) с ln RV_{d+k}, k = −2…+1, медиана
по тикерам, максимум на k = 0 (допуск ε = 0,01). Выход: <--out>/*.parquet + SHA256SUMS.
Запуск: python3 scripts/dannye/z2_sigma_ch4.py --main /Users/nilysenok/Desktop/Старое/PhD_Vol_trading [--out ПАПКА]"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "analysis")]
import h2_d0_ch4 as H  # noqa: E402  только sigma_ch3, naive_from_rv (тот же код, что ch4_dev_sigma.py)

END = pd.Timestamp("2025-01-01")        # исключительно
DEV_END = pd.Timestamp("2022-01-01")    # граница ch4_dev / Z2
LAST = pd.Timestamp("2024-12-31")
EPS = 0.01
SHIFTS = (-2, -1, 0, 1)
Z2R2 = "data/regen/test1_2022_2024/regen/R2/final"
# имя файла ch4_dev: (часть Z2, столбец ряда, столбец выхода)
MODELS = {f"sigma_{m}_h{h}_dev": (f"{p}_d0_s2emb_h{h}__t1", c, f"sigma_pred_h{h}")
          for h in (1, 5) for m, p, c in (("hyb", "final", "rv_pred"), ("harj_b", "etalon", "har_corr"))}
NAIVE = "sigma_naive_dev"


def _max_date(path: Path):
    pf = pq.ParquetFile(path)
    i, mx = pf.schema_arrow.names.index("date"), None
    for g in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(g).column(i).statistics
        if st is None or not st.has_min_max:
            return None
        mx = st.max if mx is None else max(mx, st.max)
    return mx


def read_part(path: Path, cols: list[str], strict: bool = True) -> pd.DataFrame:
    """Чтение с фильтром date < 2025 при чтении; strict — отказ до чтения, если в файле есть даты ≥ 2025 (по статистике файла)."""
    if strict:
        mx = _max_date(path)
        if mx is not None and pd.Timestamp(mx) >= END:
            raise SystemExit(f"СТОП: {path.name}: даты ≥ 2025 ({mx})")
    d = pq.read_table(path, columns=cols, filters=[("date", "<", END.to_pydatetime())]).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    return d


def sklejka(dev: pd.DataFrame, z2: pd.DataFrame, oc: str) -> pd.DataFrame:
    if len(dev) and dev["date"].max() >= DEV_END:
        raise SystemExit("СТОП: ch4_dev содержит даты ≥ 2022")
    if len(z2) and z2["date"].min() < DEV_END:
        raise SystemExit("СТОП: часть Z2 заходит в ≤ 2021 (перекрытие с dev)")
    out = pd.concat([dev[["date", "ticker", oc]], z2[["date", "ticker", oc]]], ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_series(main: Path) -> dict[str, pd.DataFrame]:
    dev_dir, z2_dir, ser = main / "data/regen/ch4_dev", main / Z2R2, {}
    for name, (part, col, oc) in MODELS.items():
        dev = read_part(dev_dir / f"{name}.parquet", ["date", "ticker", oc])
        raw = read_part(z2_dir / f"{part}.parquet", ["date", "ticker", col])
        ser[name] = sklejka(dev, H.sigma_ch3(raw, col, oc), oc)
    rv = rv_frame(main)
    ser[NAIVE] = H.naive_from_rv(rv, cut=LAST)
    return ser


def rv_frame(main: Path) -> pd.DataFrame:
    rv = read_part(main / "data/iss_check/rv_1850_noquar.parquet", ["date", "ticker", "rv"], strict=False)
    return rv.sort_values(["ticker", "date"]).reset_index(drop=True)


# ======================= гейты =======================

def gate_dev(new: pd.DataFrame, dev: pd.DataFrame, oc: str) -> float:
    """Гейт 1: строки < 2022 совпадают с ch4_dev по датам, тикерам и значениям; -> макс. |разность| (0.0 — побайтно)."""
    a = new[new["date"] < DEV_END].reset_index(drop=True)
    b = dev[dev["date"] < DEV_END].sort_values(["date", "ticker"]).reset_index(drop=True)
    if len(a) != len(b) or not (a["date"].equals(b["date"]) and a["ticker"].equals(b["ticker"])):
        raise SystemExit(f"СТОП гейт 1: {oc}: строки ≤ 2021 не совпадают с ch4_dev ({len(a)} vs {len(b)})")
    d = float(np.max(np.abs(a[oc].astype("float64").values - b[oc].astype("float64").values))) if len(a) else 0.0
    if not d <= 1e-7:
        raise SystemExit(f"СТОП гейт 1: {oc}: макс. разность с ch4_dev {d:.3e} > 1e-7")
    return d


def gate_dates(df: pd.DataFrame, name: str) -> None:
    if df["date"].max() >= END:
        raise SystemExit(f"СТОП гейт 2: {name}: есть даты ≥ 2025")


def gate_calendar(df: pd.DataFrame, cal: pd.Series, bars: set, name: str) -> tuple[int, list[str]]:
    """Гейт 3: все торговые дни календаря от последней даты < 2022 до конца ряда присутствуют, кроме дней, когда торгов не было
    ни у одного тикера дневного ряда Z2 (календарь и ряд расходятся — в отчёт). -> (число дней, пропущенные из-за ряда дней)."""
    have = set(df["date"].unique())
    lo, hi = df.loc[df["date"] < DEV_END, "date"].max(), df["date"].max()
    need = cal[(cal >= lo) & (cal <= hi)]
    excl = [str(d.date()) for d in need if d not in bars]
    miss = [d for d in need if d in bars and d not in have]
    if miss:
        raise SystemExit(f"СТОП гейт 3: {name}: нет торговых дней {[str(d.date()) for d in miss[:5]]} (всего {len(miss)})")
    return len(need) - len(excl), excl


def sdvig(sig: pd.DataFrame, oc: str, rv: pd.DataFrame, diff: bool = False) -> dict:
    """Гейт 4 (Д1): Спирмен ln σ̂²_d с ln RV_{d+k} по тикерам, медиана; k в торговых днях тикера; σ̂ за 2022–2024.
    diff — по приращениям (оконный ряд — наивная, как в ch4_dev_sigma: по уровню окно отстаёт на день)."""
    res = {k: [] for k in SHIFTS}
    s = sig[(sig["date"] >= DEV_END) & (sig["date"] <= LAST)]
    for tk, g in s.groupby("ticker"):
        r = rv[rv["ticker"] == tk].sort_values("date").reset_index(drop=True)
        pos = pd.Series(r.index.values, index=r["date"]).reindex(g["date"]).values
        m = ~np.isnan(pos)
        ly, lr = np.log(np.clip(g[oc].values.astype("float64") ** 2, 1e-12, None)), np.log(np.clip(r["rv"].values, 1e-12, None))
        if diff:
            ly, lr = np.diff(ly, prepend=np.nan), np.diff(lr, prepend=np.nan)
        pos, ly = pos[m].astype(int), ly[m]
        for k in SHIFTS:
            j = pos + k
            ok = (j >= 0) & (j < len(r))
            if ok.sum() > 10:
                res[k].append(pd.Series(ly[ok]).corr(pd.Series(lr[j[ok]]), method="spearman"))
    med = {k: float(np.nanmedian(v)) if v else float("nan") for k, v in res.items()}
    am = max(med, key=lambda k: -np.inf if np.isnan(med[k]) else med[k])
    ok = all(med[k] - med[0] <= EPS for k in SHIFTS if k != 0)
    return dict(corr=med, argmax=am, ok=ok)


# ======================= запуск =======================

def run(main: Path, out: Path | None) -> dict:
    ser = build_series(main)
    cal = pd.to_datetime(pd.read_csv(main / "data/calendar/torgovye_dni_moex.csv")["date"])
    rv, rep = rv_frame(main), {}
    bars = set(read_part(main / "data/regen/test1_2022_2024/daily/ohlcv_daily_1850.parquet", ["date"])["date"].unique())
    for name, df in ser.items():
        oc = df.columns[-1]
        dev = read_part(main / f"data/regen/ch4_dev/{name}.parquet", ["date", "ticker", oc])
        gate_dates(df, name)
        nd, excl = gate_calendar(df, cal, bars, name)
        rep[name] = dict(gate1_maxdiff=gate_dev(df, dev, oc), gate3_dney=nd, bez_torgov=excl,
                         rows=len(df), last=str(df["date"].max().date()), sdvig=sdvig(df, oc, rv, diff=name == NAIVE))
    bad = [n for n, r in rep.items() if not r["sdvig"]["ok"]]
    if bad:
        raise SystemExit(f"СТОП гейт 4 (сдвиг): {bad}: {[rep[n]['sdvig'] for n in bad]}")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for name, df in ser.items():
            df.to_parquet(out / f"{name}.parquet", index=False)
        sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(out.glob("*.parquet")))
        (out / "SHA256SUMS").write_text(sums)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=H.MAIN)
    ap.add_argument("--out", type=Path, default=ROOT / "data/regen/test1_2022_2024/ch4_sigma")
    a = ap.parse_args(argv)
    for n, r in run(a.main, a.out).items():
        c = r["sdvig"]["corr"]
        print(f"{n}: строк {r['rows']}, последняя дата {r['last']}, г1 макс.разн. {r['gate1_maxdiff']:.1e}, г3 дней {r['gate3_dney']} (в календаре без торгов: {r['bez_torgov']}), "
              f"Д1 argmax {r['sdvig']['argmax']:+d} " + " ".join(f"k{k:+d}={v:.4f}" for k, v in c.items()))
    print(f"выход: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
