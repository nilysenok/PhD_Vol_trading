"""Реверс раунд 1, лист R-е: фазы рынка по месяцам и результат A / A_strogaya по фазам (Ж-117, Ж-099 пп. 2–3).
Справочно, без выбора: правила фаз зашиты в `fastbt/fazy.py` (#478), ничего не подбирается.

Запуск (train4, воспроизведение): FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_e.py --period train4
Запуск (test1, только после снятия замка Z2 и слияния R-P):
  python3 scripts/revers/r1_e.py --period test1 --imoex <ISS-CSV IMOEX (TRADEDATE, CLOSE) или parquet 10-минуток Z2 (begin, close)>
Пишет docs/tables/revers_r1/e_{fazy_mesyacy,fazy_doli,po_fazam}_<period>.csv и docs/otchety/revers_r1/e_<period>.md.
Фаза дня d — по закрытиям IMOEX до d−1 (`fastbt.fazy`, лаг 1). Метрики — `fastbt/metriki_e.py` (Д7): Sharpe ddof=1 √252 и mSR.
В строке «по фазам» метрики считаются только при ≥ MIN_DNEY дней (иначе NaN): это правило отчёта, а не отбор.
Данные ≥ 2025 не читаются (фильтр на чтении); train4 — < 2020."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import fazy, metriki_e as me, zona  # noqa: E402

SHAPKA = "# реверс раунд 1, 2022–2024, справочно, без выбора"
MIN_DNEY = 20
PLATA = 0.05
# период: (начало чтения IMOEX с прогревом фаз, конец (не включая), годы, статус)
PERIODY = {
    "train4": (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-01"), (2016, 2017, 2018, 2019),
               "воспроизведение на train4 (2016–2019), после аудита"),
    "test1": (pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01"), (2022, 2023, 2024),
              "реверс раунд 1, 2022–2024, справочно, без выбора"),
}


POLNYJ_DEN_MIN = 18 * 60 + 40      # последний бар полного дня начинается ≥ 18:40; раньше — дыра в 10-минутках


def dnevka_iz_10min(bars: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series:
    """Дневное закрытие IMOEX [s, e) из 10-минуток (begin, close): close ПОСЛЕДНЕГО бара дня, как ISS-дневка.
    День, где последний бар раньше 18:40 (обрыв ленты, ~2 дня в месяц), не берётся: подставлять нечем (Д4)."""
    zona.check(e - pd.Timedelta(days=1), "IMOEX 10-мин")
    t = pd.to_datetime(bars["begin"])
    if t.max() >= zona.HARD:
        raise PermissionError(f"IMOEX 10-мин: даты ≥ {zona.HARD.date()}")
    b = pd.DataFrame({"t": t, "close": bars["close"].to_numpy()}).sort_values("t")
    b = b[(b.t >= s) & (b.t < e)]
    last = b.groupby(b.t.dt.normalize()).tail(1)
    last = last[last.t.dt.hour * 60 + last.t.dt.minute >= POLNYJ_DEN_MIN]
    return pd.Series(last["close"].to_numpy(float), index=pd.DatetimeIndex(last.t.dt.normalize(), name="TRADEDATE"))


STYK = pd.Timestamp("2022-01-01")  # до стыка — дневка из 10-минуток, с него — дневные закрытия index_d


def dnevnye_index_d(path: Path, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series:
    """Дневные close IMOEX [s, e) из index_d (date, close); фильтр date < min(e, HARD) — на чтении, ≥ 2025 не материализуется."""
    import pyarrow.parquet as pq
    zona.check(e - pd.Timedelta(days=1), "IMOEX index_d")
    cut = min(e, zona.HARD).date()
    d = pq.read_table(path, columns=["date", "close"], filters=[("date", "<", cut)]).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    if len(d) and d["date"].max() >= zona.HARD:
        raise PermissionError(f"IMOEX index_d: даты ≥ {zona.HARD.date()}")
    d = d[(d.date >= s) & (d.date < e)].drop_duplicates("date", keep="last")
    return pd.Series(d["close"].to_numpy(float), index=pd.DatetimeIndex(d["date"], name="TRADEDATE")).sort_index()


def zagruzit_imoex(path: Path, s: pd.Timestamp, e: pd.Timestamp, path_d: Path | None = None) -> pd.Series:
    """Дневные закрытия IMOEX [s, e): ISS-CSV (TRADEDATE, CLOSE) или parquet 10-минуток зоны Z2 (begin, close).
    С path_d: [s, STYK) — из 10-минуток (прогрев), [STYK, e) — из index_d (без дыр)."""
    if path_d is not None:
        return pd.concat([zagruzit_imoex(path, s, STYK), dnevnye_index_d(path_d, STYK, e)])
    if Path(path).suffix == ".parquet":
        import pyarrow.parquet as pq
        cut = min(e, zona.HARD)
        return dnevka_iz_10min(pq.read_table(path, columns=["begin", "close"], filters=[("begin", "<", str(cut.date()))]).to_pandas(), s, e)
    im = pd.read_csv(path, usecols=["TRADEDATE", "CLOSE"])
    im["TRADEDATE"] = pd.to_datetime(im["TRADEDATE"])
    im = im[(im.TRADEDATE >= s) & (im.TRADEDATE < e)].drop_duplicates("TRADEDATE", keep="last")
    return im.set_index("TRADEDATE")["CLOSE"].sort_index()


def fazy_dnej(imoex: pd.Series, daty) -> pd.Series:
    """faza на датах (NaN до прогрева); дата без IMOEX-закрытия получает NaN."""
    return fazy.faza_rynka(daty, {"imoex": imoex}).faza


def fazy_po_mesyacam(faza: pd.Series, gody) -> pd.DataFrame:
    """Месяц × фаза: число дней, преобладающая фаза (при равенстве — первая по списку FAZY)."""
    f = faza[faza.index.year.isin(gody)]
    rows = []
    for m, x in f.groupby(f.index.to_period("M")):
        n = {k: int((x == k).sum()) for k in fazy.FAZY}
        opr = max(fazy.FAZY, key=lambda k: n[k]) if sum(n.values()) else "nan"
        rows.append({"mesyac": str(m), "dni": len(x), **n, "nan": int(x.isna().sum()), "preobladaet": opr})
    return pd.DataFrame(rows)


def doli_po_godam(faza: pd.Series, gody) -> pd.DataFrame:
    return fazy.doli_po_godam(pd.DataFrame({"faza": faza}), gody)


def po_fazam(imya: str, x: np.ndarray, baza: np.ndarray, faza: pd.Series, plata: float) -> list:
    """Строки «ряд × фаза»: дни, средний дневной excess, Sharpe, mSR (excess за вычетом платы за шорт); «все» и «nan»."""
    xx = np.asarray(x, float) - plata * np.asarray(baza, float)
    f = faza.to_numpy(object)
    rows = []
    for k in ["все", *fazy.FAZY, "nan"]:
        m = np.ones(len(xx), bool) if k == "все" else (pd.isna(f) if k == "nan" else f == k)
        v = xx[m]
        ok = len(v) >= MIN_DNEY
        rows.append(dict(ryad=imya, plata_zaem=plata, faza=k, dni=int(m.sum()),
                         sredn_den=float(v.mean()) if len(v) else np.nan,
                         sharpe=float(me.sr_dnevnoy(v) * np.sqrt(252)) if ok else np.nan,
                         msr=float(me.msr(v)) if ok else np.nan, dohodnost=float(v.sum())))
    return rows


def ryady(period: str) -> dict:
    """{'A': (dates, x, baza), 'A_strogaya': ...} через fastbt.a_strogaya; period test1 — после слияния R-P."""
    from fastbt import a_strogaya as AS
    AS.PER = period
    res = {"yakor": {}, "strogaya": {}}
    for sid in AS.SIDS:
        anc, st = AS.evaluatory(sid)
        res["yakor"][sid], res["strogaya"][sid] = AS.schitat(anc, False), AS.schitat(st, True)
        print(sid, "готово", flush=True)
    out = {}
    for imya, v in (("A", "yakor"), ("A_strogaya", "strogaya")):
        a = AS.portfel_A(res[v])
        out[imya] = (pd.to_datetime(a["dates"]), np.asarray(a["x"], float), np.asarray(a["baza"], float))
    return out


def tablicy(period: str, imoex: pd.Series, serii: dict) -> dict:
    s, e, gody, _ = PERIODY[period]
    daty = serii["A"][0]
    faza = fazy_dnej(imoex, daty)
    faza.index = daty
    faza_k = fazy_dnej(imoex, imoex.index)       # календарь IMOEX: как в таблице долей #478
    rows = []
    for imya, (d, x, b) in serii.items():
        for plata in (PLATA, 0.0):
            rows += po_fazam(imya, x, b, faza, plata)
    return {"fazy_mesyacy": fazy_po_mesyacam(faza_k, gody), "fazy_doli": doli_po_godam(faza_k, gody),
            "po_fazam": pd.DataFrame(rows)}


def zapis(period: str, tab: dict) -> None:
    out = ROOT / "docs" / "tables" / "revers_r1"
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tab.items():
        with open(out / f"e_{k}_{period}.csv", "w", encoding="utf-8") as fh:
            fh.write(f"{SHAPKA} — {PERIODY[period][3]}\n" + zona.shapka_dividendy())
            df.to_csv(fh, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODY), required=True)
    ap.add_argument("--imoex", type=Path, default=None, help="ISS-CSV IMOEX; для train4 — data/iss_mix_2015_2019/IMOEX.csv")
    ap.add_argument("--imoex-d", type=Path, default=None, help="parquet дневных IMOEX (date, close): с 2022 вместо 10-минуток (без дыр)")
    a = ap.parse_args()
    s, e, _, _ = PERIODY[a.period]
    path = a.imoex or (Path(os.environ.get("FASTBT_MAIN", ROOT)) / "data/iss_mix_2015_2019/IMOEX.csv" if a.period == "train4" else None)
    if path is None:
        sys.exit("test1: укажите --imoex (замок Z2 должен быть снят)")
    tab = tablicy(a.period, zagruzit_imoex(path, s, e, a.imoex_d), ryady(a.period))
    zapis(a.period, tab)
    print(tab["fazy_doli"].pivot(index="faza", columns="god", values="dolya").to_string())
    print(tab["po_fazam"].round(3).to_string())


if __name__ == "__main__":
    main()
