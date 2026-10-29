"""Лист R-в раунда 1 реверса (Ж-117): овернайт против внутридневной части по категориям, период train4 или test1.
Запуск: FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_v.py --period {train4,test1} [--imoex PATH] [--out CSV].
Расчёт — scripts/analysis/overnajt.py (#487) без правок: здесь только параметры периода. train4 воспроизводит
docs/tables/overnajt/razlozhenie.csv (по умолчанию пишет в docs/tables/revers_r1/v_train4_proverka.csv и сверяет).
test1 = 2022-01-01…2024-12-31, прогрев 2021; ничего ≥ 2025 (read_bars режет по END = 2025-01-01). Нужна поддержка периода
«test1» в etap1/a_strogaya (лист R-P) и 10-минутки зоны Z2; до этого test1 останавливается с понятной ошибкой.
Справочно, без выбора."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import imoex_test1 as IT  # noqa: E402
import overnajt as O  # noqa: E402
import overnajt_lib as L  # noqa: E402
from fastbt import a_strogaya as AS, kross_seksiya as ks, msp, zona  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402

HEADER = "# реверс раунд 1, 2022–2024, справочно, без выбора"
IMOEX_TRAIN4 = None          # умолчание golaya_s7_s9.IMOEX
# период → (годы метрик, начало данных с прогревом, T0, END (исключается), последний день окна)
PERIODS = {"train4": ((2016, 2017, 2018, 2019), "2015-01-01", "2016-01-01", "2020-01-01", "2019-12-31"),
           "test1": ((2022, 2023, 2024), "2021-01-01", "2022-01-01", "2025-01-01", "2024-12-31")}
ETALON = ROOT / "docs" / "tables" / "overnajt" / "razlozhenie.csv"


def nastroit(period: str, imoex: str | None, imoex_d: str | None = None) -> str:
    """Подменяет константы периода в AS и golaya_s7_s9 (читаются при вызове); возвращает последний день окна."""
    if not imoex and period != "train4":
        raise SystemExit("test1: укажите --imoex (IMOEX 2021–2024 из данных проекта зоны Z2; ISS не выгружать)")
    import golaya_s7_s9 as G
    gody, start, t0, end, last = PERIODS[period]
    AS.PER, AS.GODY = period, gody
    G.START, G.T0, G.END = pd.Timestamp(start), pd.Timestamp(t0), pd.Timestamp(end)
    if imoex and (imoex_d or Path(imoex).suffix == ".parquet"):     # Z2: 10-минутки (+ index_d с 2022) → дневной OPEN/CLOSE
        G.IMOEX = IT.dnevnoj_ryad(Path(imoex), pd.Timestamp(start), pd.Timestamp(end), Path(imoex_d) if imoex_d else None)
    elif imoex:
        G.IMOEX = Path(imoex)
    return last


def init_vorker(period: str, imoex: str | None, imoex_d: str | None = None) -> None:
    """Initializer пула: при spawn воркер импортирует модули заново (AS.PER = умолчание), поэтому период ставим в нём."""
    nastroit(period, imoex, imoex_d)
    assert AS.PER == period, f"воркер видит период {AS.PER}, нужен {period}"


def period_vorkera() -> str:
    return AS.PER


def sozdat_pul(period: str, imoex: str | None, n: int = 2, ctx=None, imoex_d: str | None = None):
    import multiprocessing
    return (ctx or multiprocessing).Pool(n, initializer=init_vorker, initargs=(period, imoex, imoex_d))


def kross(last: str) -> dict:
    """Книга S7–S9 — как overnajt.kross, но окно до last (там литерал 2019-12-31)."""
    import golaya_s7_s9 as G
    D = G.zagruzit()
    dates = pd.DatetimeIndex(D["c"].index)
    metki = ks.sektor_na_d(G.karta(), dates, TICKERS)
    sig, sk, stavka = G.sigmy(D["c"]).to_numpy(), G.skory(D, metki), msp.rate_on(dates).to_numpy()
    i0 = int(np.searchsorted(dates.values, np.datetime64(G.T0))) - 1
    win = np.asarray((dates >= G.T0) & (dates <= last))
    C, out = O.syroi_close(dates).to_numpy(), {}
    for sid, s in sk.items():
        tau, n = ks.TAU[sid]
        ok = D["o"].notna().shift(-ks.LAG, fill_value=False).to_numpy()
        W, P, Dv = ks.vesa(s.to_numpy(), sig, metki, ok), D["o"].ffill().to_numpy(), D["d"].to_numpy()
        o = ks.kniga(W, P, Dv, dates.values, stavka, tau, n, i0)
        I, N = L.kniga_razlozhenie(W, P, C, Dv, tau, n, i0, ks.LAG)
        ecc = float(np.abs(I + N - o["pnl"]).max())
        assert ecc < 1e-12, f"{sid}: части ≠ pnl книги ({ecc})"
        out[sid] = {k: v[win] for k, v in dict(I=I, N=N, x=o["excess"], d=o["gross"] > 0).items()}
        print(sid, "кросс-секция готово; макс. |части − pnl| =", f"{ecc:.1e}", flush=True)
    return out


def sverka(df: pd.DataFrame, atol: float = 1e-9) -> bool:
    """train4: совпадение с таблицей-образцом (#487): те же строки и колонки, числа в допуске atol."""
    et = pd.read_csv(ETALON)
    if list(et.columns) != list(df.columns) or len(et) != len(df):
        return False
    num = et.select_dtypes("number").columns
    return bool(et.drop(columns=num).equals(df.drop(columns=num)) and np.allclose(et[num], df[num], atol=atol, equal_nan=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODS), required=True)
    ap.add_argument("--imoex", help="csv ISS (train4) или parquet 10-минуток Z2 (begin, open, close)")
    ap.add_argument("--imoex-d", help="parquet дневных IMOEX (date, open, close): с 2022 вместо 10-минуток (без дыр)")
    ap.add_argument("--out")
    a = ap.parse_args()
    last = nastroit(a.period, a.imoex, a.imoex_d)
    R = {AS.SIDS[0]: O.schitat_sk(AS.SIDS[0])}       # первый — один: общий кэш 5-мин баров (гонка tmp → parquet)
    with sozdat_pul(a.period, a.imoex, imoex_d=a.imoex_d) as p:
        R.update(zip(AS.SIDS[1:], p.map(O.schitat_sk, AS.SIDS[1:])))
    df = pd.DataFrame(O.stroki_a(R) + O.stroki_kross(kross(last)))
    out = Path(a.out) if a.out else ROOT / "docs" / "tables" / "revers_r1" / ("v_overnajt.csv" if a.period == "test1" else "v_train4_proverka.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        if a.period == "test1":
            f.write(HEADER + "\n" + zona.shapka_dividendy())
        df.to_csv(f, index=False)
    print(df.round(3).to_string())
    if a.period == "train4":
        print("сверка с образцом #487:", "СОВПАЛО" if sverka(df) else "РАСХОДИТСЯ")


if __name__ == "__main__":
    main()
