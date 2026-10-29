"""Реверс раунд 1, лист R-е: фазы рынка по месяцам и результат A / A_strogaya по фазам (Ж-117, Ж-099 пп. 2–3).
Справочно, без выбора: правила фаз зашиты в `fastbt/fazy.py` (#478), ничего не подбирается.

Запуск (train4, воспроизведение): FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_e.py --period train4
Запуск (test1, только после снятия замка Z2 и слияния R-P); --yakor A_zashch (умолчание, A (s9)) | A_fix (фиксированная точка, якорь кода):
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
from fastbt import fazy, metriki_e as me, zona  # noqa: E402,F401
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imoex_test1 as IT  # noqa: E402
import r1_a_zashch as Z  # noqa: E402
import shapka_lib as SL  # noqa: E402
from imoex_test1 import dnevka_iz_10min, dnevnye_index_d  # noqa: E402,F401

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


def zagruzit_imoex(path: Path, s: pd.Timestamp, e: pd.Timestamp, path_d: Path | None = None) -> pd.Series:
    """Дневные закрытия IMOEX [s, e): ISS-CSV (TRADEDATE, CLOSE) или parquet Z2 (общий код — imoex_test1.py)."""
    if path_d is not None or Path(path).suffix == ".parquet":
        return IT.zagruzit(path, s, e, path_d)
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


def doli_po_godam(faza: pd.Series, gody, **kw) -> pd.DataFrame:
    return fazy.doli_po_godam(pd.DataFrame({"faza": faza}), gody, **kw)


def po_fazam(imya: str, x: np.ndarray, baza: np.ndarray, faza: pd.Series, plata: float, daty=None) -> list:
    """Строки «ряд × фаза»: дни, средний дневной excess, Sharpe, mSR (excess за вычетом платы за шорт); «все» и «nan».
    daty (datetime64[D]) задана — в конце недельные метрики *_ned по тем же дням (me.stroka_ned; при < MIN_DNEY дней — NaN)."""
    xx = np.asarray(x, float) - plata * np.asarray(baza, float)
    f = faza.to_numpy(object)
    rows = []
    for k in ["все", *fazy.FAZY, "nan"]:
        m = np.ones(len(xx), bool) if k == "все" else (pd.isna(f) if k == "nan" else f == k)
        v = xx[m]
        ok = len(v) >= MIN_DNEY
        r = dict(ryad=imya, plata_zaem=plata, faza=k, dni=int(m.sum()),
                 sredn_den=float(v.mean()) if len(v) else np.nan,
                 sharpe=float(me.sr_dnevnoy(v) * np.sqrt(252)) if ok else np.nan,
                 msr=float(me.msr(v)) if ok else np.nan, dohodnost=float(v.sum()))
        if daty is not None:
            nd = me.stroka_ned(v, np.asarray(daty).astype("datetime64[D]")[m]) if ok else {}
            r.update({c: nd.get(c, np.nan) for c in NED_COLS})
        rows.append(r)
    return rows


NED_COLS = ("sharpe_ned", "sortino_ned", "msr_ned", "skew_ned", "kurt_ned", "n_nedel", "msr_dn_opredelen")


def ryady(period: str, yakor: str = "A_zashch") -> dict:
    """{yakor: (dates, x, baza), 'A_strogaya': ...}. A_zashch — scripts/revers/r1_a_zashch.py (кэш s9, как r1_a_cd); A_fix — фиксированная
    точка этапа 1 (только якорь кода); A_strogaya — fastbt.a_strogaya. Даты рядов могут различаться: у A_zashch (s9) есть последняя строка
    2024-12-30 (выход по close, lr = 0, как в a_cd: 744 дня), у рядов этапа 1 — 743; фаза считается по датам каждого ряда."""
    from fastbt import a_strogaya as AS
    AS.PER = period
    res = {"A_fix": {}, "strogaya": {}}
    for sid in AS.SIDS:
        anc, st = AS.evaluatory(sid)
        if yakor == "A_fix":
            res["A_fix"][sid] = AS.schitat(anc, False)
        res["strogaya"][sid] = AS.schitat(st, True)
        print(sid, "готово", flush=True)
    out = {}
    if yakor == "A_zashch":
        p, _ = Z.ryad_i_pozicii(period)
        out[yakor] = (pd.to_datetime(p.index), p["excess"].to_numpy(float), p["baza"].to_numpy(float))
    else:
        a = AS.portfel_A(res["A_fix"])
        out[yakor] = (pd.to_datetime(a["dates"]), np.asarray(a["x"], float), np.asarray(a["baza"], float))
    a = AS.portfel_A(res["strogaya"])
    out["A_strogaya"] = (pd.to_datetime(a["dates"]), np.asarray(a["x"], float), np.asarray(a["baza"], float))
    return out


def tablicy(period: str, imoex: pd.Series, serii: dict) -> dict:
    s, e, gody, _ = PERIODY[period]
    faza_k = fazy_dnej(imoex, imoex.index)       # календарь IMOEX: как в таблице долей #478
    rows = []
    for imya, (d, x, b) in serii.items():
        faza = fazy_dnej(imoex, d)
        faza.index = d
        for plata in (PLATA, 0.0):
            rows += po_fazam(imya, x, b, faza, plata, d.to_numpy().astype("datetime64[D]"))
    return {"fazy_mesyacy": fazy_po_mesyacam(faza_k, gody), "fazy_doli": doli_po_godam(faza_k, gody, **({"status": SHAPKA[2:]} if period == "test1" else {})),
            "po_fazam": pd.DataFrame(rows)}


def zapis(period: str, tab: dict, out: Path | None = None) -> None:
    out = out or ROOT / "docs" / "tables" / "revers_r1"
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tab.items():
        with open(out / f"e_{k}_{period}.csv", "w", encoding="utf-8") as fh:
            fh.write(f"{SHAPKA} — {PERIODY[period][3]}\n" + SL.blok(period) + zona.shapka_dividendy())
            df.to_csv(fh, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODY), required=True)
    ap.add_argument("--imoex", type=Path, default=None, help="ISS-CSV IMOEX; для train4 — data/iss_mix_2015_2019/IMOEX.csv")
    ap.add_argument("--imoex-d", type=Path, default=None, help="parquet дневных IMOEX (date, close): с 2022 вместо 10-минуток (без дыр)")
    ap.add_argument("--out", type=Path, default=None, help="каталог таблиц (по умолчанию docs/tables/revers_r1)")
    ap.add_argument("--yakor", choices=("A_zashch", "A_fix"), default="A_zashch",
                    help="A_zashch = A (s9), ежегодный top-10 ансамбль — содержательный счёт; A_fix — фиксированная точка этапа 1, только якорь кода")
    a = ap.parse_args()
    s, e, _, _ = PERIODY[a.period]
    path = a.imoex or (Path(os.environ.get("FASTBT_MAIN", ROOT)) / "data/iss_mix_2015_2019/IMOEX.csv" if a.period == "train4" else None)
    if path is None:
        sys.exit("test1: укажите --imoex (замок Z2 должен быть снят)")
    tab = tablicy(a.period, zagruzit_imoex(path, s, e, a.imoex_d), ryady(a.period, a.yakor))
    zapis(a.period, tab, a.out)
    print(tab["fazy_doli"].pivot(index="faza", columns="god", values="dolya").to_string())
    print(tab["po_fazam"].round(3).to_string())


if __name__ == "__main__":
    main()
