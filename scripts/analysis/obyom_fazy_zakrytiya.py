"""Оборот фазы 18:45–18:50 (после аукциона закрытия) против типичной позиции (Ж-085 п. 3).

Данные: 1-минутки (value, ₽), только 2016–2019; чтение обрезается по дате до загрузки (fastbt.bars5m._read),
файлы лет ≥ 2020 не открываются. Оборот фазы дня = сумма value минут 18:45…18:49. День с доп. фазой
(цена закрытия в 18:49): по барам — нет минуты 18:45, но есть 18:49; тогда фаза = только минута 18:49.
Дни — объединение торговых дат 17 тикеров; у тикера нет баров в день — оборот 0 (сделок не было).
Вес позиции: razmer='ravnyy' → вес 1 на тикер, портфель делит на cnt (s9.port_rows: bincount/cnt), т. е. 1/cnt,
cnt = число тикеров с барами в день входа. Берётся медиана 1/cnt по сделкам журналов S1…S6 (2018–2019).
Запуск: python3 scripts/analysis/obyom_fazy_zakrytiya.py   (самопроверка — tests/test_obyom_fazy.py)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import bars5m  # noqa: E402

START, END = "2016-01-01", "2019-12-31"
CAPS = (100e6, 200e6, 500e6)
F, L, DOP = 18 * 60 + 45, 18 * 60 + 49, 18 * 60 + 49
OUT_CSV = ROOT / "docs" / "tables" / "inv" / "obyom_fazy_1845_1850.csv"
OUT_MD = ROOT / "docs" / "otchety" / "2026-10-28_obyom_fazy_zakrytiya.md"
DIAG = ROOT / "docs" / "tables" / "diag"


def chitat(ticker: str, src: Path) -> pd.DataFrame:
    m1 = bars5m._read(ticker, START, END, Path(src))
    if len(m1) and m1["begin"].max() >= pd.Timestamp(END) + pd.Timedelta(days=1):
        raise AssertionError(f"{ticker}: чтение за {END}")
    return m1


def po_dnyam(m1: pd.DataFrame) -> pd.DataFrame:
    """Индекс — дата (есть хотя бы одна минута); oborot (₽), dop (bool)."""
    if m1.empty:
        return pd.DataFrame({"oborot": [], "dop": []}, index=pd.DatetimeIndex([]))
    mod = m1["begin"].dt.hour * 60 + m1["begin"].dt.minute
    d = m1["begin"].dt.normalize()
    days = pd.DatetimeIndex(sorted(d.unique()))
    ph = (mod >= F) & (mod <= L)
    dop = ~(mod == F).groupby(d).any().reindex(days) & (mod == DOP).groupby(d).any().reindex(days)
    val_all = m1["value"].where(ph, 0.0).groupby(d).sum().reindex(days)
    val_49 = m1["value"].where(mod == DOP, 0.0).groupby(d).sum().reindex(days)
    return pd.DataFrame({"oborot": np.where(dop, val_49, val_all), "dop": dop.to_numpy()}, index=days)


def sobrat(src: Path, tickers) -> dict[str, pd.DataFrame]:
    raw = {t: po_dnyam(chitat(t, src)) for t in tickers}
    days = sorted(set().union(*[set(r.index) for r in raw.values()]))
    out = {}
    for t, r in raw.items():
        r = r.reindex(pd.DatetimeIndex(days))
        r["oborot"], r["dop"] = r["oborot"].fillna(0.0), r["dop"].astype("boolean").fillna(False).astype(bool)
        r["est"] = r.index.isin(raw[t].index)
        out[t] = r
    return out


def ves_pozitsii(dni: dict[str, pd.DataFrame], diag: Path) -> float:
    """Медиана 1/cnt по сделкам журналов S*_sdelki_posle370.csv (2018–2019)."""
    cnt = sum(r["est"].astype(int) for r in dni.values())
    ws = []
    for p in sorted(Path(diag).glob("S[0-9]_sdelki_posle370.csv")):
        d = pd.read_csv(p, comment="#", parse_dates=["data_vhoda"]).dropna(subset=["data_vhoda"])
        assert d["data_vhoda"].max() <= pd.Timestamp(END), f"{p.name}: сделки за {END}"
        ws += list(1.0 / cnt.reindex(d["data_vhoda"]).to_numpy())
    return float(np.median(ws))


def stroka(t: str, god: str, x: pd.DataFrame, w: float) -> dict:
    o = x["oborot"].to_numpy()
    row = dict(tiker=t, god=god, dney=len(o), mediana_rub=float(np.median(o)), p10_rub=float(np.percentile(o, 10)),
               dolya_nulevyh=float((o == 0).mean()), dolya_dop=float(x["dop"].mean()))
    for c in CAPS:
        row[f"poz_{int(c / 1e6)}"] = c * w
    row["flag_p10_lt_poz200"] = bool(row["p10_rub"] < row["poz_200"])
    return row


def tablitsa(dni: dict[str, pd.DataFrame], w: float) -> pd.DataFrame:
    rows = []
    for t, r in dni.items():
        for y in (2016, 2017, 2018, 2019):
            rows.append(stroka(t, str(y), r[r.index.year == y], w))
        rows.append(stroka(t, "2016-2019", r, w))
    return pd.DataFrame(rows)


def otchet(tab: pd.DataFrame, w: float) -> str:
    itog = tab[tab.god == "2016-2019"].set_index("tiker")
    fl = list(itog.index[itog.flag_p10_lt_poz200])
    fl_y = sorted(set(tab[tab.flag_p10_lt_poz200 & (tab.god != "2016-2019")].tiker) - set(fl))
    m = lambda v: f"{v / 1e6:.2f}"  # noqa: E731
    L_ = [f"Флаг (p10 оборота фазы < позиции при 200 млн): за 2016–2019 — {', '.join(fl) or 'нет'}; "
          f"только в отдельных годах — {', '.join(fl_y) or 'нет'}.", "",
          "# Оборот фазы 18:45–18:50 против позиции (Ж-085 п. 3)", "",
          "Статус: разработка 2016–2019, диагностика; не в реестре. Скрипт `scripts/analysis/obyom_fazy_zakrytiya.py`, "
          "таблица `docs/tables/inv/obyom_fazy_1845_1850.csv` (тикер × год).", "",
          "Определения: оборот = сумма `value` (₽) минутных баров 18:45…18:49 за день; день с доп. фазой (по барам: нет "
          "минуты 18:45, есть 18:49) — только минута 18:49. Дни — объединение торговых дат 17 тикеров, нет баров — 0. "
          f"Вес позиции = медиана 1/cnt по сделкам журналов S1…S6 (2018–2019) = {w * 100:.2f} % капитала "
          "(razmer = ravnyy: вес 1 на тикер, портфель делит на cnt). Минута 18:45 содержит и сам аукцион, "
          "поэтому оборот — верхняя оценка объёма, доступного после определения цены.", "",
          f"Позиция, млн ₽: 100 → {m(100e6 * w)}; 200 → {m(200e6 * w)}; 500 → {m(500e6 * w)}. "
          "Суммы ниже — млн ₽ за день, 2016–2019 целиком.", "",
          "| тикер | медиана | p10 | нулевых дней, % | доп. фаза, % | флаг |", "|---|---|---|---|---|---|"]
    for t, r in itog.iterrows():
        L_.append(f"| {t} | {m(r.mediana_rub)} | {m(r.p10_rub)} | {r.dolya_nulevyh * 100:.1f} | "
                  f"{r.dolya_dop * 100:.1f} | {'ФЛАГ' if r.flag_p10_lt_poz200 else ''} |")
    L_ += ["", "Медиана по тикерам (медиан дневного оборота), млн ₽: " + m(float(itog.mediana_rub.median())) + ".", "",
           "По годам (медиана / p10, млн ₽; * — флаг):", "", "| тикер | 2016 | 2017 | 2018 | 2019 |", "|---|---|---|---|---|"]
    for t in itog.index:
        c = [tab[(tab.tiker == t) & (tab.god == str(y))].iloc[0] for y in (2016, 2017, 2018, 2019)]
        L_.append(f"| {t} | " + " | ".join(f"{m(x.mediana_rub)} / {m(x.p10_rub)}{'*' if x.flag_p10_lt_poz200 else ''}"
                                          for x in c) + " |")
    L_ += ["", "Не проверено: доп. фаза определена по барам (эвристика), а не по журналу биржи; вес — по 1/cnt, где cnt "
           "считается по датам с барами, а не по строкам движка; сделки только 2018–2019."]
    return "\n".join(L_) + "\n"


def main() -> None:
    from fastbt.data import TICKERS
    src = Path(os.environ.get("FASTBT_BARS1M", bars5m.SRC))
    dni = sobrat(src, TICKERS)
    w = ves_pozitsii(dni, DIAG)
    tab = tablitsa(dni, w)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_CSV, index=False, float_format="%.6g")
    OUT_MD.write_text(otchet(tab, w), encoding="utf-8")
    print(f"вес {w:.5f}; строк {len(tab)}; флагов (итог) {int(tab[tab.god == '2016-2019'].flag_p10_lt_poz200.sum())}")


if __name__ == "__main__":
    main()
