"""Таблица дат режимов рынка для предрегистрации v3 (Ж-159 п. 7, Ж-161 п. 1; В-4 v3).

Режим дня d — ровно компонента `trend` фаз (`fazy._priznaki`, `_LAG = 1`): рост — close > MA200 и MA50 > MA200; падение —
close < MA200 и MA50 < MA200; иначе боковик; по закрытиям IMOEX ≤ d−1 (18:50, простые MA по торговым дням). Второго определения нет (Д7).
Закрытия — `rezhim_e1.imoex_daily` (10-минутки IMOEX до 18:50, граница чтения зоны); читается весь ряд до `--do`, срез [s, do] — на выходе.
Дни без MA200 пишутся строкой `net_ma`; их число — в шапке. Граница: `--do` < `period_v2.chtenie_konec("polnaya")`, иначе PermissionError.

Запуск: FASTBT_MAIN=<копия> python3 -m fastbt.tablica_rezhimov_v3 --do 2024-12-31 --out <csv> [--s 2014-08-26] [--imoex <parquet>]
(приложение 2025 после акта владельца — тот же код: --s 2025-01-01 --do 2025-12-31). Статус: разработка, до аудита.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from fastbt import daily_1850 as d18
from fastbt import fazy, rezhim_e1, zona

IMOEX_10M = "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet"   # как в rezhim_e1.imoex_daily
METKI = {"рост": "rost", "падение": "padenie", "боковик": "bokovik"}
REZHIMY = ("rost", "padenie", "bokovik", "net_ma")
S0, PRAVILO = "2014-08-26", "В-4 v3"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tablica(imoex_put=None, s: str = S0, do: str = "2024-12-31") -> tuple:
    """(DataFrame[data, rezhim], число дней net_ma). Зона чтения — текущая (zona.end()); do ≥ zona.end() — отказ."""
    s, do = pd.Timestamp(s), pd.Timestamp(do)
    if do >= zona.end():
        raise PermissionError(f"--do {do.date()}: граница чтения зоны {zona.name()} — {zona.end().date()} (исключительно) — отказ")
    im = rezhim_e1.imoex_daily(imoex_put)
    close = im.set_index("date")["close"].sort_index()
    trend = fazy._priznaki(close)["trend"].map(METKI).fillna("net_ma")
    trend = trend[(trend.index >= s) & (trend.index <= do)]
    return pd.DataFrame({"data": trend.index.strftime("%Y-%m-%d"), "rezhim": trend.to_numpy()}), int((trend == "net_ma").sum())


def dni_po_godam(tab: pd.DataFrame) -> pd.DataFrame:
    god = tab["data"].str[:4]
    return pd.crosstab(god, tab["rezhim"]).reindex(columns=list(REZHIMY), fill_value=0).rename_axis("god")


def zapisat(tab: pd.DataFrame, net_ma: int, out: Path, istochnik: str) -> str:
    shapka = (f"# v: 1\n# kod: {_sha(Path(fazy.__file__))}\n# istochnik: {istochnik}\n# pravilo: {PRAVILO}\n# net_ma: {net_ma}\n")
    out = Path(out)
    out.write_text(shapka + tab.to_csv(index=False), encoding="utf-8")
    return _sha(out)


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s", default=S0)
    ap.add_argument("--do", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imoex", default=None, help="parquet 10-минуток IMOEX (по умолчанию — файл проекта)")
    a = ap.parse_args(argv)
    with zona.zone_ctx("polnaya"):
        tab, net_ma = tablica(a.imoex, a.s, a.do)
        put = Path(a.imoex or zona.path("imoex", d18.default_main() / IMOEX_10M))
        sha = zapisat(tab, net_ma, Path(a.out), _sha(put))
    print(f"sha256 {sha}\nдней {len(tab)}, net_ma {net_ma}, период {tab['data'].iloc[0]} … {tab['data'].iloc[-1]}")
    print(dni_po_godam(tab).to_string())
    return sha


if __name__ == "__main__":
    main()
