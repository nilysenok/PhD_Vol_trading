"""Этап 3, п. 3: бенчмарк с той же экспозицией: доля w в индексе, остальное под ключевую ставку, ежедневная ребалансировка.

Формулы метрик - из stage5_msp.compute_metrics (глава 4): ann_ret = sum(total)/лет, vol = std(total)*sqrt(252),
Шарп = mean(total - rf*cd)*sqrt(252)/std(total), Сортино = mean(excess)*sqrt(252)/std(total[total<0]), MDD по сумме (cumsum).
Индекс: MCFTR в репозитории нет. Запасной вариант - ценовой IMOEX из 10-минуток (без дивидендов): нижняя граница.
Свой ряд: python3 etap3_bench.py путь.csv (колонки date, close).
"""
import sys
import numpy as np
import pandas as pd
from etap3_common import find, key_rate_calendar, in_halt, ru, END_ARCHIVE, END_MSP

W = {"A (15,7%)": 0.157, "C (7,2%)": 0.072, "D (12,2%)": 0.122}
MSP = "moex_volatility/results/final/strategies/msp/extended_daily_returns/msp_MSP_ALL6_EW_{}_inst.csv"


def index_close(path=None):
    if path:
        d = pd.read_csv(path, parse_dates=["date"])
        return d.set_index("date")["close"].sort_index(), "MCFTR (файл " + path + ")"
    p = find("moex_discovery/data/external/moex_iss/indices_10m/IMOEX.parquet")
    d = pd.read_parquet(p, columns=["begin", "close"])
    d["date"] = pd.to_datetime(d["begin"]).dt.normalize()
    return d.groupby("date")["close"].last(), "IMOEX ценовой (без дивидендов, НЕ MCFTR)"


def metrics(total, rf_cd, n_years=None):
    n = len(total)
    n_years = n_years or n / 252
    exc = total - rf_cd
    down = total[total < 0]
    cum = np.cumsum(total)
    return {"ann": 100 * total.sum() / n_years, "vol": 100 * np.std(total, ddof=1) * np.sqrt(252),
            "sharpe": np.mean(exc) * np.sqrt(252) / np.std(total, ddof=1),
            "sortino": np.mean(exc) * np.sqrt(252) / np.std(down, ddof=1),
            "mdd": 100 * np.max(np.maximum.accumulate(cum) - cum), "n": n}


def build(close, rate, first, last):
    """Дневные ряды на календаре самого индекса (будни; акции с 03.2025 торгуются и в выходные, индекс - нет):
    доходность индекса и ставка за календарные дни с прошлой даты индекса. Остановка 28.02-24.03.2022 исключена."""
    c = close.dropna().loc["2021-12-01":]
    rs = c.pct_change(fill_method=None)  # простая доходность; в метрики идёт логарифмическая, как у акций в главе 4 (pos*lr)
    df = pd.DataFrame({"rs": rs, "r": np.log1p(rs), "cd": pd.Series(c.index, index=c.index).diff().dt.days}).loc[first:last]
    df = df[~in_halt(df.index)]
    df["rf_cd"] = rate.reindex(df.index).values / 100 / 365 * df["cd"]
    return df.dropna()


def table(df, label, msp_dates=None):
    rows = []
    for name, w in list(W.items()) + [("100% индекс", 1.0)]:
        total = w * df["r"].values + (1 - w) * df["rf_cd"].values
        m = metrics(total, df["rf_cd"].values)
        simple = w * df["rs"].values + (1 - w) * df["rf_cd"].values  # точная дневная доходность при ребалансировке
        m["cagr"] = 100 * (np.prod(1 + simple) ** (252 / len(simple)) - 1)
        rows.append((f"{label}, {name}", m))
    return rows


def msp_rows():
    out = []
    for a in "ACD":
        f = pd.read_csv(find(MSP.format(a)), index_col=0, parse_dates=True)
        f = f[~in_halt(f.index)]  # в рядах МСП остался день возобновления 24.03.2022: для сопоставимости снимаем
        out.append((f"МСП-{a} (ALL6xEW, после комиссии)", metrics(f["total_ret"].values, (f["total_ret"] - f["excess_ret"]).values)))
    return out


def show(title, rows):
    print(f"\n{title}\n\n| Портфель | Годовая доходность, % | Волатильность, % | Шарп (избыт.) | Сортино | MDD, % | Дней | CAGR, % |\n|---|---|---|---|---|---|---|---|")
    for name, m in rows:
        cg = ru(m["cagr"]) if "cagr" in m else "—"
        print(f"| {name} | {ru(m['ann'])} | {ru(m['vol'])} | {ru(m['sharpe'])} | {ru(m['sortino'])} | {ru(m['mdd'])} | {m['n']} | {cg} |")


def main():
    close, src = index_close(sys.argv[1] if len(sys.argv) > 1 else None)
    rate = key_rate_calendar()
    print("Источник индекса:", src)
    d1 = build(close, rate, "2022-01-03", END_MSP)
    d2 = build(close, rate, "2022-01-03", END_ARCHIVE)
    show("Окно 1: 03.01.2022-30.12.2025 (окно таблиц МСП), без остановки", table(d1, "бенчмарк") + msp_rows())
    if len(sys.argv) > 1:  # второй вариант: ценовой IMOEX (нижняя граница), те же веса и окно
        ic, _ = index_close(None)
        show("Окно 1, IMOEX ценовой (без дивидендов) - вторая строка к MCFTR", table(build(ic, rate, "2022-01-03", END_MSP), "IMOEX")[:3])
    show("Окно 2: 03.01.2022-03.02.2026 (весь дневной архив), без остановки", table(d2, "бенчмарк"))
    if src.startswith("IMOEX"):  # потолок: ценовой индекс + условная дивидендная доходность d, начисляемая по календарным дням
        for dy in (5, 8):
            add = dy / 100 / 365 * d1["cd"]
            dd = d1.assign(r=d1["r"] + add, rs=d1["rs"] + add)
            show(f"Оценка-потолок (НЕ данные): ценовой IMOEX + {dy}% годовых дивидендов, окно 1", table(dd, f"бенчмарк, див. {dy}%")[:3])
    # проверка формулы на МСП-A: печать главы 4 - ann 19,55 / vol 4,41 / sortino 2,061 / MDD 2,8 (extended_msp.csv)
    a = pd.read_csv(find(MSP.format("A")), index_col=0, parse_dates=True)
    m = metrics(a["total_ret"].values, (a["total_ret"] - a["excess_ret"]).values)
    print(f"\nпроверка формулы на МСП-A: ann {ru(m['ann'])} (печать 19,55), vol {ru(m['vol'])} (4,41), "
          f"Шарп {ru(m['sharpe'])} (1,26 в extended_msp.csv), Сортино {ru(m['sortino'])} (2,06), MDD {ru(m['mdd'])} (2,8)")
    # сдвиг окна на год - должен менять результат (защита от тихой ошибки окна)
    d3 = build(close, rate, "2023-01-03", END_MSP)
    print(f"контроль окна: тот же бенчмарк с 2023: Шарп(A) {ru(metrics(0.157*d3.r.values+0.843*d3.rf_cd.values, d3.rf_cd.values)['sharpe'])}")


if __name__ == "__main__":
    main()
