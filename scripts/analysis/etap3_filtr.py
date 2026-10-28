"""Этап 3, п. 2: фильтрация сигналов подходов C и D относительно A.

Вход: дневной архив walkforward_v4/daily_positions.parquet (tf=daily, 102 пары стратегия x тикер, позиция и
валовая доходность по дням) и дневные ряды МСП ALL6xEW (msp/extended_daily_returns). Без переобучения.
Сигнал (сделка) A = непрерывный эпизод ненулевой позиции A одного знака в паре. Эпизод 'отсеян' X, если X
ни в один день эпизода не держит позицию того же знака. Результат эпизода = произведение (1+валовая доходность A) - 1.
"""
import sys
import numpy as np
import pandas as pd
from etap3_common import find, ru, HALT, END_MSP

POS = "moex_volatility/results/final/strategies/walkforward_v4/daily_positions.parquet"
MSP = "moex_volatility/results/final/strategies/msp/extended_daily_returns/msp_MSP_ALL6_EW_{}_inst.csv"


def load(drop_0324=True):
    d = pd.read_parquet(find(POS), columns=["date", "strategy", "tf", "ticker", "approach", "position", "daily_gross_return"])
    d = d[(d.tf == "daily") & (d.date >= "2022-01-01")]
    if drop_0324:  # 28.02-24.03.2022 исключены из метрик; в архиве остался только день возобновления 24.03
        d = d[~((d.date >= HALT[0]) & (d.date <= HALT[1]))]
    cols = ["strategy", "ticker"]
    pos = d.pivot_table(index="date", columns=["approach"] + cols, values="position")
    gr = d.pivot_table(index="date", columns=["approach"] + cols, values="daily_gross_return")
    return pos, gr


def episodes(pos, gr, x):
    """Список эпизодов A: (пара, длина, результат A, доля дней X того же знака, есть ли данные X, результат X в дни эпизода)."""
    out = []
    for pair in pos["A"].columns:
        a = pos["A"][pair].fillna(0).values
        ga = gr["A"][pair].fillna(0).values
        xs = pos[x][pair].values if pair in pos[x].columns else np.full(len(a), np.nan)
        gx = gr[x][pair].fillna(0).values if pair in gr[x].columns else np.zeros(len(a))
        prev = np.r_[0.0, a[:-1]]
        start = (a != 0) & ((prev == 0) | (np.sign(prev) != np.sign(a)))
        eid = np.cumsum(start) * (a != 0)
        for e in range(1, eid.max() + 1):
            m = eid == e
            same = (np.sign(np.nan_to_num(xs[m])) == np.sign(a[m]))
            out.append(dict(pair=pair, n=int(m.sum()), ra=np.prod(1 + ga[m]) - 1, f=same.mean(),
                            nodata=bool(np.isnan(xs[m]).any()), rx=np.prod(1 + gx[m]) - 1))
    return pd.DataFrame(out)


def summarize(ep, x):
    n_all = len(ep)
    e = ep[~ep.nodata]
    rem = e[e.f == 0]
    kept = e[e.f > 0]
    part = e[(e.f > 0) & (e.f < 1)]
    days_all = e.n.sum()
    days_same = (e.n * e.f).sum()
    return {
        "подход": x, "сигналов A": n_all, "без данных X (исключены)": int(ep.nodata.sum()),
        "сигналов в расчёте": len(e), "отсеяно целиком": len(rem), "доля отсеянных, %": 100 * len(rem) / len(e),
        "сокращено частично": len(part), "доля позиций-дней A без X, %": 100 * (1 - days_same / days_all),
        "убыточных среди отсеянных, % (по A)": 100 * (rem.ra < 0).mean(),
        "убыточных среди оставшихся, % (по A)": 100 * (kept.ra < 0).mean(),
        "убыточных среди оставшихся, % (по X)": 100 * (kept.rx < 0).mean(),
        "убыточных среди всех сигналов A, %": 100 * (e.ra < 0).mean(),
        "средний результат отсеянных, %": 100 * rem.ra.mean(), "средний результат оставшихся, % (A)": 100 * kept.ra.mean(),
    }


def day_compare(pos, gr, x, last=None):
    """Дни, где EW-среднее валовой доходности X по парам лучше A (только пары, где у X есть данные)."""
    ga, gx = gr["A"].copy(), gr[x].reindex(columns=gr["A"].columns)
    ok = pos[x].reindex(columns=gr["A"].columns).notna() & gr["A"].notna()
    ra = ga.where(ok).mean(axis=1)
    rx = gx.where(ok).mean(axis=1)
    if last is not None:
        ra, rx = ra[:last], rx[:last]
    ra, rx = ra.dropna(), rx.dropna()
    diff = (rx - ra) * 100  # п.п. за день на единицу позиции
    return {"подход": x, "дней": len(diff), "X лучше A": int((diff > 1e-12).sum()), "X хуже A": int((diff < -1e-12).sum()),
            "поровну": int((diff.abs() <= 1e-12).sum()), "доля дней X лучше, %": 100 * (diff > 1e-12).mean(),
            "средняя разница X-A, п.п./день": diff.mean(), "в днях X лучше, п.п.": diff[diff > 1e-12].mean(),
            "в днях X хуже, п.п.": diff[diff < -1e-12].mean()}


def msp_days(x):
    a = pd.read_csv(find(MSP.format("A")), index_col=0, parse_dates=True)["excess_ret"]
    b = pd.read_csv(find(MSP.format(x)), index_col=0, parse_dates=True)["excess_ret"]
    d = ((b - a) * 100).dropna()
    return {"подход": x, "дней": len(d), "X лучше A": int((d > 0).sum()), "доля, %": 100 * (d > 0).mean(),
            "средняя разница X-A, п.п./день": d.mean(), "в днях X лучше, п.п.": d[d > 0].mean(), "в днях X хуже, п.п.": d[d < 0].mean()}


def show(title, rows):
    df = pd.DataFrame(rows).set_index("подход")
    print(f"\n{title}\n")
    print("| показатель | " + " | ".join(df.index) + " |\n|---|" + "---|" * len(df.index))
    for c in df.columns:
        print(f"| {c} | " + " | ".join(ru(v, 4 if "п.п." in c else 2) if isinstance(v, float) else str(v) for v in df[c]) + " |")


def main():
    keep = len(sys.argv) > 1 and sys.argv[1] == "keep0324"
    pos, gr = load(drop_0324=not keep)
    print(f"дней в расчёте: {len(pos)} (архив 1 089 дней, из них 24.03.2022 {'оставлен' if keep else 'исключён'}), пар: {pos['A'].shape[1]}")
    eps = {x: episodes(pos, gr, x) for x in ("C", "D")}
    show("Сигналы A и их фильтрация (окно: весь архив 2022-01-03..2026-02-03)", [summarize(eps[x], x) for x in eps])
    show("Дни, где X лучше A (EW по парам, валовая доходность, весь архив)", [day_compare(pos, gr, x) for x in ("C", "D")])
    show("То же, только 2022-2025 (до 30.12.2025)", [day_compare(pos, gr, x, END_MSP) for x in ("C", "D")])
    show("Контроль: дневные ряды МСП ALL6xEW (excess_ret, после комиссии, 2022-2025)", [msp_days(x) for x in ("C", "D")])
    # «попробуй сломать»: перепутанный знак C обязан увеличить долю отсеянных; сдвиг окна на год - сменить число дней/долю
    neg = pos.copy()
    neg["C"] = -pos["C"]
    assert summarize(episodes(neg, gr, "C"), "C")["доля отсеянных, %"] > summarize(eps["C"], "C")["доля отсеянных, %"]
    shifted = day_compare(pos, gr, "C", pd.Timestamp("2024-12-30"))
    assert shifted["дней"] < day_compare(pos, gr, "C")["дней"]
    print("самопроверка (знак C, сдвиг окна): OK")
    a = eps["C"]
    print(f"\nвсего эпизодов A: {len(a)}, средняя длина {a.n.mean():.1f} дн.")


if __name__ == "__main__":
    main()
