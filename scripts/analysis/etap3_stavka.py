"""Этап 3, п. 1: средняя ключевая ставка по календарным дням и происхождение старых чисел (14,3 / 14,7 / 21,0)."""
import numpy as np
import pandas as pd
from etap3_common import key_rate_calendar, in_halt, ru, END_ARCHIVE, END_MSP, find


def cal_mean(cal, a, b, drop_halt=False):
    x = cal[a:b]
    if drop_halt:
        x = x[~in_halt(x.index)]
    return float(x.mean()), len(x)


def main():
    cal = key_rate_calendar()
    k = pd.read_parquet(find("moex_discovery/data/external/macro/raw/key_rate_cbr.parquet"))
    last = pd.to_datetime(k["date"]).max()
    print(f"ряд ставки: {k['date'].min()} .. {last.date()}, наблюдений {len(k)} (по рабочим дням), до {last.date()}")
    rows = [
        ("2022 (01.01-31.12)", "2022-01-01", "2022-12-31", False),
        ("2022 без 28.02-24.03", "2022-01-01", "2022-12-31", True),
        ("2023", "2023-01-01", "2023-12-31", False),
        ("2024", "2024-01-01", "2024-12-31", False),
        ("2025", "2025-01-01", "2025-12-31", False),
        ("янв-фев 2026 (01.01-03.02)", "2026-01-01", str(END_ARCHIVE.date()), False),
        ("контроль 2022-2025 (03.01.2022-30.12.2025), без остановки", "2022-01-03", str(END_MSP.date()), True),
        ("контроль до 03.02.2026 (03.01.2022-03.02.2026), без остановки", "2022-01-03", str(END_ARCHIVE.date()), True),
        ("контроль 2022-2025 с остановкой (для сверки)", "2022-01-03", str(END_MSP.date()), False),
    ]
    print("\n| Период | Средняя ставка, % годовых | Календарных дней |\n|---|---|---|")
    for name, a, b, dh in rows:
        m, n = cal_mean(cal, a, b, dh)
        print(f"| {name} | {ru(m)} | {n} |")
    # происхождение старых чисел
    print("\nДиагностика старых чисел (год: календарная / по рабочим дням / ставка на 1 января / максимум года)")
    s = k.set_index(pd.to_datetime(k["date"]))["value"].sort_index()
    old = {2022: 10.6, 2023: 9.9, 2024: 17.4, 2025: 21.0}
    means = []
    for y in range(2022, 2026):
        c = cal[str(y)]
        t = s[str(y)]
        print(f"{y}: календарная {ru(c.mean())} | по рабочим дням {ru(t.mean())} | на 1.01 {ru(cal[f'{y}-01-01'], 1)}"
              f" | максимум {ru(t.max(), 1)} | в старой таблице KR {ru(old[y], 1)}")
        means.append(c.mean())
    print(f"среднее четырёх календарных средних: {ru(np.mean(means))}")
    print(f"среднее четырёх старых KR (10,6/9,9/17,4/21,0): {ru(np.mean(list(old.values())))}")
    print(f"календарная средняя 2022-2025 (01.01.2022-31.12.2025): {ru(cal_mean(cal, '2022-01-01', '2025-12-31')[0])}")
    print(f"по рабочим дням 2022-2025: {ru(s['2022':'2025'].mean())}; 2022 - 03.02.2026: {ru(s['2022':'2026-02-03'].mean())}")
    # «попробуй сломать»: сдвиг окна на год обязан менять среднюю, перепутанный год - не проходить
    assert abs(cal_mean(cal, "2025-01-01", "2025-12-31")[0] - 19.18) < 0.01
    assert abs(cal_mean(cal, "2024-01-01", "2024-12-31")[0] - cal_mean(cal, "2025-01-01", "2025-12-31")[0]) > 1.0
    print("самопроверка (год 2025 = 19,18; сдвиг на год меняет число): OK")
    # контроль: изменения ставки внутри 2025
    ch = s[s.diff().fillna(1) != 0]["2024-12-01":"2026-02-06"]
    print("решения 12.2024-02.2026:", ", ".join(f"{d.date()}={v}" for d, v in ch.items()))


if __name__ == "__main__":
    main()
