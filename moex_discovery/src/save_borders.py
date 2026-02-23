"""Получение границ свечей для всех тикеров TQBR.

Использование: python -m moex_discovery.src.save_borders [interval ...]
Интервалы по умолчанию: 10
Примеры:
    python -m moex_discovery.src.save_borders 31 60 24
"""

import sys

import pandas as pd

from moex_discovery.src.moex_api import get_candle_borders

INPUT_PATH = "moex_discovery/data/raw/tickers_list.csv"

INTERVAL_LABELS = {10: "10m", 31: "30m", 60: "60m", 24: "1d"}


def fetch_borders(interval: int) -> None:
    label = INTERVAL_LABELS.get(interval, str(interval))
    output_path = f"moex_discovery/data/raw/borders_{label}.csv"

    tickers = pd.read_csv(INPUT_PATH)["ticker"].tolist()
    total = len(tickers)
    rows = []

    print(f"\n--- Интервал {interval} ({label}) ---")
    for i, ticker in enumerate(tickers, 1):
        borders = get_candle_borders(ticker, interval)
        if borders:
            rows.append({
                "ticker": ticker,
                "interval": interval,
                "first_date": borders["first_date"],
                "last_date": borders["last_date"],
            })
        if i % 50 == 0:
            print(f"  Прогресс: {i}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Готово: {len(rows)}/{total} тикеров с данными → {output_path}")


def main() -> None:
    intervals = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [10]
    for interval in intervals:
        fetch_borders(interval)


if __name__ == "__main__":
    main()
