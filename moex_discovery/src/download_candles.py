"""D-001: Скачать 10-минутные свечи для quality-тикеров TQBR."""

import os
import time

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss"
INPUT_PATH = "moex_discovery/data/processed/quality_tickers.csv"
OUTPUT_DIR = "moex_discovery/data/raw/candles_10m"

INTERVAL = 10
DATE_FROM = "2014-06-09"
DATE_TILL = "2026-02-03"
PAGE_SIZE = 500
RATE_LIMIT = 0.2
TIMEOUT = 30
MAX_RETRIES = 3

_last_req: float = 0.0


def _fetch_page(ticker: str, start: int) -> list[dict]:
    """Загрузить одну страницу свечей с retry."""
    global _last_req
    url = (
        f"{BASE_URL}/engines/stock/markets/shares/boards/TQBR"
        f"/securities/{ticker}/candles.json"
    )
    params = {
        "interval": INTERVAL,
        "from": DATE_FROM,
        "till": DATE_TILL,
        "start": start,
        "iss.meta": "off",
        "iss.json": "extended",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.monotonic() - _last_req
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed)
        _last_req = time.monotonic()
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()[1]["candles"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)


def download_ticker(ticker: str) -> pd.DataFrame:
    """Скачать все 10-мин свечи для одного тикера."""
    all_rows = []
    start = 0
    while True:
        page = _fetch_page(ticker, start)
        if not page:
            break
        all_rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"begin": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def main() -> None:
    tickers = pd.read_csv(INPUT_PATH)["ticker"].tolist()

    # Определить уже скачанные
    existing = {
        f.replace(".parquet", "")
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".parquet")
    }
    missing = [t for t in tickers if t not in existing]

    print(f"Всего: {len(tickers)}, уже есть: {len(existing)}, осталось: {len(missing)}")
    if existing:
        print(f"Пропускаю: {', '.join(sorted(existing))}\n")

    summary = []

    for i, ticker in enumerate(missing, 1):
        t0 = time.monotonic()
        df = download_ticker(ticker)
        elapsed = time.monotonic() - t0

        if df.empty:
            print(f"[{i:2d}/{len(missing)}] {ticker:6s} — нет данных")
            continue

        path = f"{OUTPUT_DIR}/{ticker}.parquet"
        df.to_parquet(path, index=False)

        first = df["timestamp"].min()
        last = df["timestamp"].max()
        summary.append({
            "ticker": ticker,
            "rows": len(df),
            "first": str(first.date()),
            "last": str(last.date()),
            "elapsed_s": round(elapsed, 1),
        })
        print(
            f"[{i:2d}/{len(missing)}] {ticker:6s}  "
            f"{len(df):>7,} строк  {first.date()} — {last.date()}  "
            f"({elapsed:.1f}с)"
        )

    # Сводная таблица по ВСЕМ файлам (старые + новые)
    print("\n" + "=" * 65)
    print("Сводка по всем скачанным файлам:")
    print(f"{'Тикер':>8} {'Строк':>9} {'Начало':>12} {'Конец':>12}")
    print("-" * 65)
    total_rows = 0
    for ticker in tickers:
        path = f"{OUTPUT_DIR}/{ticker}.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        first = df["timestamp"].min()
        last = df["timestamp"].max()
        print(f"{ticker:>8} {len(df):>9,} {str(first.date()):>12} {str(last.date()):>12}")
        total_rows += len(df)
    print("-" * 65)
    all_files = [t for t in tickers if os.path.exists(f"{OUTPUT_DIR}/{t}.parquet")]
    print(f"{'ИТОГО':>8} {total_rows:>9,}  тикеров: {len(all_files)}/{len(tickers)}")


if __name__ == "__main__":
    main()
