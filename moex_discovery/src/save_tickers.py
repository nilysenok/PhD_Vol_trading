"""Сохранение списка тикеров TQBR в CSV."""

from moex_discovery.src.moex_api import get_all_tickers

OUTPUT_PATH = "moex_discovery/data/raw/tickers_list.csv"

COLUMN_MAP = {
    "SECID": "ticker",
    "SHORTNAME": "name",
    "LOTSIZE": "lot_size",
    "PREVPRICE": "prev_price",
}


def main() -> None:
    df = get_all_tickers()
    df = df[df["PREVPRICE"] > 0]
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено тикеров: {len(df)} → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
