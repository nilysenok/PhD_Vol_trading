"""M-008: Объединение границ свечей всех интервалов и оценка качества."""

import pandas as pd

RAW_DIR = "moex_discovery/data/raw"
OUT_DIR = "moex_discovery/data/processed"

FILES = {
    "10m": f"{RAW_DIR}/borders_10m.csv",
    "30m": f"{RAW_DIR}/borders_30m.csv",
    "60m": f"{RAW_DIR}/borders_60m.csv",
    "1d":  f"{RAW_DIR}/borders_1d.csv",
}

ACTIVE_CUTOFF = pd.Timestamp("2026-02-03")
HISTORY_CUTOFF = pd.Timestamp("2014-06-09")


def main() -> None:
    # 1. Загрузить и объединить
    merged = None
    for label, path in FILES.items():
        df = pd.read_csv(path)
        df = df.rename(columns={
            "first_date": f"first_{label}",
            "last_date": f"last_{label}",
        })[["ticker", f"first_{label}", f"last_{label}"]]
        merged = df if merged is None else merged.merge(df, on="ticker", how="outer")

    # 2. Привести даты к datetime
    date_cols = [c for c in merged.columns if c.startswith(("first_", "last_"))]
    for col in date_cols:
        merged[col] = pd.to_datetime(merged[col])

    # 3. Расчётные колонки
    merged["days_10m"] = (merged["last_10m"] - merged["first_10m"]).dt.days
    merged["years_10m"] = merged["days_10m"] / 365.25
    merged["active"] = merged["last_10m"] >= ACTIVE_CUTOFF
    merged["has_10y"] = merged["first_10m"] <= HISTORY_CUTOFF
    merged["quality"] = merged["active"] & merged["has_10y"]

    # 4. Сохранить availability_all.csv
    merged.to_csv(f"{OUT_DIR}/availability_all.csv", index=False)

    # 5. Сохранить quality_tickers.csv
    quality = merged[merged["quality"]][["ticker", "first_10m", "last_10m", "years_10m"]]
    quality = quality.sort_values("years_10m", ascending=False).reset_index(drop=True)
    quality.to_csv(f"{OUT_DIR}/quality_tickers.csv", index=False)

    # 6. Статистика
    print(f"Всего тикеров:        {len(merged)}")
    print(f"Активных тикеров:     {merged['active'].sum()}")
    print(f"С историей 10+ лет:   {merged['has_10y'].sum()}")
    print(f"Quality (оба):        {merged['quality'].sum()}")
    print(f"\nQuality тикеры ({len(quality)}):")
    print(", ".join(quality["ticker"].tolist()))


if __name__ == "__main__":
    main()
