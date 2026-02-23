#!/usr/bin/env python3
"""
АВТОНОМНЫЙ ПАЙПЛАЙН: D-001 → D-002 → D-003
Выполняет полный цикл обработки данных MOEX без остановок.
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests

# ============================================
# КОНСТАНТЫ
# ============================================

BASE_DIR = "moex_discovery"
RAW_DIR = f"{BASE_DIR}/data/raw/candles_10m"
CLEAN_DIR = f"{BASE_DIR}/data/clean/candles_10m"
QUARANTINE_DIR = f"{BASE_DIR}/data/quarantine/candles_10m"
PROCESSED_DIR = f"{BASE_DIR}/data/processed"
REPORTS_DIR = f"{BASE_DIR}/reports"

TICKERS_FILE = f"{PROCESSED_DIR}/quality_tickers.csv"
LOG_FILE = f"{REPORTS_DIR}/full_log.txt"

# API settings
BASE_URL = "https://iss.moex.com/iss"
INTERVAL = 10
DATE_FROM = "2014-06-09"
DATE_TILL = "2026-02-03"
PAGE_SIZE = 500
RATE_LIMIT = 0.2
TIMEOUT = 30
MAX_RETRIES = 3

_last_req = 0.0
_log_buffer = StringIO()


def log(msg: str) -> None:
    """Логировать сообщение в консоль и буфер."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _log_buffer.write(line + "\n")


def save_log() -> None:
    """Сохранить лог в файл."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(_log_buffer.getvalue())


# ============================================
# ЭТАП 1: D-001 — ПРОВЕРКА И ДОЗАГРУЗКА
# ============================================

def fetch_page(ticker: str, start: int) -> list[dict]:
    """Загрузить одну страницу свечей."""
    global _last_req
    url = f"{BASE_URL}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
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
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)
    return []


def download_ticker(ticker: str) -> pd.DataFrame:
    """Скачать все свечи для тикера."""
    all_rows = []
    start = 0
    while True:
        page = fetch_page(ticker, start)
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


def check_file_quality(path: str) -> dict:
    """Проверить качество файла."""
    result = {"exists": False, "rows": 0, "size_mb": 0, "first_date": None, "needs_redownload": False, "reason": ""}

    if not os.path.exists(path):
        result["needs_redownload"] = True
        result["reason"] = "missing"
        return result

    result["exists"] = True
    result["size_mb"] = os.path.getsize(path) / (1024 * 1024)

    try:
        df = pd.read_parquet(path)
        result["rows"] = len(df)
        result["first_date"] = df["timestamp"].min()

        if result["rows"] < 100000:
            result["needs_redownload"] = True
            result["reason"] = f"rows={result['rows']}<100k"
        elif result["size_mb"] < 2:
            result["needs_redownload"] = True
            result["reason"] = f"size={result['size_mb']:.1f}MB<2MB"
    except Exception as e:
        result["needs_redownload"] = True
        result["reason"] = f"read_error: {e}"

    return result


def run_d001(tickers: list[str]) -> pd.DataFrame:
    """Этап D-001: Проверка и дозагрузка."""
    log("=" * 60)
    log("ЭТАП 1: D-001 — ПРОВЕРКА И ДОЗАГРУЗКА")
    log("=" * 60)

    download_log = []

    for i, ticker in enumerate(tickers, 1):
        path = f"{RAW_DIR}/{ticker}.parquet"
        quality = check_file_quality(path)

        status = "ok"
        action = "skip"
        rows = quality["rows"]

        if quality["needs_redownload"]:
            log(f"[{i:2d}/{len(tickers)}] {ticker}: {quality['reason']} — перезагрузка...")
            try:
                df = download_ticker(ticker)
                if not df.empty:
                    df.to_parquet(path, index=False)
                    rows = len(df)
                    action = "redownload"
                    status = "ok"
                    log(f"          → {rows:,} строк загружено")
                else:
                    status = "empty"
                    action = "failed"
                    log(f"          → ПУСТО!")
            except Exception as e:
                status = "error"
                action = "failed"
                log(f"          → ОШИБКА: {e}")
        else:
            log(f"[{i:2d}/{len(tickers)}] {ticker}: ok ({quality['rows']:,} строк, {quality['size_mb']:.1f}MB)")

        download_log.append({
            "ticker": ticker,
            "status": status,
            "action": action,
            "rows": rows,
            "size_mb": round(quality["size_mb"], 2),
            "reason": quality["reason"],
        })

    log_df = pd.DataFrame(download_log)
    log_df.to_csv(f"{REPORTS_DIR}/d001_download_log.csv", index=False)
    log(f"\nD-001 завершён. Лог: {REPORTS_DIR}/d001_download_log.csv")

    return log_df


# ============================================
# ЭТАП 2: D-002 — РАЗДЕЛЕНИЕ ДАННЫХ
# ============================================

def detect_quarantine_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Определить строки для карантина."""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # Вычисления
    df["prev_close"] = df["close"].shift(1)
    df["log_return"] = np.log(df["close"] / df["prev_close"])
    df["rolling_std"] = df["log_return"].rolling(window=100, min_periods=10).std()

    # Время предыдущего бара
    df["prev_timestamp"] = df["timestamp"].shift(1)
    df["time_gap"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 60

    # Извлечь час
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    # Флаги карантина
    df["q_zero_volume"] = df["volume"] == 0
    df["q_large_return"] = df["log_return"].abs() > 0.15
    df["q_outlier_return"] = df["log_return"].abs() > 10 * df["rolling_std"]
    df["q_ohlc_invalid"] = (df["high"] < df["low"]) | (df["close"] < df["low"]) | (df["close"] > df["high"])

    # Gap в рамках торгового дня (10:00-18:50)
    same_day = df["date"] == df["date"].shift(1)
    in_trading_hours = (df["hour"] >= 10) & (df["hour"] < 19)
    df["q_time_gap"] = same_day & in_trading_hours & (df["time_gap"] > 10)

    # Общий флаг
    df["is_quarantine"] = (
        df["q_zero_volume"] |
        df["q_large_return"] |
        df["q_outlier_return"] |
        df["q_ohlc_invalid"] |
        df["q_time_gap"]
    )

    return df


def get_quarantine_reason(row: pd.Series) -> str:
    """Определить причину карантина."""
    reasons = []
    if row.get("q_zero_volume"):
        reasons.append("zero_volume")
    if row.get("q_large_return"):
        reasons.append("large_return")
    if row.get("q_outlier_return"):
        reasons.append("outlier_return")
    if row.get("q_ohlc_invalid"):
        reasons.append("ohlc_invalid")
    if row.get("q_time_gap"):
        reasons.append("time_gap")
    return "|".join(reasons) if reasons else ""


def run_d002(tickers: list[str]) -> pd.DataFrame:
    """Этап D-002: Разделение данных."""
    log("\n" + "=" * 60)
    log("ЭТАП 2: D-002 — РАЗДЕЛЕНИЕ ДАННЫХ")
    log("=" * 60)

    summary_rows = []
    quarantine_reports = []

    for i, ticker in enumerate(tickers, 1):
        raw_path = f"{RAW_DIR}/{ticker}.parquet"
        clean_path = f"{CLEAN_DIR}/{ticker}.parquet"
        quar_path = f"{QUARANTINE_DIR}/{ticker}.parquet"

        try:
            df = pd.read_parquet(raw_path)
            total_rows = len(df)

            # Обработка
            df = detect_quarantine_flags(df)

            # Разделение
            clean_df = df[~df["is_quarantine"]].copy()
            quar_df = df[df["is_quarantine"]].copy()

            # Сохранение clean
            clean_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            clean_df[clean_cols].to_parquet(clean_path, index=False)

            # Сохранение quarantine
            if not quar_df.empty:
                quar_df[clean_cols].to_parquet(quar_path, index=False)

                # Отчёт по карантину
                for _, row in quar_df.iterrows():
                    quarantine_reports.append({
                        "ticker": ticker,
                        "timestamp": row["timestamp"],
                        "reason": get_quarantine_reason(row),
                        "log_return": row["log_return"],
                        "close": row["close"],
                        "prev_close": row["prev_close"],
                        "volume": row["volume"],
                    })

            clean_rows = len(clean_df)
            quar_rows = len(quar_df)
            pct = 100 * quar_rows / total_rows if total_rows > 0 else 0

            summary_rows.append({
                "ticker": ticker,
                "total_rows": total_rows,
                "clean_rows": clean_rows,
                "quarantine_rows": quar_rows,
                "pct_quarantine": round(pct, 2),
            })

            log(f"[{i:2d}/{len(tickers)}] {ticker}: {total_rows:,} → clean:{clean_rows:,}, quar:{quar_rows:,} ({pct:.1f}%)")

        except Exception as e:
            log(f"[{i:2d}/{len(tickers)}] {ticker}: ОШИБКА — {e}")
            summary_rows.append({
                "ticker": ticker,
                "total_rows": 0,
                "clean_rows": 0,
                "quarantine_rows": 0,
                "pct_quarantine": 0,
            })

    # Сохранение отчётов
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{REPORTS_DIR}/d002_summary.csv", index=False)

    quar_report_df = pd.DataFrame(quarantine_reports)
    if not quar_report_df.empty:
        quar_report_df.to_csv(f"{REPORTS_DIR}/quarantine_report.csv", index=False)

    log(f"\nD-002 завершён.")
    log(f"  Сводка: {REPORTS_DIR}/d002_summary.csv")
    log(f"  Карантин: {REPORTS_DIR}/quarantine_report.csv ({len(quarantine_reports)} записей)")

    return summary_df


# ============================================
# ЭТАП 3: D-003 — РАСЧЁТ ВОЛАТИЛЬНОСТИ
# ============================================

def calculate_daily_rv(df: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать дневную реализованную волатильность."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # Фильтр торговых часов (10:00 - 18:50)
    df = df[(df["hour"] >= 10) & ((df["hour"] < 18) | ((df["hour"] == 18) & (df["minute"] <= 50)))]

    # Log return внутри дня
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Группировка по дням
    daily = df.groupby("date").agg(
        n_bars=("close", "count"),
        rv_daily=("log_return", lambda x: (x ** 2).sum()),
    ).reset_index()

    # Фильтр: минимум 42 бара (80% от 52)
    daily = daily[daily["n_bars"] >= 42]

    # Годовая волатильность
    daily["rv_annualized"] = np.sqrt(daily["rv_daily"] * 252)

    return daily


def run_d003(tickers: list[str]) -> pd.DataFrame:
    """Этап D-003: Расчёт реализованной волатильности."""
    log("\n" + "=" * 60)
    log("ЭТАП 3: D-003 — РАСЧЁТ РЕАЛИЗОВАННОЙ ВОЛАТИЛЬНОСТИ")
    log("=" * 60)

    all_rv = []
    stats_rows = []

    for i, ticker in enumerate(tickers, 1):
        clean_path = f"{CLEAN_DIR}/{ticker}.parquet"

        try:
            df = pd.read_parquet(clean_path)
            daily_rv = calculate_daily_rv(df)
            daily_rv["ticker"] = ticker

            n_days = len(daily_rv)
            if n_days > 0:
                mean_rv = daily_rv["rv_annualized"].mean()
                median_rv = daily_rv["rv_annualized"].median()
                std_rv = daily_rv["rv_annualized"].std()
                min_rv = daily_rv["rv_annualized"].min()
                max_rv = daily_rv["rv_annualized"].max()
            else:
                mean_rv = median_rv = std_rv = min_rv = max_rv = 0

            all_rv.append(daily_rv)
            stats_rows.append({
                "ticker": ticker,
                "n_days": n_days,
                "mean_rv": round(mean_rv, 4),
                "median_rv": round(median_rv, 4),
                "std_rv": round(std_rv, 4),
                "min_rv": round(min_rv, 4),
                "max_rv": round(max_rv, 4),
            })

            log(f"[{i:2d}/{len(tickers)}] {ticker}: {n_days} дней, mean_rv={mean_rv:.1%}, median={median_rv:.1%}")

        except Exception as e:
            log(f"[{i:2d}/{len(tickers)}] {ticker}: ОШИБКА — {e}")
            stats_rows.append({
                "ticker": ticker,
                "n_days": 0,
                "mean_rv": 0,
                "median_rv": 0,
                "std_rv": 0,
                "min_rv": 0,
                "max_rv": 0,
            })

    # Сохранение
    if all_rv:
        rv_df = pd.concat(all_rv, ignore_index=True)
        rv_df = rv_df[["ticker", "date", "rv_daily", "n_bars", "rv_annualized"]]
        rv_df.to_parquet(f"{PROCESSED_DIR}/rv_daily.parquet", index=False)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(f"{REPORTS_DIR}/d003_rv_stats.csv", index=False)

    log(f"\nD-003 завершён.")
    log(f"  RV данные: {PROCESSED_DIR}/rv_daily.parquet")
    log(f"  Статистика: {REPORTS_DIR}/d003_rv_stats.csv")

    return stats_df


# ============================================
# ЭТАП 4: ФИНАЛЬНЫЙ ОТЧЁТ
# ============================================

def generate_final_report(tickers: list[str], d001_log: pd.DataFrame, d002_summary: pd.DataFrame, d003_stats: pd.DataFrame) -> None:
    """Создать финальный отчёт в Markdown."""
    log("\n" + "=" * 60)
    log("ЭТАП 4: ФИНАЛЬНЫЙ ОТЧЁТ")
    log("=" * 60)

    report = []
    report.append("# ФИНАЛЬНЫЙ ОТЧЁТ: Обработка данных MOEX ISS")
    report.append(f"\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nПериод данных: {DATE_FROM} — {DATE_TILL}")
    report.append(f"\nИнтервал: 10 минут")

    # 1. Сводка по загрузке
    report.append("\n## 1. Сводка по загрузке (D-001)")
    report.append(f"\n- Всего тикеров: {len(tickers)}")
    report.append(f"- Успешно загружено: {len(d001_log[d001_log['status'] == 'ok'])}")
    total_rows = d001_log["rows"].sum()
    report.append(f"- Всего строк: {total_rows:,}")

    # 2. Сводка по очистке
    report.append("\n## 2. Сводка по очистке (D-002)")
    total_clean = d002_summary["clean_rows"].sum()
    total_quar = d002_summary["quarantine_rows"].sum()
    pct_quar = 100 * total_quar / (total_clean + total_quar) if (total_clean + total_quar) > 0 else 0
    report.append(f"\n- Чистых строк: {total_clean:,}")
    report.append(f"- В карантине: {total_quar:,} ({pct_quar:.2f}%)")

    report.append("\n### Карантин по тикерам:")
    report.append("\n| Тикер | Всего | Чистых | Карантин | % |")
    report.append("|-------|-------|--------|----------|---|")
    for _, row in d002_summary.sort_values("pct_quarantine", ascending=False).head(10).iterrows():
        report.append(f"| {row['ticker']} | {row['total_rows']:,} | {row['clean_rows']:,} | {row['quarantine_rows']:,} | {row['pct_quarantine']:.1f}% |")

    # 3. Топ-20 событий карантина
    report.append("\n## 3. Топ-20 событий в карантине (по |log_return|)")
    quar_file = f"{REPORTS_DIR}/quarantine_report.csv"
    if os.path.exists(quar_file):
        quar_df = pd.read_csv(quar_file)
        quar_df["abs_return"] = quar_df["log_return"].abs()
        top20 = quar_df.nlargest(20, "abs_return")
        report.append("\n| Тикер | Дата | Причина | Return | Close | Prev |")
        report.append("|-------|------|---------|--------|-------|------|")
        for _, row in top20.iterrows():
            ts = str(row["timestamp"])[:19]
            ret = f"{row['log_return']:.1%}" if pd.notna(row["log_return"]) else "N/A"
            report.append(f"| {row['ticker']} | {ts} | {row['reason']} | {ret} | {row['close']:.2f} | {row['prev_close']:.2f} |")

    # 4. Статистика RV
    report.append("\n## 4. Статистика реализованной волатильности (D-003)")
    report.append("\n| Тикер | Дней | Mean RV | Median RV | Std RV | Min | Max |")
    report.append("|-------|------|---------|-----------|--------|-----|-----|")
    for _, row in d003_stats.sort_values("mean_rv", ascending=False).iterrows():
        report.append(f"| {row['ticker']} | {row['n_days']} | {row['mean_rv']:.1%} | {row['median_rv']:.1%} | {row['std_rv']:.1%} | {row['min_rv']:.1%} | {row['max_rv']:.1%} |")

    # 5. Сравнение с диссертацией
    report.append("\n## 5. Сравнение с диссертацией")
    overall_mean_rv = d003_stats["mean_rv"].mean()
    report.append(f"\n- Средняя волатильность по всем тикерам: **{overall_mean_rv:.1%}**")
    report.append(f"- Ожидание из диссертации: ~28%")
    diff = overall_mean_rv - 0.28
    if abs(diff) < 0.05:
        report.append(f"- Результат: ✓ Соответствует ожиданиям (отклонение {diff:+.1%})")
    else:
        report.append(f"- Результат: ⚠ Отклонение {diff:+.1%} от ожидания")

    # 6. Проблемы и предупреждения
    report.append("\n## 6. Проблемы и предупреждения")
    issues = []

    failed = d001_log[d001_log["status"] != "ok"]
    if len(failed) > 0:
        issues.append(f"- Не удалось загрузить: {', '.join(failed['ticker'].tolist())}")

    high_quar = d002_summary[d002_summary["pct_quarantine"] > 5]
    if len(high_quar) > 0:
        issues.append(f"- Высокий % карантина (>5%): {', '.join(high_quar['ticker'].tolist())}")

    low_days = d003_stats[d003_stats["n_days"] < 2000]
    if len(low_days) > 0:
        issues.append(f"- Мало торговых дней (<2000): {', '.join(low_days['ticker'].tolist())}")

    if not issues:
        issues.append("- Нет критических проблем")

    report.extend(issues)

    # Сохранение
    report_text = "\n".join(report)
    with open(f"{REPORTS_DIR}/FINAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    log(f"\nФинальный отчёт: {REPORTS_DIR}/FINAL_REPORT.md")


# ============================================
# MAIN
# ============================================

def main():
    """Запуск полного пайплайна."""
    start_time = time.monotonic()

    log("=" * 60)
    log("АВТОНОМНЫЙ ПАЙПЛАЙН: D-001 → D-002 → D-003")
    log(f"Запуск: {datetime.now()}")
    log("=" * 60)

    try:
        # Загрузить список тикеров
        tickers = pd.read_csv(TICKERS_FILE)["ticker"].tolist()
        log(f"\nТикеров к обработке: {len(tickers)}")
        log(f"Тикеры: {', '.join(tickers)}")

        # Этапы
        d001_log = run_d001(tickers)
        d002_summary = run_d002(tickers)
        d003_stats = run_d003(tickers)

        # Финальный отчёт
        generate_final_report(tickers, d001_log, d002_summary, d003_stats)

    except Exception as e:
        log(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        log(traceback.format_exc())

    elapsed = time.monotonic() - start_time
    log(f"\n{'=' * 60}")
    log(f"ПАЙПЛАЙН ЗАВЕРШЁН")
    log(f"Время выполнения: {elapsed/60:.1f} минут")
    log("=" * 60)

    # Сохранить лог
    save_log()


if __name__ == "__main__":
    main()
