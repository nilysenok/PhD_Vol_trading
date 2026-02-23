"""MOEX ISS API — функции для получения тикеров и границ свечей."""

import time
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss"
_REQUEST_INTERVAL = 0.1
_TIMEOUT = 10

# Коды интервалов: 10=10мин, 31=30мин, 60=1час, 24=1день
INTERVAL_CODES = {10: "10 мин", 31: "30 мин", 60: "1 час", 24: "1 день"}

_last_request_time: float = 0.0


def _rate_limit() -> None:
    """Ожидание между запросами для соблюдения rate limiting."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _REQUEST_INTERVAL:
        time.sleep(_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def _get_json(url: str, params: Optional[dict] = None, retries: int = 3) -> dict:
    """GET-запрос к MOEX ISS с обработкой ошибок, retry и rate limiting."""
    if params is None:
        params = {}
    params["iss.meta"] = "off"
    params["iss.json"] = "extended"

    for attempt in range(1, retries + 1):
        _rate_limit()
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == retries:
                raise
            time.sleep(1 * attempt)


def _block_to_df(data: list, block_name: str) -> pd.DataFrame:
    """Извлечь блок из iss.json=extended ответа в DataFrame."""
    block = data[1].get(block_name, [])
    return pd.DataFrame(block) if block else pd.DataFrame()


def get_all_tickers() -> pd.DataFrame:
    """Получить все тикеры с доски TQBR.

    Returns:
        DataFrame с колонками: SECID, SHORTNAME, SECNAME, PREVPRICE, LOTSIZE и др.
    """
    url = f"{BASE_URL}/engines/stock/markets/shares/boards/TQBR/securities.json"
    data = _get_json(url)
    return _block_to_df(data, "securities")


def get_candle_borders(ticker: str, interval: int) -> dict:
    """Получить границы доступных свечей для тикера и интервала.

    Args:
        ticker: Тикер инструмента (например, 'SBER').
        interval: Код интервала (10, 31, 60, 24).

    Returns:
        dict с ключами 'first_date' и 'last_date' (строки ISO datetime),
        или пустой dict, если данных для интервала нет.
    """
    url = (
        f"{BASE_URL}/engines/stock/markets/shares/boards/TQBR"
        f"/securities/{ticker}/candleborders.json"
    )
    data = _get_json(url)
    borders = data[1].get("borders", [])

    for row in borders:
        if row["interval"] == interval:
            return {"first_date": row["begin"], "last_date": row["end"]}

    return {}
