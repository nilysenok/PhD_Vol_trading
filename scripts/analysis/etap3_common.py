"""Общие функции этапа 3: пути, ключевая ставка, формулы метрик (только чтение сохранённых файлов)."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
# Игнорируемые git данные лежат только в основной копии; читаем их ТОЛЬКО на чтение.
MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
HALT = (pd.Timestamp("2022-02-28"), pd.Timestamp("2022-03-24"))  # остановка торгов, включительно
END_ARCHIVE = pd.Timestamp("2026-02-03")  # последний день дневного архива
END_MSP = pd.Timestamp("2025-12-30")  # последний день таблиц МСП (2022-2025)


def find(rel):
    """Путь к файлу: сначала эта копия, потом основная (только чтение)."""
    for base in (ROOT, MAIN):
        p = base / rel
        if p.exists():
            return p
    raise FileNotFoundError(rel)


def key_rate_calendar():
    """Ставка ЦБ по каждому календарному дню (ставка действует до следующего решения), % годовых."""
    k = pd.read_parquet(find("moex_discovery/data/external/macro/raw/key_rate_cbr.parquet"))
    s = k.set_index(pd.to_datetime(k["date"]))["value"].sort_index()
    return s.reindex(pd.date_range(s.index[0], s.index[-1])).ffill()


def in_halt(idx):
    return (idx >= HALT[0]) & (idx <= HALT[1])


def excess_sharpe(total, rf_cd):
    """Формула главы 4 (stage8_economic.calc_excess_sharpe): mean(total - rf*cd)*sqrt(252)/std(total, ddof=1)."""
    s = np.std(total, ddof=1)
    return float(np.mean(total - rf_cd) * np.sqrt(252) / s) if s > 1e-12 else 0.0


def ru(x, nd=2):
    """Число с запятой-разделителем."""
    return f"{x:.{nd}f}".replace(".", ",")
