"""PBO по ledger проб стратегии (Ж-031, Э1-а/Э1-д): матрица T×N из `Sweep.excess` -> `sweep_drivers.pbo_cscv(…, 8)`.

Столбец = проба (точка на периоде отбора при комиссии отбора 0,04), строка = день, значение = дневная избыточная
доходность (`excess_d`). Все столбцы обязаны иметь одни и те же даты: иначе матрица склеивает разные дни — отказ.
CSCV выбирает лучшую пробу на половине блоков по средней избыточной (порядок как по Шарпу при общем масштабе)
и смотрит её ранг на второй половине. Сама статистика — модуль владельца; здесь только сбор матрицы.
"""
from __future__ import annotations

from fastbt import sweep_drivers as sd

N_BLOCKS = 8


def perf_matrix(excess: dict) -> tuple:
    """(daty, матрица T×N списком строк, ключи проб в порядке столбцов). Разные даты у проб — ValueError."""
    keys = sorted(excess)
    if not keys:
        raise ValueError("ledger без excess_d: PBO считать не из чего")
    daty = list(excess[keys[0]]["daty"])
    for k in keys:
        if list(excess[k]["daty"]) != daty or len(excess[k]["excess_d"]) != len(daty):
            raise ValueError(f"пробы на разных датах (проба {k}): матрица T×N невозможна")
    cols = [excess[k]["excess_d"] for k in keys]
    return daty, [list(r) for r in zip(*cols)], keys


def pbo(excess: dict, n_blocks: int = N_BLOCKS) -> dict:
    """PBO стратегии по её пробам. Возвращает ответ `pbo_cscv` + `n_prob`, `T`; меньше 2 проб или T < блоков — pbo = nan."""
    daty, m, keys = perf_matrix(excess)
    return {**sd.pbo_cscv(m, n_blocks), "n_prob": len(keys), "T": len(daty)}


def pbo_sweep(sw, n_blocks: int = N_BLOCKS) -> dict:
    """PBO по ledger оценщика `Sweep` (пробы накопленные: OAT + факторный + …)."""
    return pbo(sw.excess, n_blocks)
