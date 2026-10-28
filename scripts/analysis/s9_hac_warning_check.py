"""Проверка к отчёту docs/otchety/2026-10-26_test_glava4.md, раздел 10: предупреждения журнала теста
`fastbt/s9_stats.py:89: RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul` —
ложные флаги BLAS (Accelerate, numpy 2.0.2, arm64); матрица HAC конечна и равна расчёту без BLAS (np.einsum).

Синтетика того же размера, что окна теста (T = 1089 и 345 дней, 4 столбца моментов Ледуа–Вольфа).
Данные теста не читаются, тест не запускается.

python3 scripts/analysis/s9_hac_warning_check.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastbt import s9_stats  # noqa: E402


def hac_ref(y):
    """Та же HAC-оценка (Ньюи–Уэст, Бартлетт), что s9_stats._hac, но через einsum без BLAS."""
    T = len(y)
    L = int(4 * (T / 100.0) ** (2.0 / 9.0))
    S = np.einsum("ti,tj->ij", y, y, optimize=False) / T
    for k in range(1, L + 1):
        G = np.einsum("ti,tj->ij", y[k:], y[:-k], optimize=False) / T
        S += (1 - k / (L + 1)) * (G + G.T)
    return S


def main():
    rng = np.random.RandomState(0)
    for T in (1089, 345):
        x1 = rng.normal(0, 0.003, T)
        x2 = 0.6 * x1 + rng.normal(0, 0.002, T)
        _, _, y = s9_stats._lw_parts(x1, x2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            S = s9_stats._hac(y)
            _, se, p = s9_stats.lw_sharpe_diff(x1, x2, reps=200, block=5)
        R = hac_ref(y)
        print(f"T={T}: предупреждений {len(w)} {sorted({str(x.message) for x in w})}; "
              f"HAC конечна: {bool(np.isfinite(S).all())}; max|S − S_einsum| / max|S| = "
              f"{np.max(np.abs(S - R)) / np.max(np.abs(R)):.2e}; se = {se:.4f}, p = {p:.4f}")
    print("numpy", np.__version__)


if __name__ == "__main__":
    main()
