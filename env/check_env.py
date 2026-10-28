#!/usr/bin/env python3
"""Проверка окружения R2: версия Python и пакетов против FINAL_DESIGN §6 и env/requirements_r2.txt. Выход 1 — не та версия.
python3 env/check_env.py"""
import platform
import sys

WANT = {'numpy': '2.0.2', 'pandas': '2.3.3', 'scipy': '1.13.1', 'sklearn': '1.6.1', 'lightgbm': '4.6.0', 'xgboost': '2.1.4',
        'optuna': '4.7.0', 'statsmodels': '0.14.6', 'arch': '7.2.0', 'pyarrow': '21.0.0', 'yaml': '6.0.3'}


def main():
    bad = []
    for m, v in WANT.items():
        try:
            got = __import__(m).__version__
        except Exception as e:  # noqa: BLE001 — печатаем любую ошибку импорта
            got = f'нет ({type(e).__name__})'
        ok = got == v
        bad += [] if ok else [m]
        print(f'{m:12s} {got:14s} {"OK" if ok else "НЕ ТА ВЕРСИЯ (ждём " + v + ")"}')
    pv = platform.python_version()
    print(f'python       {pv:14s} {"OK" if pv == "3.9.6" else "НЕ ТА ВЕРСИЯ (ждём 3.9.6)"}   {platform.system()} {platform.machine()}')
    sys.exit(1 if bad or pv != '3.9.6' else 0)


if __name__ == '__main__':
    main()
