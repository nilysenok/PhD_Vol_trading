#!/usr/bin/env python3
"""Только дописывание (Ж-025 п. 1.1): новый файл = старый + строки в конце.

Использование: proverka_append.py <старый> <новый>. Код 0 — можно; 2 — нельзя.
Отсутствующий старый файл считается пустым (создание). Хук PreToolUse (Э3) зовёт
proverit(old, new).
"""
import os
import sys

MSG = ("Ж-025 п. 1.1: журнал решений — только дописывание; удаление или правка "
       "существующей строки запрещены. Допустимо добавлять строки в конец файла.")


def proverit(old, new):
    """(ok, сообщение). Старый текст должен быть байт-в-байт началом нового,
    а граница — по концу строки."""
    if new == old:
        return True, "без изменений"
    if not new.startswith(old):
        return False, MSG
    if old and not old.endswith("\n"):
        return False, MSG + " (последняя строка старого файла дописана на месте)"
    return True, "добавлено строк: %d" % len(new[len(old):].splitlines())


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv):
    if len(argv) != 3:
        print("использование: proverka_append.py <старый> <новый>", file=sys.stderr)
        return 2
    ok, msg = proverit(_read(argv[1]), _read(argv[2]))
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
