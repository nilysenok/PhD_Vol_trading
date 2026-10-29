#!/usr/bin/env python3
"""Акт снятия замка test_locked/ для единого walk-forward (Ж-136 п. 1–2): ПЕЧАТАЕТ команды владельцу, сам ничего не снимает.

Не пишет файлов, не правит `zamki.json`, не создаёт отметку `docs/resheniya/wf_zamok_snyat.json` — всё это действие владельца (класс A).
Читает один файл — предрегистрацию v2 — только для sha256 (содержимое не печатается).
    python3 scripts/server/akt_snyatiya_zamka.py <docs/preregistraciya_v2.md>
Выход: 0 — команды напечатаны; 1 — нет файла предрегистрации (акт без SHA не печатается)."""
import hashlib
import shlex
import sys
from pathlib import Path

ZAMKI = ".claude/hooks/zamki.json"
OTMETKA = "docs/resheniya/wf_zamok_snyat.json"
DO_WF = "2026-02-28"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def akt(prereg: Path) -> str:
    sha = sha256(prereg)
    p = shlex.quote(str(prereg))
    otm = '{"preregistraciya_v2": "%s", "sha256": "%s", "data_snyatiya": "\'"$(date +%%F)"\'"}' % (prereg, sha)
    return "\n".join([
        f"АКТ СНЯТИЯ ЗАМКА (печать; ничего не выполнено). Предрегистрация v2: {prereg}, sha256 {sha}",
        "0. Проверьте: SHA предрегистрации записан в журнал решений (resheniya.md), запись п. 3 — в reverse_log.",
        f"1. Сверка SHA перед актом:  shasum -a 256 {p}   (ожидается {sha})",
        f"2. Правка {ZAMKI} (вручную, владелец): снять с закрытых путей данные ≥ 2022 для счёта wf;"
        " агент и дирижёр файл не правят.",
        f"3. Отметка (после п. 2):  printf '%s\\n' '{otm}' > {OTMETKA}",
        f"4. Манифест wf одним актом:  python3 scripts/server/manifest_perenos.py --koren <данные> --spisok scripts/server/perenos_spisok.tsv"
        f" --do {DO_WF} --otmetka {OTMETKA}",
        "5. Перенос: PERENOS_DO=%s scripts/server/sobrat_perenos.sh <коммит>; на сервере положить отметку рядом со скриптами"
        " (wf_zamok_snyat.json) и запустить ./zapusk_sobol.sh wf." % DO_WF,
    ])


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 1 or not Path(a[0]).is_file():
        print("FAIL akt: укажите существующий файл предрегистрации v2 (нужен его sha256); акт не напечатан")
        return 1
    print(akt(Path(a[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
