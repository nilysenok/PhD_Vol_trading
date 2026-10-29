# GATES П-0 — профиль точки v5 (малая проверка, класс B, Ж-135 не нарушает)
- G1 скрипт компилируется. CHECK: `python3 -m py_compile scripts/zamer_profil_v5.py` · EXPECT: код 0 · EVIDENCE: pending
- G2 самопроверка без аргументов (разбор pstats, без счёта). CHECK: `python3 scripts/zamer_profil_v5.py` · EXPECT: selftest ok · EVIDENCE: pending
- G3 профиль 20 точек на Sk: топ-10. CHECK: `python3 scripts/zamer_profil_v5.py --sid S1 --n 20 --out <scratchpad>` · EXPECT: таблица топ-10, файл только в scratchpad · EVIDENCE: pending
- G4 `docs/tables/` не тронут. CHECK: `git status --short docs/tables` · EXPECT: пусто · EVIDENCE: pending
- G5 таблица «есть / нет в быстром слое» с листами — ручной гейт: в отчёте.
