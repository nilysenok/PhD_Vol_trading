# Б3: NUMBA_NUM_THREADS меняется внутри процесса и роняет тесты — kodir, 29.10 16:10
Бюджет ~35м · ≤ 100т. Общая часть: `docs/zadaniya/2026-10-29_obshchee_16.md`. Ветка `fix-numba-izolyaciya`.

## Цель
Проверяющий #610: `pytest tests -k "exec or ispoln or fast"` без `NUMBA_NUM_THREADS=1` даёт 466 failed (`RuntimeError: Cannot set NUMBA_NUM_THREADS`); с переменной — 2 failed (давние: `test_07_corp_actions::test_split_by_hand`, `test_22_a_weights_by_year[2020-S3]`). Кандидаты: `fastbt/s9.py:457`, `:549`, `fastbt/pipeline.py:113` ставят `os.environ["NUMBA_NUM_THREADS"] = "1"` в процессе и не возвращают; `fastbt/legacy.py:42–57`, `fastbt/fast_signals.py:179–182` — восстанавливают. Сначала расследование: перечислить все гипотезы, воспроизвести минимальным тестом (два импорта/вызова в одном процессе), назвать причину. Затем правка: значение переменной после вызова — прежнее (или установка только в дочернем процессе), числа расчёта побайтно прежние. Класс ошибки — `mutations.md` № 43 (подмена, остающаяся после вызова).

## Владение
`fastbt/s9.py` (только строки установки переменной), `fastbt/pipeline.py` (то же), новый `tests/test_numba_threads_izolyaciya.py`. Остальные файлы — только чтение.

## Гейты (минимум)
G1 причина названа, тест воспроизводит `Cannot set NUMBA_NUM_THREADS` до правки. G2 после правки тест зелёный. G3 `pytest tests -k "exec or ispoln or fast" -p no:randomly` без переменной окружения: ≤ 2 failed, и это те же два давних (перечислить). G4 побайтность: якорь s9 / печать (тесты `tests/audit_fastbt/` и `tests/test_s9*.py`, что быстрее 5 мин) зелёные. G5 мутация «убрать восстановление переменной» роняет G1.
