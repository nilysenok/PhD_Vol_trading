# GATES — тестировщик PR #722

## G1 — мутации (а)–(е) прогнаны в отдельной копии, откат после каждой
CHECK: таблица в описании PR (копия: scratchpad/mut722)
EXPECT: каждая мутация поймана хотя бы моим тестом
EVIDENCE: 8 мутаций; PR ловит 1 из 8 полностью (+ половину допуска); мои тесты ловят 8 из 8

## G2 — тесты
CHECK: NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/test_etalony_v6*.py tests/*perenos*.py tests/*ekviv*.py -p no:cacheprovider
EXPECT: passed, без failed/error
EVIDENCE: 161 passed, 1 warning (код 0)

## G3 — файлы автора не тронуты
CHECK: git diff --stat origin/etalony-v6 -- scripts tests/test_etalony_v6.py
EXPECT: пусто
EVIDENCE: пусто (код 0)
