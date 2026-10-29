# GATES — серверный запуск Соболя (Ж-131 п. 3–5), kodir, 29.10.2026

## G1 — оба новых файла компилируются
CHECK: python3 -m py_compile fastbt/progon_server.py scripts/server/svodka_ledger.py && echo ok
EXPECT: ok
EVIDENCE: ok

## G2 — прогон обёртки = обычный progon_sobol.schet (канонический вид ledger побайтно)
CHECK: python3 -m pytest -q tests/test_progon_server.py -k ravno_progon_sobol
EXPECT: 1 passed (параметризован: obshchie/prezhniy × без poz / с poz Л7b)
EVIDENCE: 4 passed, 8 deselected in 2.63s

## G3 — обрыв после 1-го чекпоинта + --prodolzhit = непрерванный прогон побайтно
CHECK: python3 -m pytest -q tests/test_progon_server.py -k obryv
EXPECT: 1 passed
EVIDENCE: 1 passed, 11 deselected in 1.20s

## G4 — мутация «чекпоинт не проверяет хэш куба» ловится; отказ при чужом kod/vhody/seed
CHECK: python3 -m pytest -q tests/test_progon_server.py -k "mutaciya or otkaz"
EXPECT: все passed (≥ 2)
EVIDENCE: 3 passed, 9 deselected in 1.15s

## G5 — сбой в процессе S1 не меняет каталог S2
CHECK: python3 -m pytest -q tests/test_progon_server.py -k sboy
EXPECT: 1 passed
EVIDENCE: 1 passed, 11 deselected in 1.15s

## G6 — fork-путь не копирует массивы (fabrika только в родителе; адрес буфера и id оценщика в детях = родительские)
CHECK: python3 -m pytest -q tests/test_progon_server.py -k fork
EXPECT: 1 passed
EVIDENCE: 2 passed (test_fork_obshchie_bez_kopii + test_manifest_iz_roditelya_do_fork), 10 deselected in 1.34s; мутация M1 (fabrika не подменена) — 2 failed, мутация M2 (manifest не из родителя) — 1 failed, обе откатаны

## G7 — сводка по ledger: точки, сверка хэшей, top-N, объём, время одной таблицей; порча шарда видна
CHECK: python3 -m pytest -q tests/test_progon_server.py -k svodka
EXPECT: 1 passed
EVIDENCE: 1 passed, 11 deselected in 0.94s

## G8 — period v2 отказывает явно (драйвер и ledger 2.1 — только train4)
CHECK: python3 -m pytest -q tests/test_progon_server.py -k v2
EXPECT: 1 passed
EVIDENCE: 1 passed, 11 deselected in 0.77s

## G9 — соседние тесты драйвера не сломаны
CHECK: python3 -m pytest -q tests/test_progon_sobol.py
EXPECT: все passed
EVIDENCE: 13 passed in 1.64s (вместе с tests/test_ledger_v2.py и новым файлом — 42 passed)

## G10 — размеры: файлы ≤ 200 строк
CHECK: wc -l fastbt/progon_server.py scripts/server/svodka_ledger.py tests/test_progon_server.py
EXPECT: каждый ≤ 200
EVIDENCE: 200 / 94 / 200

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## Реестр
закрыто 10 из 10; ABANDON: 0; pending: 0
Стабильность: 25 подряд прогонов tests/test_progon_server.py — 0 падений (после фикса manifest и заглушки диска в фикстуре).
