# GATES — пакет переноса perenos/, kodir, 29.10 (составлен задним числом при продолжении; EVIDENCE — выводы запусков 15:19–15:39)

## G1 — сборщик собирает perenos/ одной командой
CHECK: scripts/server/sobrat_perenos.sh HEAD (PERENOS_OUT в scratchpad)
EXPECT: последняя строка OK perenos: ...; данных 201 файлов
EVIDENCE: OK perenos: ...p7/perenos; kod 0af65e88f3fd (tar sha256 300f5910c34f...); данных 201 файлов, 1,1G

## G2 — манифест: ≤ 2021-12-31, sha256 по файлу
CHECK: вывод шага 2 сборщика
EXPECT: OK manifest: 201 файлов, max_data ≤ 2021-12-31
EVIDENCE: OK manifest: 201 файлов, 1.136 ГБ, max_data 2021-12-31, sha256 и размеры совпадают

## G3 — файл с датой 2022 отвергается (тест, мутация поймана)
CHECK: python3 -m pytest tests/test_perenos_manifest_otkaz.py -q
EXPECT: passed; при подмене границы на 2099 — 1 failed
EVIDENCE: 10 passed; с мутацией `md > "2099-01-01"` — 1 failed (test_fajl_s_datoj_2022...)

## G4 — razvernut.sh: OK и версии numba/numpy/pandas
CHECK: ./razvernut.sh (локально, PERENOS_SYSTEM_SITE=1 PIP_NO_INDEX=1)
EXPECT: OK razvernut: ... numba 0.60.0 ... numpy 2.0.2 ... pandas 2.3.3
EVIDENCE: OK razvernut: python 3.9.6 (arm64), numba 0.60.0, llvmlite 0.43.0, numpy 2.0.2, pandas 2.3.3, ...

## G5 — proverka_yakor.sh: якорь A_fix и 200 точек побайтно
CHECK: ./proverka_yakor.sh
EXPECT: OK: якорь A_fix 6 Sk max |Δ| ≤ 1e-12; 200 точек побайтно
EVIDENCE: OK: якорь A_fix 6 Sk max |Δ| = 0.0e+00 ≤ 1e-12; 200 точек побайтно; 67 с

## G6 — без отметки якоря запуск отказывает
CHECK: mv proverka/OK ...; ./zapusk_sobol.sh train4
EXPECT: FAIL: нет отметки proverka/OK
EVIDENCE: FAIL: нет отметки proverka/OK — сначала ./proverka_yakor.sh (строка OK); счёт не запускаю

## G7 — zapusk_sobol.sh train4: чекпоинт и строка состояния (замена progon_sobol, S1, 50 точек)
CHECK: PERENOS_STRATEGII=S1 PERENOS_TOCHEK=50 ./zapusk_sobol.sh train4; ./zapusk_sobol.sh status train4
EXPECT: чекпоинт 1; S1: закрыто 50 из 50
EVIDENCE: чекпоинт 1: блок 0…4999 за 33 с, ошибок оценщика 0; S1: закрыто 50 из 50 (сборка a6b13531b98b; после правки под CLI v5 (ветка progon-server-v5) — не перезапускалось: диск, IG-6)

## G8 — zabrat.sh: dry-run (локальный источник)
CHECK: ./zabrat.sh $PWD ../zabrano
EXPECT: OK zabrat: N файлов (ledger/, tables/, logs/)
EVIDENCE: OK zabrat: 13 файлов в ../zabrano (ledger/, tables/, logs/)

## G9 — запуск через progon_server с кубом v5 (CLI --kub v5 --period --seed --k3)
CHECK: ./zapusk_sobol.sh train4 на коде с progon-server-v5, PERENOS_PROHODY=s1, малое N
EXPECT: чекпоинт и строка состояния по Sk
EVIDENCE: pending — зависит от слияния progon-server-v5 (ABANDON на этом шаге, см. ниже)

## G10 — инструкция docs/otchety/2026-10-29_server_perenos.md: шаги 1–6 с ожидаемым выводом
CHECK: grep -c '^## Шаг' docs/otchety/2026-10-29_server_perenos.md
EXPECT: 6
EVIDENCE: 6 (после написания)

## G11 — каталог проверки удалён после проверки
CHECK: du -sh scratchpad
EXPECT: удалён
EVIDENCE: не выполнено — хук запрещает rm -r; в scratchpad ≈ 9 ГБ (p1…p7)

## ABANDON
ABANDON: G9 зависит от progon-server-v5 (другой кодер); не выполнимо до слияния.
ABANDON: G11 хук v23 запрещает рекурсивное удаление; чистит дирижёр (каталог scratchpad сессии).
ABANDON: G7 повтор на финальном коммите — диск < 5,3 ГБ (IG-6), перезапуск невозможен.

## Реестр
закрыто 8 из 11; ABANDON: 3; pending: 0 → «передача», не «готово»
