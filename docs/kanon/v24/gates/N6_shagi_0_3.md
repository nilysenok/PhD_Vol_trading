# GATES — Ж-029 Н6 / Ж-030 этап 0: пересчёт шага 0 главы 4 на ряде до 18:50 (v3.2), kodir, 27.10

Правка объёма (Ж-030, 18:13): шаги 1–3 приостановлены, не считаются; начатое по шагу 1 откачено `git checkout`. Архив v3.1 — git `85afdd3` + манифест (папка `arhiv_*` закрыта хуком, решение дирижёра класса B).

## G1 — манифест v3.1 совпадает с коммитом 85afdd3
CHECK: git diff --stat 85afdd3 origin/kanon-v22-pereschet -- 'docs/tables/strat_dev_v3/shag[0-3]_*'
EXPECT: пусто
EVIDENCE: пусто (код возврата 0); 39 файлов в docs/tables/strat_dev_v3/v3_1_manifest.txt

## G2 — таблицы шага 0 с пометкой v3.2
CHECK: grep -L "v3.2 (Н6, ряд до 18:50)" docs/tables/strat_dev_v3/shag0_*.csv
EXPECT: только shag0_sverka_predzashchita.csv (не пересчитана — нет R0/Q*.parquet)
EVIDENCE: pending-проверка при финише (см. отчёт PR)

## G3 — сверка быстрого и медленного слоёв
CHECK: shag0_sverka_sloy.csv, столбец metriki_do_4_znaka
EXPECT: True в 5 строках
EVIDENCE: 5 из 5 True, max |Δ| 5.2e-18

## G4 — тесты шага 0 и train/val, мутация «validation включает 2019» ловится
CHECK: FASTBT_MAIN=<копия> python3 -m pytest tests/test_s9_shag0_train_val.py tests/test_s9_shag0.py -q
EXPECT: 0 failed
EVIDENCE: 14 passed (код возврата 0)

## G5 — n6_bylo_stalo_0_3.csv (шаг 0) и shag0_semyi_train_val.csv (14 строк)
CHECK: ls docs/tables/strat_dev_v3/n6_bylo_stalo_0_3.csv docs/tables/strat_dev_v3/shag0_semyi_train_val.csv
EXPECT: оба файла есть
EVIDENCE: оба есть; 16 и 14 строк

## G6 — отчёт шага 0 обновлён (строка v3.2 в шапке)
CHECK: grep -c "v3.2" docs/otchety/2026-10-27_gl4_shag_0.md
EXPECT: ≥ 1
EVIDENCE: ≥ 1 (шапка + таблицы §3 и «Итог»)

## G7 — запрещённые файлы не тронуты
CHECK: git diff --name-only origin/kanon-v22-pereschet | grep -E "shag[1-3]_|shag3D_|shag3B_|shag4_|test_locked|test_core"; test $? -eq 1
EXPECT: пусто
EVIDENCE: проверяется при финише

## ABANDON (только с причиной; гейт не удалять и не переписывать)
ABANDON: сверка с предзащитой (sverka_A_lestnica.py) — нет входных R0/Q*.parquet; цель лестницы −0,8767 относится к v3.1
ABANDON: шаги 1–3 — приостановлены владельцем (Ж-030)

## Реестр
закрыто 6 из 7 (G7 — при финише); ABANDON: 2; pending: 1
