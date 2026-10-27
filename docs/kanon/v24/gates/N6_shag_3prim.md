# GATES — Ж-029 Н6: пересчёт шага 3′ (D и B) на дневном ряде до 18:50, v3.2, kodir, 27.10

## G1 — прежние shag3D_*/shag3B_* в манифесте v3_1_manifest_3prim.txt (путь, SHA-256, git show), 39 строк
CHECK: grep -vc '^#' docs/tables/strat_dev_v3/v3_1_manifest_3prim.txt
EXPECT: 39
EVIDENCE: pending (папка arhiv_* закрыта хуками; решение дирижёра, класс B: архив — git)

## G2 — самопроверка скрипта без аргументов проходит
CHECK: FASTBT_MAIN=$PWD python3 scripts/analysis/s9_shag_db.py
EXPECT: код возврата 0
EVIDENCE: pending

## G3 — D пересчитан: 3 журнала с проверкой сдвига первой строкой, таблицы с пометкой v3.2
CHECK: head -qn1 docs/tables/strat_dev_v3/shag3D_zhurnal_*.txt; grep -c "v3.2 (Н6" docs/tables/strat_dev_v3/shag3D_otbor_naive.csv
EXPECT: 3 строки проверки сдвига, не «НЕ ПРОЙДЕНА»; счётчик > 0
EVIDENCE: pending

## G4 — B пересчитан по naive|harj|hyb, то же
CHECK: head -qn1 docs/tables/strat_dev_v3/shag3B_zhurnal_*.txt; ls docs/tables/strat_dev_v3/shag3B_itog.csv
EXPECT: 3 строки проверки сдвига; shag3B_itog.csv есть
EVIDENCE: pending

## G5 — n6_bylo_stalo_3prim.csv: воронка 1/2/3/С8 v2 по каждой σ для D и B, лучшая точка, Шарп при 0,04, число прошедших
CHECK: wc -l docs/tables/strat_dev_v3/n6_bylo_stalo_3prim.csv
EXPECT: строки D×6 рядов и B×6 рядов (было и стало)
EVIDENCE: pending

## G6 — отчёты 3D и 3B: шапка v3.2, «что изменилось»; при прошедшей точке — первая строка «ПРОШЛА ТОЧКА»
CHECK: grep -c "v3.2" docs/otchety/2026-10-27_gl4_shag_3D.md docs/otchety/2026-10-27_gl4_shag_3B.md
EXPECT: > 0 в обоих
EVIDENCE: pending

## G7 — shag{0..4}_*, test_locked, данные ≥ 2022 не тронуты
CHECK: git diff --stat origin/kanon-v22-pereschet -- 'docs/tables/strat_dev_v3/shag[0-4]_*' 'fastbt/tests/test_core.py'
EXPECT: пусто
EVIDENCE: pending

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## СТАТУС WIP (Ж-030, 18:13): шаги 1–4 v3 приостановлены. Готово: манифест (G1), B пересчитан целиком (naive, harj, hyb), D — naive и harj; D hyb снят по pid, его файлы остались v3.1. Воронки и отчёты не сделаны. Гейты G2–G7 не закрыты.

## Реестр
закрыто 0 из 7; ABANDON: 0; pending: 7 (WIP, передача)
