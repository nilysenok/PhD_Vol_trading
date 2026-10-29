# GATES: карта ТФ на В2 + срез утечки 5а по стратегиям (Ж-130)

G1 train4 побайтно. CHECK: `FASTBT_MAIN=$PWD TF_MAIN=<осн.> python3 scripts/revers/r1_g.py --period train4 --out <scratch>/tf_train4.csv; cmp <scratch>/tf_train4.csv docs/tables/progon_e1/timeframes_golaya.csv` · EXPECT: код 0 · EVIDENCE: CMP_OK (29.10)
G2 v2 посчитан, шапка со статусом/коммитом/хэшами. CHECK: `head -3 docs/tables/progon_e1/timeframes_v2.csv; grep -vc '^#' docs/tables/progon_e1/timeframes_v2.csv` · EXPECT: статус, коммит, 150 строк + шапка столбцов = 151 · EVIDENCE: 17 хэшей, 151 строка без комментариев (170 всего)
G3 покрытие 10-минуток объявлено. CHECK: абзац «Данные и ранние бары» в docs/otchety/2026-10-29_karta_tf_v2.md · EXPECT: дата начала 2014-06-09, решение по отсутствующим барам · EVIDENCE: есть (дата 2014-06-09, ничего не достраивается)
G4 тесты v2 и мутация «год прогрева в метриках». CHECK: `pytest -q tests/test_revers_r1_g_v2.py tests/test_revers_r1_g.py` · EXPECT: passed · EVIDENCE: 6 passed (в наборе 16 passed вместе с 5а)
G5 срез по Sk, ворота полный шаг = P1 из sverka_5a.csv ≤ 1e-3. CHECK: `FASTBT_MAIN=$PWD FASTBT_DATA_ROOT=<осн.>/data/fastbt/daily python3 scripts/revers/r1_sverka_5a_po_sk.py` · EXPECT: «ворота: пройдены» · EVIDENCE: пройдены
G6 основной вывод sverka_5a.csv не изменился. CHECK: `r1_sverka_5a.py svod --out <scratch>/sverka_5a_new.csv`, cmp без строк `#` и cmp строк `# вход` · EXPECT: идентично · EVIDENCE: SVOD_IDENTICAL, HASH_IDENTICAL
G7 тесты среза и мутации. CHECK: `pytest -q tests/test_revers_r1_sverka_5a_po_sk.py tests/test_revers_r1_sverka_5a.py` · EXPECT: passed · EVIDENCE: 10 passed
G8 py_compile. CHECK: `python3 -m py_compile` трёх .py · EXPECT: 0 · EVIDENCE: PYC_OK
