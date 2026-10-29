# GATES — P-2 (wf в переносе), kodir, 29.10

## G1 — wf: 80 процессов по умолчанию, переменная, потолок ядра−4
CHECK: python3 -m pytest tests/test_perenos_wf_sh.py -q
EXPECT: 5 passed (до правки: 4 failed)
EVIDENCE: 5 passed; на старом obshchee.sh — 4 failed, 1 passed

## G2 — период wf в zapusk_sobol.sh, отказ без отметки владельца
CHECK: python3 -m pytest tests/test_perenos_wf_sh.py -q -k "otmetki or period_wf"
EXPECT: passed
EVIDENCE: входит в G1 (5 passed)

## G3 — манифест: --do, отметка, умолчание 2021-12-31; мутация «отметка не проверяется» ловится
CHECK: python3 -m pytest tests/test_perenos_wf_manifest.py tests/test_manifest_perenos.py tests/test_perenos_manifest_otkaz.py -q
EXPECT: passed; мутация (`if False and d > GRANICA`) -> test_sborshchik_bez_otmetki_otkaz_do_chteniya падает
EVIDENCE: 27 passed; на мутации 1 failed, 7 passed (из 8)

## G4 — акт снятия замка печатает, ничего не снимает
CHECK: python3 -m pytest tests/test_perenos_wf_manifest.py -q -k akt
EXPECT: 2 passed
EVIDENCE: входит в G3

## G5 — svodka_ledger и svod_server_v5 на синтетике wf
CHECK: python3 -m pytest tests/test_perenos_wf_svod.py -q
EXPECT: 2 passed, 1 xfailed (xfail strict = находка: pul_iz_ledger не передаёт period=wf, после К-wf2b #645)
EVIDENCE: 2 passed, 1 xfailed. Свод wf БЕЗ отказа достигнут только с пулом period=wf; сам по себе отказывает — вне владения P-2

## G6 — сбор test_svod_k1_prefiks.py вне корня; падал до правки
CHECK: python3 -m pytest tests/test_perenos_sbor_iz_drugogo_kataloga.py tests/test_svod_k1_prefiks.py tests/test_svod_server_v5.py -q
EXPECT: 15 passed (до правки: сбор падал ModuleNotFoundError: tests)
EVIDENCE: 15 passed; до правки 1 failed (ModuleNotFoundError)

## G7 — fmt/сводные проверки
CHECK: python3 -m py_compile scripts/server/akt_snyatiya_zamka.py scripts/server/manifest_perenos.py scripts/server/manifest_proverka.py
EXPECT: 0
EVIDENCE: код 0 (cargo xtask в проекте нет — CLAUDE.md «Не применяется»)

## Реестр
закрыто 7 из 7 по тестам; ABANDON: 0; частично: G5 (свод wf на реальном pul_iz_ledger — xfail, правка вне владения)
