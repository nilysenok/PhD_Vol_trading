# GATES — ставка ЦБ wf до 2026-02-28, kodir, 30.10.2026

Корень: `FASTBT_MAIN=<scratchpad>/stavka_koren`; новый ряд — `<scratchpad>/wf_dannye/stavka/key_rate_cbr.parquet` (побайтно равен файлу репозитория из #753/#774; исходный 3101 строка — `…/stavka/key_rate_cbr_iskhodnyj_0206.parquet`, коммит c22a4b36).

## G1 — префикс до 2026-02-06 побайтно = исходный
CHECK: python3 -m pytest tests/test_stavka_wf.py -k prefiks
EXPECT: passed
EVIDENCE: 1 passed

## G2 — значения 07–28.02.2026 по решениям с датами вступления (16,0 до 13.02; 15,5 с 16.02; 23.02 нет в hd_base)
CHECK: python3 -m pytest tests/test_stavka_wf.py -k znacheniya
EXPECT: passed
EVIDENCE: 1 passed

## G3 — мутация «ставка с дня объявления» ловится
CHECK: python3 -m pytest tests/test_stavka_wf.py -k "obyavlenie or vhod_pozzhe"
EXPECT: passed
EVIDENCE: 2 passed

## G4 — мутация «ffill вперёд без решения» ловится
CHECK: python3 -m pytest tests/test_stavka_wf.py -k ffill
EXPECT: passed
EVIDENCE: 1 passed

## G5 — rate_on на 2026-02-27: новый ряд 0,155; старый — отказ; dev — zona.check отказ, wf — открыт
CHECK: python3 -m pytest tests/test_stavka_wf.py -k zona
EXPECT: passed
EVIDENCE: 1 passed (всего файл: 6 passed; с tests/test_stavka.py — 20 passed)

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
