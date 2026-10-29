# GATES — К-wf2b, kodir, 29.10

## G1 — wf без отметки замка: отказ; с отметкой (tmp): пул строится
CHECK: python3 -m pytest tests/test_kalibr_wf2b.py -q
EXPECT: 8 passed
EVIDENCE: 8 passed (без мутации); с мутацией «охрана пропущена» — 1 failed (test_wf_bez_otmetki_otkaz)

## G2 — прежняя охрана (train4/v2, test_kalibr_v5.py::test_daty_2020_otkaz) цела
CHECK: python3 -m pytest tests/test_kalibr_v5.py tests/test_kalibr_wf2.py tests/test_kontroli_v5.py tests/test_period_wf.py -q
EXPECT: все passed
EVIDENCE: 83 passed вместе с G1 и kalibr_v5_21/doschet/golos_v2

## G3 — PERIODS["wf"] = (2015, 2026)
CHECK: тот же файл, test_periods_wf_v_etap1_eval
EXPECT: passed
EVIDENCE: passed

## Реестр
закрыто 3 из 3
