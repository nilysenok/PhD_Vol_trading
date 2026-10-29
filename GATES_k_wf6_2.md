# GATES — К-wf6 лист 2 (правило процедуры, мутация № 42), kodir, 29.10

## G1 — на синтетике с пиком: правило процедуры даёт PBO <= 0,05, argmax >= 0,30
CHECK: python3 -m pytest tests/test_pbo_wf_pravilo.py -q -k "izvestny"
EXPECT: passed
EVIDENCE: passed (20 passed во всём наборе)

## G2 — мутация № 42 (argmax вместо правила) сдвигает PBO > 0,25
CHECK: python3 -m pytest tests/test_pbo_wf_pravilo.py -q -k "mutatsiya_42"
EXPECT: passed
EVIDENCE: passed; правило 0,0, argmax 0,456 (замер); первая версия синтетики давала 0,09 и порог не проходила

## G3 — PBO на шуме в среднем по 12 выборкам ≈ 0,5
CHECK: python3 -m pytest tests/test_pbo_wf_pravilo.py -q -k "shum"
EXPECT: passed
EVIDENCE: passed (среднее ~0,52; одна выборка 0,13–0,81)

## G4 — границы: край без соседей, ворота никто не прошёл, ничья
CHECK: python3 -m pytest tests/test_pbo_wf_pravilo.py -q -k "gran"
EXPECT: passed
EVIDENCE: passed

## Реестр
закрыто 4 из 4; ABANDON: 0; pending: 0
