# GATES — прогрев Морриса v7 (canon_v6 нет в canon_v7), kodir, 31.10.2026

## G1 — тест плана Морриса под каноном v7 (S1, S8, S10) падает на базе
CHECK: python3 -m pytest tests/test_pilot_zapusk_realno.py -q -k plan   (на коммите теста до правки)
EXPECT: failed (AttributeError canon_v6)
EVIDENCE: pending

## G2 — после правки тот же тест зелёный
CHECK: python3 -m pytest tests/test_pilot_zapusk_realno.py -q -k plan
EXPECT: passed
EVIDENCE: pending

## G3 — прогрев (progon_morris_v7 --period train4 --r 1 --procs 1) на S1, S8, S10, код 0
CHECK: python3 -m fastbt.progon_morris_v7 --kub v7 --period train4 --out <tmp> --procs 1 --r 1 --strategii S1 (S8, S10)
EXPECT: код 0 для каждой
EVIDENCE: pending

## G4 — пилот (progon_v7 --pilot --n1 2 --procs 1) на S1, S8, S10, код 0
CHECK: python3 -m fastbt.progon_v7 --kub v7 --period train4 --pilot --n1 2 --semeystva S1 --procs 1 --out <tmp>
EXPECT: код 0
EVIDENCE: pending

## G5 — мутации (адаптер не отдаёт v7-ось / S8 идёт каноном S1) ловятся
CHECK: pytest tests/test_pilot_zapusk_realno.py -q
EXPECT: passed, мутации падают
EVIDENCE: pending

## G6 — регрессии: pytest tests/ -q -x -k "v6 or v7 or eval or kub or canon" (<= 4 процесса)
EXPECT: 0 failed
EVIDENCE: pending

## Реестр
закрыто 0 из 6; ABANDON: 0; pending: 6
