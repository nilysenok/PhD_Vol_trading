# GATES — Л9 оценщик точки куба v5, kodir, 29.10

## G1 — v5 на якорях = v4 ≤ 1e-12
CHECK: python3 -m pytest tests/test_eval_v5.py -k yakor_v5_ravno_v4 -q
EXPECT: passed
EVIDENCE: tests/test_eval_v5.py целиком — 51 passed вместе с G4 (29.10)

## G2 — мутации (ось не дошла до оценщика; фильтр на дне d+1) ловятся
CHECK: python3 -m pytest tests/test_eval_v5.py -k "mutaciya" -q
EXPECT: passed
EVIDENCE: в составе G4: passed

## G3 — каждая ось куба v5 доходит до разбора и меняет результат
CHECK: python3 -m pytest tests/test_eval_v5.py -k "kazhdaya_os or novaya_os" -q
EXPECT: passed
EVIDENCE: в составе G4: passed

## G4 — три набора тестов зелёные
CHECK: python3 -m pytest tests/test_eval_v5.py tests/test_progon_sobol_v5.py tests/test_progon_sobol.py -q
EXPECT: 0 failed
EVIDENCE: 51 passed in 256.51s

## G5 — замер с/точку, 1 процесс, train4
CHECK: python3 scripts/zamer_eval_v5.py --n 8
EXPECT: печатает с/точку (медиана, среднее), ошибок, вне бара
EVIDENCE: S1 0.60, S2 0.60, S3 2.33, S4 1.04, S5 0.61, S6 0.56 с/точку (медиана); ошибок 0; вне бара 0

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
