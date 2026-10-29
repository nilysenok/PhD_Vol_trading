# GATES — V6-9 time_decay (движок), 29.10.2026

## G1 — тесты оси проходят (в т.ч. мутации ловятся)
CHECK: NUMBA_CACHE_DIR=$(mktemp -d) python3 -m pytest tests/test_exec_e_time_decay.py -q
EXPECT: все passed
EVIDENCE: pending

## G2 — тест падает до правки (поля Params нет)
CHECK: тот же тест при откатанных fastbt/exec_e_run.py и exec_e_types.py
EXPECT: failed
EVIDENCE: pending

## G3 — «выкл.» = v5: соседние тесты движка и быстрого слоя зелёные
CHECK: python3 -m pytest tests/test_exec_e.py tests/test_exec_e2.py tests/test_v6_a.py tests/test_fast_v5_osi3.py -q
EXPECT: все passed
EVIDENCE: pending

## G4 — py_compile, fmt, clippy-аналог
CHECK: python3 -m py_compile fastbt/exec_e_run.py fastbt/exec_e_types.py tests/test_exec_e_time_decay.py
EXPECT: код 0
EVIDENCE: pending

## Реестр
закрыто 0 из 4
