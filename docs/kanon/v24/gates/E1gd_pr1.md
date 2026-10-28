# GATES — Э1-г/д, PR 1: excess_d, daty, PBO по ledger (kodir T1, 27.10)

## G1 — строка evaluate отдаёт excess_d и daty; Шарп из excess_d == sharpe строки
CHECK: python3 -m pytest tests/test_sweep_pbo.py -k "excess_d_sovpadaet" -q
EXPECT: passed
EVIDENCE: 1 passed

## G2 — ledger хранит excess_d/daty для пробы только на setup при 0,04 (не на control/train/0,035)
CHECK: python3 -m pytest tests/test_sweep_pbo.py -k "ledger_excess" -q
EXPECT: passed
EVIDENCE: 2 passed

## G3 — PBO через ledger на шуме ≈ 0,5 (среднее по 12 наборам), на сильном сигнале < 0,2
CHECK: python3 -m pytest tests/test_sweep_pbo.py -k "pbo_shum or pbo_signal" -q
EXPECT: passed
EVIDENCE: 2 passed

## G4 — мутации: знак excess_d, запись на любом периоде, нет контроля дат, инверсия матрицы, постоянная матрица — ловятся (5 из 5)
CHECK: python3 -m pytest tests/test_sweep_pbo_mut.py -q
EXPECT: passed
EVIDENCE: 5 passed

## G5 — существующие тесты движка не сломаны
CHECK: python3 -m pytest tests/test_sweep_engine.py tests/test_sweep_post_ledger.py tests/test_sweep_mutacii.py tests/test_sweep_drivers.py tests/test_sweep_iter.py -q
EXPECT: passed, без failed
EVIDENCE: 72 passed

## G6 — контракт: excess_d и daty в таблице реализованных ключей
CHECK: grep -n "excess_d" docs/kanon/v24/kontrakty/sweep_evaluate.md
EXPECT: строка в таблице ключей; в «v1.1, не реализовано» excess_d/daty нет
EVIDENCE: строки 22 и 25; в v1.1 остались v, okno_s, okno_po, chestnaya

## Реестр
закрыто 6 из 6; ABANDON: 0; pending: 0
