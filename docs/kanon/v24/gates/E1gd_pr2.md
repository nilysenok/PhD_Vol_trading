# GATES — Э1-г/д, PR 2: сборки A (`fastbt/sweep_a.py`, `sweep_a_core.py`), kodir T1, 27.10

## G1 — эталон (равные, net, net) совпадает с ручным расчётом A (вес 1/cnt_d, комиссия с обеих сторон, плата за выход на последнем баре, ставка на свободный капитал)
CHECK: python3 -m pytest tests/test_sweep_a.py -k "etalon" -q
EXPECT: passed
EVIDENCE: 1 passed

## G2 — веса обратной волатильности и паритета риска не зависят от текущего дня; известный ответ: диагональная ковариация -> 1/σ
CHECK: python3 -m pytest tests/test_sweep_a.py -k "ves" -q
EXPECT: passed
EVIDENCE: 3 passed

## G3 — лимит позиций L и лимит на стратегию K соблюдаются в каждый день; FIFO и сила сигнала при равенстве
CHECK: python3 -m pytest tests/test_sweep_a.py -k "limit" -q
EXPECT: passed
EVIDENCE: 4 passed

## G4 — 36 сборок через factorial_sweep, N в ledger = 36, топ-2 по цели setup, строки setup/control × 0,04/0,035
CHECK: python3 -m pytest tests/test_sweep_a_run.py -q
EXPECT: passed
EVIDENCE: 6 passed (в одном прогоне с G5)

## G5 — экспорт A′ в формате main.yaml (ключи main.yaml сохранены)
CHECK: python3 -m pytest tests/test_sweep_a_run.py -k "eksport" -q
EXPECT: passed
EVIDENCE: 1 passed

## G6 — мутации (8): окно весов с текущим днём, лимит не соблюдается, нет платы за выход, комиссия с одной стороны, сила наоборот, нет FIFO, σ вместо 1/σ, сетка не 36 — ловятся 8 из 8
CHECK: python3 -m pytest tests/test_sweep_a_mut.py -q
EXPECT: passed
EVIDENCE: 8 passed

## G7 — тесты движка не сломаны
CHECK: python3 -m pytest tests/test_sweep_engine.py tests/test_sweep_mutacii.py tests/test_sweep_pbo.py -q
EXPECT: passed
EVIDENCE: 41 passed

## Реестр
закрыто 7 из 7; ABANDON: 0; pending: 0
Не проверено на реальных данных: `SkPanelSource` и `load_point` по настоящим `S{k}_top3.csv` (в базе PR их нет; прогон — дирижёр после итераций).
