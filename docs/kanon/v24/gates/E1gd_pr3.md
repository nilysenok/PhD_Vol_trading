# GATES — Э1-г/д, PR 3: validation A′ (`fastbt/sweep_val.py`, Ж-037), kodir T1, 27.10

Реальный прогон на 2020–2021 не запускался и не запускается: его запускает лид после «да» владельца.

## G1 — отказ без «да», при чужом sha списка, при «да» не позже заморозки: ни замка, ни реестра, ни каталога, ни чтения данных
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val.py -k "otkaz" -q
EXPECT: passed
EVIDENCE: 3 passed (+ test_realnyy_progon_bez_dry_otkaz)

## G2 — замок стоит до чтения (перехват read_parquet, read_table, read_metadata, np.fromfile)
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val.py -k "zamok" -q
EXPECT: passed
EVIDENCE: 1 passed

## G3 — реестр не создаётся до взгляда; под замком — «начато», затем «просмотр состоялся» (время, sha, Ж-номер)
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val.py -k "reestr" -q
EXPECT: passed
EVIDENCE: 1 passed

## G4 — сбой посередине: каталог удалён без чтения, реестр «прервано до просмотра», результатов нет
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val.py -k "sboy" -q
EXPECT: passed
EVIDENCE: 1 passed

## G5 — повтор без изменения — отказ; со записанным изменением — один, N×2; третий — отказ
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val.py -k "povtor" -q
EXPECT: passed
EVIDENCE: 1 passed

## G6 — к2 с лагом 0 = исходный ряд (№ 9); Δ = разность табличных Шарпов (№ 13); ворота; решение «исходная версия»
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val.py -k "k2 or delta or vorota or reshenie" -q
EXPECT: passed
EVIDENCE: 4 passed

## G7 — мутации: 9 из 9 ловятся
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_val_mut.py -q
EXPECT: passed
EVIDENCE: 9 passed

## G8 — `--dry` на синтетике работает; без `--dry` — отказ
CHECK: FASTBT_MAIN=$PWD python3 -m fastbt.sweep_val --dry
EXPECT: таблица кандидатов и решение по группам, код 0
EVIDENCE: таблица из 4 кандидатов (S1:1–3, A:1), решение {'S1': 'ishodnaya', 'A': 'ishodnaya'} (на шуме никто не прошёл — как и должно быть)

## G9 — PR 1–2 не сломаны
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_a.py tests/test_sweep_a_run.py tests/test_sweep_a_mut.py tests/test_sweep_pbo.py tests/test_sweep_engine.py tests/test_sweep_mutacii.py -q
EXPECT: passed
EVIDENCE: 92 passed вместе с G1–G7

## Реестр
закрыто 9 из 9; ABANDON: 0; pending: 0
Не сделано: фабрика реальных кандидатов (топ-3 из `S{k}_top3.csv`, ledger итераций, топ-2 сборки, `SkPanelSource`) — нужна, когда будут таблицы итераций и список кандидатов; без неё `main` без `--dry` отказывает.
