# GATES — Б3 NUMBA_NUM_THREADS изоляция, kodir, 29.10

Причина: numba после запуска потоков (первое parallel-ядро) отказывает на любой новой компиляции, если NUMBA_NUM_THREADS в окружении изменился (`config.reload_config` -> RuntimeError). `s9._job`, `s9.build_cache`, `pipeline.run_all` ставили "1" в процессе и не возвращали (mutations.md № 43).

## G1 — тест воспроизводит до правки
CHECK: pytest tests/test_numba_threads_izolyaciya.py -p no:randomly (код до правки)
EXPECT: failed
EVIDENCE: 7 failed (assert '1' == '4' / '1' == None; сквозной: RuntimeError Cannot set NUMBA_NUM_THREADS)

## G2 — после правки зелёный
CHECK: pytest tests/test_numba_threads_izolyaciya.py -p no:randomly
EXPECT: passed
EVIDENCE: 7 passed

## G3 — широкий прогон без переменной
CHECK: pytest tests -k "exec or ispoln or fast" -p no:randomly
EXPECT: <= 2 failed, давние
EVIDENCE: 2 failed, 916 passed, 1 skipped (test_07_corp_actions::test_split_by_hand, test_22_a_weights_by_year[2020-S3]::test_a_weights_only_before_Y)

## G4 — побайтность
CHECK: pytest tests/audit_fastbt tests/test_s9*.py -p no:randomly
EXPECT: те же два давних
EVIDENCE: 2 failed, 349 passed, 2 skipped (на базе без правки — те же 2). Один из прогонов дал ещё 2 падения в test_s9_scope (идут при хуке, параллельном прогоне); одиночно и повторно — зелёные.

## G5 — мутация «убрать восстановление» роняет G1
CHECK: finally: pass в s9._threads1
EXPECT: failed
EVIDENCE: 5 failed, 2 passed

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
