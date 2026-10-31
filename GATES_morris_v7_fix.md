# GATES — прогрев Морриса v7 (canon_v6 нет в canon_v7), kodir, 31.10.2026

## G1 — тест плана Морриса под каноном v7 (S1, S8, S10) падает на базе
CHECK: python3 -m pytest tests/test_pilot_zapusk_realno.py -q (коммит 2ff83ea0, до правки)
EXPECT: failed
EVIDENCE: 3 failed (S1: AttributeError canon_v6; S8, S10: FileNotFoundError kub v6)

## G2 — после правки тест зелёный
CHECK: pytest tests/test_pilot_zapusk_realno.py -q (все 12, с данными train4)
EXPECT: passed
EVIDENCE: 12 passed in 242.71s

## G3 — прогрев train4 --r 1 --procs 1, S1/S8/S10, код 0
CHECK: python3 -m fastbt.progon_morris_v7 --kub v7 --period train4 --procs 1 --r 1 --strategii <S>
EXPECT: rc 0; S8, S10 — oshibka 0
EVIDENCE: S1 rc0 7s (52 oshibka — только «фон:», граница train4); S8 rc0 37s ok 6/6; S10 rc0 154s ok 16/16

## G4 — пилот --pilot --n1 2 --procs 1 --fast, S1/S8/S10, код 0
CHECK: python3 -m fastbt.progon_v7 --kub v7 --period train4 --pilot --n1 2 --semeystva <S> --procs 1 --fast
EXPECT: rc 0, sboy null, статус строк ok
EVIDENCE: S1 n=1 ok; S8 n=2 ok; S10 n=1 ok (тест test_pilot_realno)

## G5 — мутации ловятся
CHECK: GOTOVY_V7 снят в Kanon7.osi; снят `vhod == "T"` в pokryto
EXPECT: тест падает
EVIDENCE: test_plan_morris_pod_kanonom_v7[S1] failed; test_vhod_T_ne_idyot_v_bystryj_sloj failed

## G6 — регрессии узким набором (Ж-160 п. 1)
CHECK: pytest test_progon_v7 test_progon_v7_pilot1 test_pilot1_paket test_morris_v6_plan(_mut) test_fast_etap1_smoke; test_fast_etap1_equiv test_fast_v6_osi
EXPECT: 0 failed
EVIDENCE: 518 passed 16.5s; 29 passed 123s

## Реестр
закрыто 6 из 6; ABANDON: 0; pending: 0 (полный -k "v6 or v7 …" на Маке запрещён Ж-160 — не гонялся)
