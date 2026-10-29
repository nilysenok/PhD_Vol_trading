# GATES — Ф-open2 (ядро быстрого слоя знает t_vyhod = open2), ветка fast-open2, 29.10.2026

Свежий `NUMBA_CACHE_DIR` у каждой мутации. Данные ≤ 2021 (train4), ≤ 1 процесса.

## G1 — 10 красных тестов pokryto зелёные
CHECK: pytest tests/test_fast_v5_pokryto.py tests/test_fast_v5_s34_pokryto.py -q
EXPECT: 0 failed
EVIDENCE: 22 passed, 10 skipped (skipped — ekviv без FASTBT_V5_N), 0 failed

## G2 — синтетика ≥ 1000 путей побайтно (Result вкл. x_t)
CHECK: pytest tests/test_fast_open2.py -q
EXPECT: 6 passed; покрытие: x+1-выходов ≥ 150, срок ≥ 150, шорт ≥ 150
EVIDENCE: 6 passed; покрытие 1200 путей: open x+1 — 395, срок — 261, шорт-сигнал — 306, сигнал на последнем дне — 58

## G3 — 200 точек S1, S2, S5, S6 побайтно (данные ≤ 2021)
CHECK: FASTBT_MAIN=$PWD FASTBT_V5_N=200 pytest tests/test_fast_v5_pokryto.py -q -k "ekviv and (S1 or S2 or S5 or S6)"
EXPECT: 4 passed
EVIDENCE: 4 passed, 11 deselected in 328 s (S1, S2, S5, S6 по 200 точек, быстрый = медленный побайтно, один процесс)

## G4 — мутации ловятся (свежий кэш numba)
CHECK: подмена в fast_exec.py: (а) P_TV = 0 / (б) τ → PEND / (в) выход по open в день сигнала; pytest tests/test_fast_open2.py
EXPECT: тесты падают
EVIDENCE: (а) test_mutatsiya_yadro_bez_open2_lovitsya (внутри теста, зелёный = мутация поймана); (б) τ по open x+1: упали sintetika, izvestnyj_otvet, shorty; (в) выход по open x в день сигнала: упали те же три

## G5 — замер с/точку через оценщик v5, 20 точек
CHECK: scratchpad/fast_open2_zamer.py 20 S1 S5
EXPECT: быстрый < эталон; ядро вызвано с open2
EVIDENCE: S1: эталон 0,582 → ядро 0,184 с/точку (×3,2); S5: 0,496 → 0,190 (×2,6); вызовов run_v5 с open2: 68 из 68

## G6 — регрессии быстрого слоя
CHECK: pytest test_fast_open2 smoke s34 osi3 pir pokryto s34_pokryto vyhod_tau_open2 kub_v5_open2 exec_e2
EXPECT: 0 failed
EVIDENCE: 545 passed, 10 skipped
