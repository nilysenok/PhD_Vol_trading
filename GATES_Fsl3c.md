# GATES Ф-сл3c (ветка fast-v5-pir, стоит поверх #653): ось pir в kernel_v5

G1 CHECK: NUMBA_CACHE_DIR=<свежий> FASTBT_NO_CACHE=1 pytest tests/test_fast_v5_pir.py -k sintetika  EXPECT: passed (1200 путей побайтно; >= 200 с ногами pir, >= 20 с двумя добавками / урезанием / закрытой ногой)
EVIDENCE: passed.
G2 CHECK: то же -k realnye  EXPECT: passed (50 точек AFLT train4, >= 20 с ногами pir)
EVIDENCE: passed.
G3 CHECK: то же -k "krajnie or vne_granic"  EXPECT: passed
EVIDENCE: passed.
G4 CHECK: то же -k mutatsiya  EXPECT: 3 passed (эталон: 3ρ вместо 2ρ; добавка в баре входа; комиссия выхода на w)
EVIDENCE: passed. Ядро вручную: RISK_MAX 3.0 -> sintetika падает; комиссия (fs[W] - 0.0) вместо (fs[W] - tot) -> sintetika падает (свежий numba-кэш каждый раз).
G5 CHECK: pytest test_fast_v5_pir test_fast_v5_osi3 test_fast_v5_s34 test_fast_v5_pokryto test_fast_etap1_smoke  EXPECT: без регрессий
EVIDENCE: 477 passed, 6 skipped.
