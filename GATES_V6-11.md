# GATES V6-11 (тренд: перенос осей v5 на стоп-заявку; PR a — единицы/Chandelier, b — безубыток/«в плюс»/ступени/пирамидинг, c — reentry_den)
G1 CHECK: python3 -m pytest tests/test_exec_e_stop_osi.py -q  EXPECT: все passed  EVIDENCE: 18 passed
G2 «выкл.» = v5 побайтно: pytest tests/test_exec_e_stop_osi.py -q -k vykl_ravno  EXPECT: passed  EVIDENCE: 4 passed, 14 deselected
G3 гэп через стоп — по open; цена в [low; high]: -k "gep or low_high"  EXPECT: passed  EVIDENCE: 2 passed, 16 deselected
G4 мутации ловятся: -k mutaciya  EXPECT: passed  EVIDENCE: 2 passed, 16 deselected
G5 быстрый слой: новая ось → эталон: -k etalonom  EXPECT: passed  EVIDENCE: 1 passed, 17 deselected
G6 соседи: pytest tests/test_exec_e*.py tests/test_stop_zayavki.py tests/test_fast_v5_s34_pokryto.py tests/test_fast_etap1_smoke.py -q  EXPECT: без failed  EVIDENCE: 688 passed, 4 skipped
