# GATES V6-11 (тренд: перенос осей v5 на стоп-заявку; PR a — единицы/Chandelier, b — безубыток/«в плюс»/ступени/пирамидинг, c — reentry_den)
G1 CHECK: python3 -m pytest tests/test_exec_e_stop_osi.py -q  EXPECT: все passed  EVIDENCE: pending (заполняется по PR)
G2 «выкл.» = v5 побайтно: test_vykl_ravno_v5_pobaytno (слепки кодом базы)  EVIDENCE: pending
G3 гэп через стоп — по open; цена в [low; high]  EVIDENCE: pending
G4 мутации ловятся (test_mutaciya_*)  EVIDENCE: pending
G5 быстрый слой: новая ось на стоп-заявке — эталон (test_novaya_os_na_stop_zayavke_idet_etalonom)  EVIDENCE: pending
G6 соседи: pytest -k "exec_e or stop or fast_v5 or fast_etap1"  EVIDENCE: pending
