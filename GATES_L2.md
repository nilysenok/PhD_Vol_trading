# GATES_L2 (лист Л2 сетки v5)
G1 CHECK: python3 -m pytest tests/test_v5_s12.py -q | EXPECT: 24 passed | EVIDENCE: 24 passed in 0.63s
G2 v4-точка побайтно: test_s1_sma_std_ravno_v4_na_setke_v4, test_s2_sma_none_ravno_v4_na_setke_v4 | EVIDENCE: passed
G3 мутация MA t+1 падает: test_mutaciya_ma_s_barom_t_plyus_1_lovitsya | EVIDENCE: passed
G4 py_compile | EVIDENCE: ok
