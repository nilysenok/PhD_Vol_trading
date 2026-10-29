# GATES П3
G1 CHECK: python3 -m pytest tests/test_vyhod_parabolic.py -q | EXPECT: 12 passed | EVIDENCE: 12 passed
G2 CHECK: python3 -m py_compile fastbt/vyhod_parabolic.py tests/test_vyhod_parabolic.py | EXPECT: rc 0 | EVIDENCE: rc 0
G3 мутации 3, 4, look-ahead ловятся: тесты test_mutaciya_* и test_gep_*, test_stop_i_signal_* в G1 | EVIDENCE: passed
