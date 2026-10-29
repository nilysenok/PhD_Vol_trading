# GATES П2
G1 CHECK: python3 -m pytest tests/test_exec_e_v4sloj.py -q | EXPECT: 11 passed | EVIDENCE: 11 passed
G2 мутации № 20, cooldown от входа, consec d+1 ловятся (test_mutatsii_lovyatsya, test_mut20_...) | EVIDENCE: passed в G1
G3 CHECK: git diff --stat origin/kanon-v22-pereschet -- fastbt/exec_e_types.py exec_e.py exec_e_run.py exec_e_v6.py exec_e_stop.py | EXPECT: пусто | EVIDENCE: пусто
G4 CHECK: pytest tests/test_exec_e.py tests/test_exec_e2.py tests/test_kalibr_v5.py -q | EXPECT: passed | EVIDENCE: 56 passed
