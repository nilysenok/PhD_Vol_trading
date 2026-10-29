# GATES К-wf2
G1 CHECK: python3 -m pytest tests/test_kalibr_wf2.py -q | EXPECT: passed | EVIDENCE: pending
G2 CHECK: окно t=2026 на wf = 44 квартала (2015Q1..2025Q4), без 2026Q1 | EXPECT: тест passed | EVIDENCE: pending
G3 CHECK: мутация granica side=right (окно с 1 января t) роняет тест | EXPECT: failed на мутации | EVIDENCE: pending
G4 CHECK: pytest tests/test_kontroli_v5.py tests/test_kalibr_v5.py (train4/v2 прежние) | EXPECT: passed | EVIDENCE: pending
G5 CHECK: py_compile; cargo xtask check | EXPECT: красных 0 | EVIDENCE: pending
