# GATES_b7 — ledger период В2
G1 CHECK: pytest tests/test_ledger_period_v2.py -k g1 · EXPECT: passed (падает до правки) · EVIDENCE: до правки 2 failed + 2 error (ValueError на 2015), после — 6 passed
G2 CHECK: pytest tests/test_ledger_period_v2.py -k g2 · EXPECT: passed (sha шардов старого кода) · EVIDENCE: passed
G3 CHECK: pytest tests/test_ledger_period_v2.py -k g3 · EXPECT: passed · EVIDENCE: passed
G4 CHECK: pytest tests/test_ledger_period_v2.py -k g4 · EXPECT: passed · EVIDENCE: passed
G5 CHECK: pytest tests/test_ledger_period_v2.py -k mutaciya · EXPECT: passed · EVIDENCE: passed
G6 CHECK: pytest (список задания, 10 файлов) · EXPECT: зелёные · EVIDENCE: 109 passed
