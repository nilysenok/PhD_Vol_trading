# Гейты починки эталона S5/S6
G1 CHECK: python3 -m pytest fastbt/tests/test_s56_equiv.py -q | EXPECT: 16 passed | EVIDENCE: 16 passed
G2 CHECK: python3 -m pytest tests/test_rassl_s56_equiv.py -q (мутация: эталон на сыром ряду) | EXPECT: 3 passed | EVIDENCE: прошло в общем прогоне
G3 CHECK: прогон s56+rassl+s12+s34+v5_s56 | EXPECT: все зелёные | EVIDENCE: 70 passed
