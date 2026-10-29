# GATES Р-5 sigma_h22
G1 CHECK: python3 -m pytest tests/test_sigma_h22_sdvig.py -q  EXPECT: 3 passed, 1 xfailed  EVIDENCE: 3 passed, 1 xfailed
G2 (причина названа) CHECK: тест test_skript_pokazyvaet_otstavanie_v_minus_odin  EXPECT: pass  EVIDENCE: pass
G3 (ряд не проходит, xfail strict) CHECK: test_h22_proxodit_d1  EXPECT: xfailed  EVIDENCE: xfailed
