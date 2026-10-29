# GATES — К-wf3 a_strogaya на wf, kodir, 29.10

## G1 — годы wf = 2017…2026 (10 окон), train4 = 2016…2019 как прежде
CHECK: python3 -m pytest tests/test_a_strogaya_wf.py -q
EXPECT: passed
EVIDENCE: 5 passed

## G2 — паспорт train4 прежний; паспорт wf на синтетике — 10 годов + «все»; старые тесты зелёные
CHECK: python3 -m pytest tests/test_a_strogaya_wf.py tests/test_a_strogaya.py tests/test_a_strogaya_ryady.py -q
EXPECT: passed
EVIDENCE: 28 passed

## G3 — мутация (годы wf = годы train4) ловится
CHECK: test_mut_wf_gody_kak_train4, test_mut_pasport_wf_s_godami_train4 (pytest.raises(AssertionError))
EXPECT: passed
EVIDENCE: оба passed

## G4 — компиляция
CHECK: python3 -m py_compile fastbt/a_strogaya.py
EXPECT: rc 0
EVIDENCE: ok

## Реестр
закрыто 4 из 4; ABANDON: 0; pending: 0
