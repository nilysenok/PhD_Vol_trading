# GATES_L6

G1 k_e, тейк от силы, chast (доля, r) работают
CHECK: python3 -m pytest tests/test_exec_e_v5_vhod.py -q
EXPECT: 18 passed
EVIDENCE: 18 passed (прогон вместе с регрессией: 643 passed)

G2 k_e = 0, a_rang=(a0, 0), chast=(½, 1) побайтно прежнее
CHECK: тесты test_tejk_ot_sily_a1_nol_*, test_chast_poloviny_*, test_ke_sdvigaet_buy_stop_vverkh[0.0]
EXPECT: passed
EVIDENCE: passed

G3 мутации «ранг дня входа», «против позиции» (k_e, тейк) падают
CHECK: test_mutaciya_* (3 теста)
EXPECT: passed (мутация даёт иной результат)
EVIDENCE: passed

G4 регрессия: tests/test_exec_e*.py, test_fast_etap1_smoke, audit_fastbt 14/15/04
CHECK: pytest этих файлов
EXPECT: без падений
EVIDENCE: 643 passed
