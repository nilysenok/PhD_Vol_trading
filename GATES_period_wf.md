# GATES Б8 period-wf
G1 v2 отдаёт 2015Q1..Q4 | CHECK: pytest tests/test_period_wf.py -k g1 | EXPECT: passed | EVIDENCE: passed (test_g1_v2_otdayot_2015; до правки уже проходил - отказ был только в progon_server)
G2 train4 побайтно прежний | CHECK: pytest -k g2 | EXPECT: passed | EVIDENCE: passed
G3 wf синтетика: 45 кварталов, суммы = прямому расчёту | CHECK: -k g3 | EXPECT: passed | EVIDENCE: passed
G4 wf на реальных путях до снятия замка — отказ | CHECK: -k g4 | EXPECT: passed | EVIDENCE: passed (2 теста)
G5 мутация «первый квартал выпал» ловится | CHECK: -k g5 | EXPECT: passed | EVIDENCE: passed
G6 тесты ledger/progon/svod зелёные | CHECK: pytest test_eval_v2_test1 test_etap1_eval_v2 test_period_wf test_progon_server test_ledger_period_v2 | EXPECT: 0 failed | EVIDENCE: 73 passed, 0 failed (210 с); ранее 147 passed по списку #619
