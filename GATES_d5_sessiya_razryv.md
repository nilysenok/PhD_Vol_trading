# GATES d5-sessiya-razryv

G1 сокращённый день вне RV: CHECK: pytest tests/test_ostanovka_2022.py tests/test_sessiya_razryv.py -q · EXPECT: 36 passed, xfail снят · EVIDENCE: 36 passed
G2 мутации (сокращённый как полный, медиана с днём d, заявка не снимается, порог >=) ловятся: те же тесты (test_mut_*, test_porog_razryva_strogo_bolshe) · EVIDENCE: passed
G3 ≤ 2021 прежнее побайтно: test_bez_razryvov_pobaytno_prezhnee + замер (0 сокращённых дней и 0 разрывов > 5 дней, 17 тикеров, 2017–2021) · EVIDENCE: passed; замер в PR
G4 соседние тесты exec_e/sigma_real: pytest tests/test_exec_e*.py tests/test_sigma_real*.py · EVIDENCE: 870 passed (вместе с fast_*), 1 failed — test_fast_v5_pokryto (не от правки, см. PR)
