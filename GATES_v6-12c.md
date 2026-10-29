# GATES v6-12c (срез: оси исполнения + свёртки; пивот/тренд/tf — не в этом PR)

- G1 «выкл.» = v5 побайтно, 20 точек каждого Sk
  CHECK: SDF_TYAZHELYE=1 python3 -m pytest tests/test_eval_v6c_real.py -k vykl
  EXPECT: 6 passed
  EVIDENCE: pending
- G2 каждая ось исполнения меняет результат там, где есть в кубе
  CHECK: SDF_TYAZHELYE=1 python3 -m pytest tests/test_eval_v6c_real.py -k menyaet
  EXPECT: passed, skipped только «оси нет в кубе»
  EVIDENCE: pending
- G3 fkey содержит новые оси; Соболь не даёт «не подключена»; GOTOVY по осям
  CHECK: python3 -m pytest tests/test_eval_v6c.py tests/test_eval_v6b.py tests/test_kub_v6.py tests/test_v6_nez_kuby.py
  EXPECT: все passed
  EVIDENCE: pending
- G4 мутации ловятся (лесенка без /2, ось не доходит, ключ)
  CHECK: pytest tests/test_eval_v6c.py -k lestnica; tests/test_eval_v6c_real.py -k mutatsiya
  EXPECT: passed
  EVIDENCE: pending
- ABANDON: G5 пивот S5/VWAP S6 (stop_nach piv:/vwap:, tejk piv:, pol_puti), G6 оси тренда на стоп-заявке (V6-11), G7 tf — таймбокс 50 тыс. токенов; см. отчёт
