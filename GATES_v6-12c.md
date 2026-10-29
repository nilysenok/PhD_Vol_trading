# GATES v6-12c (срез: оси исполнения + свёртки; пивот/тренд/tf — не в этом PR)

- G1 «выкл.» = v5 побайтно, 20 точек каждого Sk
  CHECK: SDF_TYAZHELYE=1 python3 -m pytest tests/test_eval_v6c_real.py
  EXPECT: 30 passed, 8 skipped (оси нет в кубе Sk)
  EVIDENCE: 30 passed, 8 skipped in 120.01s (6 тестов vykl среди них)
- G2 каждая ось исполнения меняет результат там, где есть в кубе
  CHECK: тот же прогон (test_kazhdaya_os_ispolneniya_menyaet_rezultat)
  EXPECT: passed
  EVIDENCE: passed, см. G1
- G3 fkey содержит новые оси; Соболь не даёт «не подключена»; GOTOVY по осям
  CHECK: python3 -m pytest tests/test_eval_v6c.py tests/test_eval_v6b.py tests/test_kub_v6.py tests/test_v6_nez_kuby.py tests/test_klyuchi_kesha_v6.py
  EXPECT: все passed
  EVIDENCE: 147 passed in 12.55s
- G4 мутации ловятся (лесенка без /2, ось не доходит, ключ)
  CHECK: test_eval_v6c.py -k lestnica; test_eval_v6c_real.py -k mutatsiya
  EXPECT: passed
  EVIDENCE: passed (в прогонах G1, G3)
- G8 независимый тест базы
  CHECK: python3 -m pytest tests/test_v6_nez_eval.py
  EXPECT: passed (медленный, ~140 с)
  EVIDENCE: 81 passed in 138.20s
- ABANDON: G5 пивот S5/VWAP S6 (+pol_puti) · ABANDON: G6 оси тренда на стоп-заявке (V6-11) · ABANDON: G7 tf — таймбокс 50 тыс. токенов; см. PR #692
