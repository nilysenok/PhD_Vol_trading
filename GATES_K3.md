# GATES — К-3 к1 по префиксу Соболя, kodir, 29.10
## G1 — к1 по префиксу: при выборке = пулу равен kv.k1; при усечении — среднему точек sobol_i < n
CHECK: python -m pytest tests/test_svod_k1_prefiks.py -q
EXPECT: все passed
EVIDENCE: 6 passed
## G2 — мутация (отбор по sobol_i убран) ловится
CHECK: подмена `i < n` в k1_prefiks, тот же файл
EXPECT: failed
EVIDENCE: 4 failed, 2 passed (возвращено)
## G3 — seed и индексы в шапке csv, подключение в svod, CLI прежний
CHECK: pytest tests/test_svod_k1_prefiks.py tests/test_svod_server_v5.py -q
EXPECT: passed
EVIDENCE: 14 passed
## Реестр
закрыто 3 из 3
