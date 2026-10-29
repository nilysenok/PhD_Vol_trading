# GATES V6-1a (rel_ryad: синхронные бары, td_rel)
G1 CHECK: python3 -m pytest tests/test_rel_ryad.py -q  EXPECT: все passed  EVIDENCE: 17 passed
G2 мутации (индекс d+1, сдвиг бара +1, left-join вместо внутреннего) ловятся тестами test_mutatsiya_*  EVIDENCE: 17 passed (мутации внутри тестов)
G3 CHECK: python3 -m py_compile fastbt/rel_ryad.py  EVIDENCE: код 0. `cargo xtask check` — в этом проекте не существует (CLAUDE.md), не запускался
