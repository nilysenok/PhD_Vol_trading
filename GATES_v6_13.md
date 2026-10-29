# GATES — V6-13 кубы v6 + canon_v6 + obem_v6, агент kodir, 29.10.2026 (PR 1 из 2: canon_v6 + кубы S1, S3)

## G1 — canon_v6 + кубы S1, S3 проходят тесты
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_kub_v6.py -q
EXPECT: все passed
EVIDENCE: 20 passed in 0.71s (29.10)

## G2 — v5 вложена в v6 (канон побайтно); мутация «выкл. не выбрасывается» ловится; Соболь без готовых осей = Соболь v5
CHECK: pytest tests/test_kub_v6.py -k "vlozhena or mutaciya or bez_gotovykh"
EXPECT: passed
EVIDENCE: входит в G1 (test_v5_vlozhena_v_v6, test_mutaciya_vykl_ne_vybrasyvaetsya_lovitsya, test_sobol_bez_gotovykh_osej_raven_v5): passed

## G3 — у каждой новой оси vykl = первое значение, smysl, otkuda; фильтры p_vykl 0,9 (14 контр / 12 тренд); доля «выкл.» 0,9 ± 0,02 на 4096 точках
CHECK: pytest tests/test_kub_v6.py -k "vykl or p_vykl"
EXPECT: passed
EVIDENCE: входит в G1: passed

## G4 — объём сходится с таблицей инвентаризации §3.1 (S1, S3); замороженный куб = v5
CHECK: pytest tests/test_kub_v6.py -k "obem"
EXPECT: passed (S1 ±0,5 %, S3 −2,9 % с ↺)
EVIDENCE: входит в G1: passed

## G5 — py_compile; cargo xtask check
CHECK: python3 -m py_compile fastbt/canon_v6.py tests/test_kub_v6.py; cargo xtask check
EXPECT: красных 0
EVIDENCE: py_compile — 0; `cargo xtask check` — «no such command: xtask» (в этом проекте xtask нет, CLAUDE.md «Не применяется»): не выполнен

## Реестр
закрыто 4 из 5; ABANDON: 0; pending: 0; G5 выполнен частично (xtask отсутствует)
