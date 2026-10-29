# GATES — V6-13 кубы v6 + canon_v6 + obem_v6, агент kodir, 29.10.2026 (PR 1: canon_v6 + S1, S3; PR 2: S2, S4, S5, S6 + obem_v6)

## G1 — canon_v6 + шесть кубов проходят тесты
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_kub_v6.py -q
EXPECT: все passed
EVIDENCE: PR 1 — 20 passed; PR 2 — 51 passed in 1.09s (29.10)

## G2 — v5 вложена в v6 (канон побайтно); мутация «выкл. не выбрасывается» ловится; Соболь без готовых осей = Соболь v5
CHECK: pytest tests/test_kub_v6.py -k "vlozhena or mutaciya or bez_gotovykh"
EXPECT: passed
EVIDENCE: входит в G1 (test_v5_vlozhena_v_v6 ×6, test_mutaciya_vykl_ne_vybrasyvaetsya_lovitsya, test_sobol_bez_gotovykh_osej_raven_v5 ×6): passed

## G3 — у каждой новой оси vykl = первое значение, smysl, otkuda; фильтры p_vykl 0,9 (14 контр / 12 тренд); доля «выкл.» 0,9 ± 0,02
CHECK: pytest tests/test_kub_v6.py -k "vykl or p_vykl"
EXPECT: passed
EVIDENCE: входит в G1: passed

## G4 — объём сходится с таблицей §3.1 (все шесть); замороженный куб = v5; obem_v6.py печатает таблицу
CHECK: pytest tests/test_kub_v6.py -k "obem or zamorozka"; FASTBT_MAIN=$PWD python3 fastbt/designs/v6/obem_v6.py
EXPECT: passed; S1 1,87e38 S2 1,17e38 S3 1,76e31 S4 7,75e31 S5 2,07e37 S6 2,81e37
EVIDENCE: passed; 1,871e38 / 1,169e38 / 1,708e31 / 7,536e31 / 2,073e37 / 2,809e37 (S3, S4 на 2,9 % ниже — ↺ тейк 35)

## G5 — свёртки соседних листов: dlit тренда (#647), period S4 при s252 (#648)
CHECK: pytest tests/test_kub_v6.py -k "dlit or period"
EXPECT: passed
EVIDENCE: входит в G1: passed

## G6 — py_compile; cargo xtask check
CHECK: python3 -m py_compile fastbt/canon_v6.py fastbt/designs/v6/obem_v6.py tests/test_kub_v6.py; cargo xtask check
EXPECT: красных 0
EVIDENCE: py_compile — 0; `cargo xtask check` — «no such command: xtask» (в этом проекте xtask нет): не выполнен

## Реестр
закрыто 5 из 6; ABANDON: 0; pending: 0; G6 частично (xtask отсутствует)
