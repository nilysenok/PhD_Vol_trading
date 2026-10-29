# GATES — Ф-сл3d: оси #653/#660 в pokryto

G1. pokryto по флагу YADRO_OSI3_GOTOV; мутация «ось принята при ядре без неё» ловится.
CHECK: pytest tests/test_fast_v5_osi3_pokryto.py -k "pokryto_po_flagu or mutatsiya"
EXPECT: passed
EVIDENCE: 4 passed (без данных, весь файл: 4 passed, 12 skipped)

G2. fast_etap1.run с rang делегирует ядру при флаге, эталону без флага; побайтно.
CHECK: pytest tests/test_fast_v5_osi3_pokryto.py -k delegiruet
EXPECT: passed
EVIDENCE: см. итог PR

G3. time_decay, piv_*, оси v5 на стоп-заявке, t_vyhod ≠ close — ne_znaet / эталон.
CHECK: pytest tests/test_fast_v5_osi3_pokryto.py -k pozdnie
EXPECT: passed
EVIDENCE: см. итог PR

G4. Быстрый = медленный побайтно, 200 точек на каждый Sk v5 (S1–S6), ≤ 2021.
CHECK: FASTBT_V5_N=200 pytest tests/test_fast_v5_osi3_pokryto.py -k ekviv
EXPECT: 6 passed
EVIDENCE: см. итог PR

G5. Замер с/точку на 20 точках каждого Sk.
CHECK: FASTBT_V5_N=200 pytest tests/test_fast_v5_osi3_pokryto.py -k zamer -s
EXPECT: 6 passed, таблица
EVIDENCE: см. итог PR
