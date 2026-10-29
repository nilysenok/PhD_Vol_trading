# GATES Л1 (сетка v5)
G1 CHECK: python3 -m pytest tests/test_v5_filtr.py fastbt/tests --ignore=fastbt/tests/test_core.py -q | EXPECT: 0 failed | EVIDENCE: см. PR
G2 совпадение с apply_filters_generic@6a7a4636 (кроме atr_lo, squeeze): test_sovpadaet_s_apply_filters_generic | EVIDENCE: passed
G3 процентиль = расширяющееся окно до d-1: test_atr_protsentil_ekvivalent_rasshiryayushchemu_oknu | passed
G4 мутация «по всему ряду» падает: test_mutatsiya_protsentil_po_vsemu_ryadu_lovitsya | passed
G5 префикс: test_prefiks_vsekh_filtrov | passed
G6 реестр setka, v4 не меняется: test_reestr_setka + fastbt/tests | passed
