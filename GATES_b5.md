# GATES Б5 — свод возврата сервера (fastbt/svod_server_v5.py)

G1 CHECK: `python3 -m pytest tests/test_svod_server_v5.py::test_g1_izvestnyj_top10_vybran -q` EXPECT: passed EVIDENCE: 8 passed (весь файл, 29.10)
G2 CHECK: `pytest ...::test_g2_k1_k2_ravny_pryamomu_vyzovu` EXPECT: passed (≤ 1e-12 для k1 против независимого sr_dnevnoy; csv — 6 знаков, допуск 1e-6) EVIDENCE: passed
G3 CHECK: `pytest ...::test_g3_okno_do_31_12_t_minus_1_i_mutaciya ...::test_g3_mutaciya_k1_po_vsem_godam_lovitsya` EXPECT: passed EVIDENCE: passed
G4 CHECK: `pytest ...::test_g4_ledger_21_chitaetsya_i_20_tozhe` EXPECT: passed EVIDENCE: passed
G5 CHECK: `pytest ...::test_g5_shapka_status_kod_hash` EXPECT: passed EVIDENCE: passed
G6 CHECK: `python3 -m pytest tests/test_svod_server_v5.py tests/test_kalibr_v5.py tests/test_kontroli_v5.py -q` EXPECT: 30 passed EVIDENCE: 30 passed in 6.66s
