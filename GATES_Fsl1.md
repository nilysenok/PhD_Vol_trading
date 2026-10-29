# GATES Ф-сл1 (ветка fast-v5-pokryto)
Находка: у точек v5 `pokryto` отсекает только ось `rang_tip` ≠ tikernyj (~52 % точек S1/S3/S5 на 175–300 первых точках); остальные оси v5 `pokryto` пропускает, а `run` сам уводит непокрытые поля (`_slow_only`) в эталон. `digest` на набор входов уже кэширует `_digest_fast` (0,18 с -> 0,0003 с на реальных S1 train4), но работает только внутри `patched()`, т.е. только на точках, прошедших `pokryto`.

G1 CHECK: FASTBT_V5_N=200 pytest tests/test_fast_v5_pokryto.py -k ekviv  EXPECT: 6 passed
EVIDENCE: 6 passed (S1 111 с, S2 111 с, S3 549 с, S4 280 с, S5 104 с, S6 112 с): быстрый = медленный побайтно (строка ledger без t_sek/sloy, port, tik).
G2 CHECK: pytest test_exec_e_v5_ed, test_v6_b, test_exec_e_parabolic, test_v6_c, test_tejk_signal, test_rodnoy_uroven  EXPECT: passed
EVIDENCE: 115 passed.
G3 CHECK: pytest tests/test_fast_v5_pokryto.py::test_mutatsiya_nepokrytaya_os_lovitsya  EXPECT: passed
EVIDENCE: passed; без правки pokryto 8 тестов файла падают.
G4 CHECK: замер с/точку до/после (малая проверка, свежая среда на режим)  EXPECT: число
EVIDENCE: 12 точек (ok 12/12), Мак, 1 процесс: S1 0,639 -> 0,466 с/точку (-27 %), S5 0,567 -> 0,392 (-31 %). Потолок: одна машина, 12 точек, не 20.
