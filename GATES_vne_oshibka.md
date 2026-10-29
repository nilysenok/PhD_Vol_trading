# GATES Б2 — исполнение вне [low; high] = ValueError

G1 CHECK: `NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/test_vne_oshibka.py -k g1` EXPECT: passed; до правки падал. EVIDENCE: 2 passed (до правки: failed)
G2 ABANDON: G2 — в быстром слое (fast_etap1/fast_exec, numba-ядро) счётчика нарушений нет, зеркалить нечего; проверку инварианта туда не добавляю (вне владения по сути — это новая функция).
G3 CHECK: `pytest -q tests/test_vne_oshibka.py -k g3` EXPECT: passed (точка с нарушением -> status oshibka, остальные ok). EVIDENCE: 2 passed
G4 CHECK: `NUMBA_NUM_THREADS=1 pytest -q tests/test_ispolnenie_v_bare.py tests/test_rassl_tf_s1.py tests/test_exec_e*.py tests/test_fast_v5_ekviv.py tests/test_fast_etap1_equiv.py` EXPECT: 0 failed. EVIDENCE: 6 failed, 233 passed — КРАСНОЕ (мутационные тесты гэпа ждали досчёта мутанта; файлы не мои)
G5 CHECK: `pytest -q tests/test_vne_oshibka.py -k g5` EXPECT: passed (мутация-счётчик роняет G1). EVIDENCE: 1 passed
