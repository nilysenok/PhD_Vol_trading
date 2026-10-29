# GATES: progon_server v5 (задание 2026-10-29_progon_server_v5.md)

Все CHECK — `python3 -m pytest <файл>::<тест> -q` из корня; EXPECT — `passed`, код 0.

- G1 обёртка ≡ драйвер, куб v5 train4 seed 1. CHECK: `tests/test_progon_server.py::test_v5_train4_seed1_ravno_drajveru`. EVIDENCE: 2 passed (obshchie, prezhniy).
- G2 period v2 (ledger 2.0) и seed 2: разные каталоги, период и seed в ключе, чекпоинте и manifest. CHECK: `::test_v5_period_v2_i_seed2_raznye_katalogi_period_i_seed_v_manifest`. EVIDENCE: passed.
- G3 `--k3` ≡ драйвер. CHECK: `::test_v5_k3_ravno_drajveru`. EVIDENCE: passed.
- G4 мутация: период не в отпечатке (POLYA без period) — train4-каталог продолжен как v2; реальный код отказывает. CHECK: `::test_mutaciya_period_ne_v_otpechatke_chekpointa_lovitsya`. EVIDENCE: passed.
- G5 мутация: выброс canon_v5 пропущен — набор sobol_i меняется. CHECK: `::test_mutaciya_vybros_canon_v5_propushchen_lovitsya`. EVIDENCE: passed.
- G6 `pytest tests/test_progon_server.py tests/test_progon_sobol.py tests/test_progon_sobol_v5.py tests/test_ledger_v2.py`. EVIDENCE: 62 passed; 5 повторов test_progon_server.py: 19 passed × 5.
- G7 команды запуска 5 проходов — в теле PR; CLI проверен `::test_main_v5_komanda_servera` (passed).
