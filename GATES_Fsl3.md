# GATES Ф-сл3 (ветка fast-v5-osi3): оси b0, b0_c, sp, step_s, a_rang + вход rang в kernel_v5

G1 CHECK: FASTBT_NO_CACHE=1 pytest tests/test_fast_v5_osi3.py -k sintetika  EXPECT: passed (1200 путей, побайтно, оси работают >= 20)
EVIDENCE: passed.
G2 CHECK: FASTBT_NO_CACHE=1 pytest tests/test_fast_v5_osi3.py -k realnye  EXPECT: passed (50 точек AFLT train4 <= 2019, > 20 сделок на >= 30)
EVIDENCE: passed.
G3 CHECK: pytest tests/test_fast_v5_osi3.py -k "krajnie or otkazy or pir"  EXPECT: passed
EVIDENCE: passed.
G4 CHECK: pytest tests/test_fast_v5_osi3.py -k mutatsiya  EXPECT: 3 passed (ранг дня входа; plus_level; ядро a1 с обратным знаком)
EVIDENCE: passed. Ручная мутация ядра (F_SQ -> F_SP в extra_level): sintetika падает (при свежем numba-кэше).
G5 CHECK: pytest test_fast_v5_s34 test_fast_v5_pokryto test_fast_etap1_smoke  EXPECT: без регрессий
EVIDENCE: 470 passed, 6 skipped (вместе с G1-G4).
G6 включение в pokryto: ABANDON: G6 fast_etap1.pokryto/patched принадлежат Ф-сл2b, флаг и вызов run_v5 ещё не слиты; включить после её слияния.
