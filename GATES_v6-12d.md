# GATES v6-12d (замена #692: оси исполнения V6-12c + пивот/VWAP/pol_puti + оси тренда на стоп-заявке + свёртка time_decay)

- G1 конфликт слияния разрешён; оси #692 и ("ist", sha) в ключе (#694) сохранены
  CHECK: git merge origin/kanon-v22-pereschet; python3 -m pytest tests/test_istochniki_v6.py tests/test_klyuchi_kesha_v6.py tests/test_eval_v6c.py -q
  EXPECT: passed
  EVIDENCE: см. итог ниже (146 passed в узком прогоне синтетики)
- G2 «выкл.» всех новых осей = v5 побайтно на 20 точках каждого Sk
  CHECK: FASTBT_V6_REAL=1 python3 -m pytest tests/test_eval_v6d_real.py -k vykl
  EXPECT: 6 passed
  EVIDENCE: 14 passed in 194.48s (6 vykl v6d + 6 vykl v6c + 2 мутации v6c), 3 тикера, 1 процесс, train4 ≤ 2021
- G3 каждая новая ось меняет результат (пивот, VWAP, pol_puti, piv_buf, bezubytok/rang, stop_v_plyus, pir, reentry, значения V6-11)
  CHECK: FASTBT_V6_REAL=1 python3 -m pytest tests/test_eval_v6d_real.py -k "kazhdaya or mutatsiya"
  EXPECT: 22 passed
  EVIDENCE: 22 passed, 6 deselected in 56.34s
- G4 Соболь по кубу v6 не даёт «не подключена»; GOTOVY открывает подключённые оси
  CHECK: python3 -m pytest tests/test_eval_v6d.py tests/test_eval_v6b.py -q
  EXPECT: passed
  EVIDENCE: 298 passed, 91 skipped (широкий прогон v6-файлов + tejk_signal; затем 113 passed на istochniki/eval_v5/v6d после правки razobrat)
- G5 мутации: «пивот с d+1» (синтетика и реальный), «ось выпала из fkey», «токен оси убран из GOTOVY»
  CHECK: tests/test_eval_v6d.py::test_mutatsiya_pivot_s_d_plyus_1_lovitsya, ::test_fkey_novye_osi_i_mutatsiya_os_vypala_lovitsya, tests/test_eval_v6.py::test_razdelit_nepodklyuchennaya_i_vykl
  EXPECT: passed
  EVIDENCE: passed (в прогонах G3/G4; реальная мутация d+1 — в 22 passed)
- G6 time_decay = da без tau / при tejk ≠ net — свёртка в «выкл.», не выброс; доля сохранённых точек Соболя > 45 %
  CHECK: tests/test_eval_v6c.py::test_time_decay_vybros_bez_tau_ili_pri_tejke; tests/test_eval_v6d.py::test_sobol_vtoroj_chastiy_otkryvaet_novye_osi_i_hranit_tochki
  EXPECT: passed
  EVIDENCE: passed (17 passed в tests/test_eval_v6d.py)
- G7 тяжёлые реальные тесты v6 по умолчанию skip (FASTBT_V6_REAL=1 включает)
  CHECK: python3 -m pytest tests/test_eval_v6b_real.py tests/test_eval_v6c_real.py tests/test_v6_nez_eval.py tests/test_eval_v6.py -q
  EXPECT: быстро (≤ 60 с), реальные — skipped
  EVIDENCE: 298 passed, 91 skipped in 39.30s (в связке с остальными v6-файлами)
- ABANDON: G8 ось `tf` — отдельный исполнитель на 10-минутках встроить нельзя; остаётся ValueError «ещё не подключена»
- ABANDON: G9 ось `T` тренда и `smena` — T>1 есть только у лимитной заявки (exec_e._check_params), smena мертва при vhod_tip = stop_zayavka, а vhod_tip в куб не открыт
