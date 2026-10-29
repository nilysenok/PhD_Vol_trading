# GATES_P4 — лист П4 сетки v5: оси Л5 в быстрый слой (29.10.2026)

Владение: `fastbt/fast_exec.py`, `fastbt/fast_etap1.py`, `tests/test_fast_v5_ekviv.py`, `GATES_P4.md`.
Медленный путь (`exec_e.run`) — эталон; быстрый = медленный побайтно (допуск 0).
Окружение реальных данных (ENV): `FASTBT_MAIN=$PWD FASTBT_BARS1M=<основная копия>/data/regen/D25/bars1m FASTBT_BARS5M_CACHE=<scratch> FASTBT_E1_CACHE=<scratch>`.
Все EVIDENCE — после слияния `origin/kanon-v22-pereschet` (коммит 1f41f721).

G1. Синтетика: быстрый ≡ медленный побайтно для каждой перенесённой оси (ed_s atr/pct, ed_t atr/pct, chand), ≥ 200 точек × 3 ленты на значение оси; ось реально меняет ленту (≥ 30 случаев на значение).
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k sintetika`
EXPECT: passed, 0 failed
EVIDENCE: `tests/test_fast_v5_ekviv.py -q -k "not realnye"` → `6 passed, 1 deselected in 10.06s` (600 точек × 3 ленты)

G2. Реальные 5-мин бары train4 (2016–2019; AFLT, SBER, LKOH): то же, 300 точек × 3 тикера (на ось ed_s 300, ed_t 200, chand 200).
CHECK: `ENV python3 -m pytest tests/test_fast_v5_ekviv.py -q -k realnye`
EXPECT: passed
EVIDENCE: `201.43s call tests/test_fast_v5_ekviv.py::test_realnye_bary_train4_bystryj_ravno_etalonu`; прогон с test_fast_etap1_equiv.py → `19 passed in 296.30s`. После перевода на 4 процесса fork (лимит хука Stop 300 с), без ENV (бары — из основной копии на чтение): `pytest tests/test_fast_v5_ekviv.py tests/test_fast_etap1_equiv.py tests/test_fast_etap1_smoke.py` → `464 passed in 147.65s`, реальные бары `55.56s`

G3. #488: режим, неизвестный быстрому слою (поле Params вне FAST_POLYA, новое поле подкласса, неизвестное значение f/s/ed, лишний вход run), уходит на медленный путь; мутация «прежний список исключений» ловится.
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k "neizvestn or mutaciya"`
EXPECT: passed
EVIDENCE: входят в `6 passed` G1 (test_neizvestnyj_rezhim_uhodit_v_etalon, test_mutaciya_488_spisok_isklyuchenij_lovitsya); поля Л6/П3b из базы (k_e, a_rang, parab, chast) уходят в эталон без правки списка

G4. Мутации медленного пути ловятся сверкой: ATR дня d в Chandelier, единица стопа = σ̂ при ed_s=atr, храповик снят.
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k mut_yadro`
EXPECT: 3 passed
EVIDENCE: входят в `6 passed` G1 (test_mut_yadro_lovitsya[3 параметра])

G5. Регрессии быстрого слоя и соседних осей.
CHECK: `python3 -m pytest tests/test_exec_e_parabolic.py tests/test_exec_e_v5_ed.py tests/test_exec_e_v5_vhod.py tests/test_fast_etap1_smoke.py tests/test_proverka_trail_atr.py tests/test_rodnoy_uroven.py tests/test_stop_zayavki.py tests/test_tejk_signal.py tests/test_v6_a.py tests/test_v6_b.py tests/test_v6_c.py -q`; `ENV python3 -m pytest tests/test_fast_etap1_equiv.py -q`
EXPECT: 0 failed, кроме теста старого контракта
EVIDENCE: `1 failed, 611 passed in 25.10s` — красный `tests/test_exec_e_v5_ed.py::test_slow_only_dlya_novykh_osej` (утверждает, что ed_s/ed_t/chand — медленный путь; это контракт до П4, автор тест не меняет — правку делает тестировщик); test_fast_etap1_equiv.py — `12 passed` (в составе 19 passed G2)

G6. Замер с/точку до/после на 50 точках train4 (точка = Params осей Л5 × 3 тикера, тёплый быстрый слой).
CHECK: `ENV python3 <scratchpad>/zamer_p4.py`
EXPECT: печать «до», «после»
EVIDENCE: `до (медленный exec_e.run): 0.6355 с/точку (3 тикера), 0.2118 с/вызов тикера`; `после (fast_etap1.run): 0.0116 с/точку (3 тикера), 0.0039 с/вызов тикера`; компиляция/первый вызов 0.21 с

G7. tt=3 (ATR-trailing, engine_trend): уже в numba (`trend_run`, коммит 88cb970e) — ручной гейт: переносить нечего.
CHECK: `grep -n "tt == 3" fastbt/engine_trend.py`
EXPECT: строки внутри `@njit trend_run`
EVIDENCE: `48: elif tt == 3:`, `61: elif tt == 3:` (функция `trend_run` под `@njit(cache=True)`, строка 9)
