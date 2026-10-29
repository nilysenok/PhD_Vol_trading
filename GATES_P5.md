# GATES_P5 — ось ТФ для S1–S6 v5 (лист П5, Ж-125 п. 3)

Владение: `fastbt/tf_os_v5.py` (новый), `fastbt/signals/v5_s56.py` (две точечные правки: группа «den» = календарный день; `vwap_fn` у `generate6`), `tests/test_tf_os_v5.py` (новый), `scripts/zamer_tf_os_v5.py` (новый), `GATES_P5.md`.

Ниже по три поля на каждый гейт: CHECK, EXPECT, EVIDENCE. EVIDENCE — прогон 29.10.2026 после слияния с `origin/kanon-v22-pereschet` @1a70ac67; источник — `python3 -m pytest tests/test_tf_os_v5.py -v` (23 passed in 0.80s) и команды ниже.

G1 tf=den побайтно = семья без оси (S1–S6).
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k den_pobajtno`
EXPECT: passed, 0 failed
EVIDENCE: test_den_pobajtno_semya_bez_osi[S1..S6] PASSED (6), test_den_pobajtno_gruppa_den_kak_nomer_bara[5 типов] PASSED (5), test_den_pobajtno_bary PASSED — 12 passed

G2 Агрегация 10 мин → 1 ч на синтетике с известным ответом.
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k agregaciya_1h`
EXPECT: passed, 0 failed
EVIDENCE: test_agregaciya_1h_izvestnyj_otvet PASSED

G3 Мутация «сигнал на незакрытом баре» ловится.
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k nezakrytyj`
EXPECT: passed, 0 failed
EVIDENCE: test_nezakrytyj_bar_mutaciya_lovitsya PASSED (верный путь не зависит от будущего; мутант `_bary_s_budushchim` зависит)

G4 Мутация «исполнение на баре сигнала» (Д2) ловится.
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k bar_signala`
EXPECT: passed, 0 failed
EVIDENCE: test_ispolnenie_na_bare_signala_lovitsya PASSED; test_signal_na_poslednem_bare_dnya_ne_ispolnyaetsya_utrom PASSED

G5 Мутация «бар через ночь» ловится.
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k cherez_noch`
EXPECT: passed, 0 failed
EVIDENCE: test_bar_cherez_noch_mutaciya_lovitsya[30m, 1h, 2h] PASSED (3)

G6 S5 на внутридневных ТФ — пивот прошлого дня/недели; S6 — VWAP с начала сессии.
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k "s5_ or s6_"`
EXPECT: passed, 0 failed
EVIDENCE: test_s5_pivot_proshlogo_dnya_i_nedeli PASSED; test_s6_vwap_s_nachala_sessii PASSED

G7 Строка издержек: комиссия на каждую сделку, число сделок в день (известный ответ).
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py -k izderzhk`
EXPECT: passed, 0 failed
EVIDENCE: test_izderzhki_kazhdaya_storona_i_sdelki_v_den PASSED (мутант «закрытие дня бесплатно» ловится); test_izderzhki_progon_stroki_i_den_otkaz PASSED

G8 Регрессия семей v5.
CHECK: `python3 -m pytest -q tests/test_v5_s56.py tests/test_v5_s56_formuly.py tests/test_v5_s12.py tests/test_v5_s34.py tests/test_v5_filtr.py tests/test_rassl_s56_equiv.py`
EXPECT: 0 failed
EVIDENCE: 106 passed in 1.77s

G9 Замер с/точку на 20 точках train4 (2016–2019, прогрев 2015), 1h, 17 тикеров.
CHECK: `FASTBT_MAIN=<worktree> python3 scripts/zamer_tf_os_v5.py`
EXPECT: строка `1h: 20 точек x 6 семей, ... с/точку`, код 0
EVIDENCE: S1 0.230 · S2 0.177 · S3 0.056 · S4 0.055 · S5 0.057 · S6 0.058 с/точку; «1h: 20 точек x 6 семей, 0.105 с/точку (17 тикеров, train4)»; баров на тикер ~11332; чтение 0.3 с

G10 Компиляция новых файлов.
CHECK: `python3 -m py_compile fastbt/tf_os_v5.py scripts/zamer_tf_os_v5.py tests/test_tf_os_v5.py fastbt/signals/v5_s56.py`
EXPECT: код 0
EVIDENCE: compile=0
