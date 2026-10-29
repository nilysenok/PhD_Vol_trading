# GATES_P5b — внутридневное исполнение Ж-111 на оси ТФ (лист П5b, Ж-125 п. 3)

Владение: `fastbt/tf_ispolnenie_v5.py` (новый), `tests/test_tf_ispolnenie_v5.py` (новый), `scripts/zamer_tf_ispolnenie_v5.py` (новый), `GATES_P5b.md`.
EVIDENCE — прогон 29.10.2026 на базе `origin/kanon-v22-pereschet` @024a1669.

G1 Известный ответ: лимит исполнен пробоем, не касанием (№ 2); тейк по уровню; издержки две стороны.
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k "limit or kasanie"`
EXPECT: passed, 0 failed
EVIDENCE: test_limit_ispolnen_probojem_i_tejk_po_urovnyu, test_kasanie_limita_ne_ispolnyaet_i_myutant_tick0_lovitsya PASSED

G2 Стоп и тейк в одном баре — стоп первый (№ 3); мутант «стоп не проверяется» ловится.
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k stop_i_tejk`
EXPECT: passed, 0 failed
EVIDENCE: test_stop_i_tejk_v_odnom_bare_stop_pervyj_i_myutant_lovitsya PASSED

G3 Гэп через стоп — по open (№ 4).
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k gep`
EXPECT: passed, 0 failed
EVIDENCE: test_gep_cherez_stop_ispolnen_po_open_a_ne_po_urovnyu PASSED

G4 Закрытие в конце сессии с комиссией выхода (№ 5); мутант без комиссии выхода ловится.
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k zakrytie`
EXPECT: passed, 0 failed
EVIDENCE: test_zakrytie_v_konce_sessii_platit_vyhod_i_myutant_bez_vyhoda_lovitsya PASSED

G5 Перенос через ночь запрещён; мутант «склеенные сессии» ловится.
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k noch`
EXPECT: passed, 0 failed
EVIDENCE: test_perenos_cherez_noch_zapreshchen_i_myutant_lovitsya PASSED

G6 Д2: исполнение на баре сигнала невозможно (k = None, vyhod_lim = False → ValueError; `proverka_d2` ловит вход на баре сигнала); выход по сигналу — заявкой на t+1; решение на последнем баре не исполняется.
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k "d2 or signal or posledniy"`
EXPECT: passed, 0 failed
EVIDENCE: test_d2_zapreshcheny_vhod_po_close_i_vyhod_po_close_bara_signala, test_vyhod_po_signalu_zayavkoj_na_sleduyushchem_bare, test_posledniy_bar_resheniya_ne_ispolnyaetsya_i_pervyj_bar_bez_pozicii PASSED

G7 σ̂ бара без будущего и без ночных разрывов; строка издержек (нетто = брутто − издержки), tf = den не здесь.
CHECK: `python3 -m pytest -q tests/test_tf_ispolnenie_v5.py -k "sigma or progon"`
EXPECT: passed, 0 failed
EVIDENCE: test_sigma_bez_budushchego_i_bez_nochnyh_razryvov, test_progon_stroka_izderzhek_i_tf_den_ne_zdes PASSED

G8 tf = den побайтно прежнее (код П5 и дневного движка не менялся).
CHECK: `python3 -m pytest -q tests/test_tf_os_v5.py tests/test_tf_ispolnenie_v5.py` и `git diff --stat origin/kanon-v22-pereschet -- fastbt/exec_e*.py fastbt/tf_os_v5.py`
EXPECT: 34 passed; diff пуст
EVIDENCE: 34 passed in 0.77s; diff пуст

G9 Замер 1h, 20 точек на семью, train4 (чтение < 2020-01-01).
CHECK: `FASTBT_MAIN=<worktree> python3 scripts/zamer_tf_ispolnenie_v5.py`
EXPECT: 6 строк семей + итог
EVIDENCE: 0.26–0.78 с/точку по семьям, в среднем 0.47 с/точку (17 тикеров); см. описание PR
