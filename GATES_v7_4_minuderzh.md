# GATES V7-4: минимальное удержание контртренда 2 дня (min_uderzhanie)
Ветка v7-4-minuderzh. Владение: fastbt/exec_e_minuderzh.py, tests/test_exec_e_minuderzh.py; минимальные правки: exec_e_types.py (поле), exec_e_run.py и exec_e2.py (вызов), eval_v7.py (одна строка).

- G0: тест падает на базе. CHECK: `python3 -m pytest tests/test_exec_e_minuderzh.py -q` (первый коммит, до кода). EXPECT: красный. EVIDENCE: pending
- G1: без константы 3 точки (S1, S5, S3) побайтно как v6. CHECK: скрипт сравнения базы и ветки (хэш результата). EXPECT: хэши равны. EVIDENCE: pending
- G2: синтетика: сигнал дня 1 -> выход дня 2; стоп/тейк дня 1 -> день 1; tau=1 -> день 2. CHECK: pytest -k g2. EXPECT: зелёный. EVIDENCE: pending
- G3: мутация «стоп тоже переносится» ловится. CHECK: pytest -k g3. EVIDENCE: pending
- G4: мутация «перенос на тренд (S3/S4)» ловится. CHECK: pytest -k g4. EVIDENCE: pending
- G5: мутация «от дня сигнала» ловится (вход d+1 и вход в T). CHECK: pytest -k g5. EVIDENCE: pending
- G0b (V7-4b): тест переноса падает на базе. CHECK: pytest tests/test_exec_e_minuderzh_perenos.py (коммит 2). EVIDENCE: 7 failed (до кода)
- G2b/G3b/G4b (V7-4b): перенос, мутации «забыт», «не привязана к сделке», разворот в T + 2 мутации. CHECK: pytest tests/test_exec_e_minuderzh_perenos.py -q. EVIDENCE: 11 passed
- G5b (V7-4b): узкий набор exec_e*/eval_v7*/eval_v6*/kub_v7*. EVIDENCE: 595 passed, 121 skipped, 1 failed — tests/test_exec_e_minuderzh.py::test_g2_signal_vernulsya_na_den_2_pozitsiya_derzhitsya (тест автора #917, закрепляет старое «держится»); 12 с
- G6: регрессий нет. CHECK: `pytest tests/ -q -x -k "v6 or v7 or eval or kub or canon"`. EXPECT: зелёный. EVIDENCE: pending
