# GATES: таблица разрыва предзащиты (Ж-129 п. 1)

G1 скрипт собирает таблицу, остаток < 1e-3 во всех 12 ячейках
- CHECK: python3 scripts/revers/r1_tabl_razryv.py && python3 -m pytest tests/test_r1_tabl_razryv.py -q
- EXPECT: код 0, 29 passed
- EVIDENCE: 29 passed in 0.06s; остаток по CSV |x| < 1e-6 во всех строках

G2 мутация «шаг пропущен» ловится (A, C, D × P0, P4, P7, P10, E5, P1, P6)
- CHECK: тот же pytest, test_realnye_propushchen_shag_lovitsya
- EXPECT: pass (шаг с нулевым вкладом, A:P7 — проверяется нулевость)
- EVIDENCE: pass

G3 CSV с шапкой (статус, коммит, хэши входов) на месте
- CHECK: head -12 docs/tables/revers_r1/tabl_razryv_predzashchita.csv
- EXPECT: статус, коммит, 6 хэшей входов
- EVIDENCE: выполнено

G4 FINAL_B: раздел в конце, дословная формулировка первым абзацем; глава 4: блок в начале файла
- CHECK: grep -n "Калибровочное окно включало" docs/final_gl4/FINAL_B.md docs/tekst_chernoviki/4_test.md
- EXPECT: по одной строке в обоих
- EVIDENCE: см. отчёт
