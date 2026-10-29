# GATES Л4 (S5@v5, S6@v5)

G1 S5 classic/fibonacci неделя/месяц fs 2–4 = S5 v4 побайтно
CHECK: python3 -m pytest tests/test_v5_s56.py -q -k ravny_v4
EXPECT: passed, 0 failed
EVIDENCE: 2 passed, 26 deselected

G2 пивоты 5 типов, den/неделя/месяц, уровни 1–3, atr_filtr = формулы Р @6a7a4636
CHECK: python3 -m pytest tests/test_v5_s56_formuly.py -q
EXPECT: passed, 0 failed
EVIDENCE: 16 passed

G3 мутация «пивот по текущему периоду» падает
CHECK: python3 -m pytest tests/test_v5_s56.py -q -k "mutaciya or ne_vidit"
EXPECT: passed, 0 failed
EVIDENCE: 19 passed, 9 deselected

G4 весь лист
CHECK: python3 -m pytest tests/test_v5_s56.py tests/test_v5_s56_formuly.py -q
EXPECT: passed, 0 failed
EVIDENCE: 44 passed

G5 регистрация в реестре setka="v5" — ABANDON: G5 Л1 не в базе (register_family / v5_filtr.py отсутствуют), по заданию — последним коммитом после слияния Л1
CHECK: -
EXPECT: -
EVIDENCE: ABANDON
