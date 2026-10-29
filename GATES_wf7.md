# GATES К-wf7: fastbt/sravneniya_wf.py (таблица §6: Δ, LW-p, Холм)
G1 Δ = разность табличных дневных Шарпов (ddof=1, √252) на общих днях. CHECK: python3 -m pytest tests/test_sravneniya_wf.py -k delta. EXPECT: passed. EVIDENCE: passed
G2 Холм по m = len(pary), m — параметр. CHECK: pytest -k holm. EXPECT: passed. EVIDENCE: passed
G3 Мутации № 12 (ddof=0) и № 13 (Δ из ЛВ) роняют проверку. CHECK: pytest -k mutaciya. EXPECT: passed. EVIDENCE: passed
G4 Формат v:"1", пустая лента/нулевая дисперсия/нет общих дней — ошибка. CHECK: pytest -k format. EXPECT: passed. EVIDENCE: passed (всего 6 passed)
G5 py_compile, файл ≤ 200 строк. CHECK: python3 -m py_compile; wc -l. EXPECT: чисто. EVIDENCE: 54 строки, ok (ruff не установлен)
