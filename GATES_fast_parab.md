# GATES Ф-parab (быстрый слой: parab в ядре)

G1 — ≥ 1000 синтетических путей с parab (лонг/шорт, лимит и стоп-заявка, тейк) побайтно
CHECK: NUMBA_CACHE_DIR=<свежий> python3 -m pytest tests/test_fast_parab.py -k 1200 -q
EXPECT: 1 passed (1200 путей, trail-выходов ≥ 200)
EVIDENCE: 10 passed, 2 skipped (весь файл), см. отчёт

G2 — гэп через параболический стоп; тейк и стоп в одном баре; стоп бара u по u−1
CHECK: pytest tests/test_fast_parab.py -k "gep or tejk or stop_bara" -q
EXPECT: passed
EVIDENCE: passed (в составе G1-запуска)

G3 — 200 точек S3 и S4 на уровне exec_e.run (t_vyhod = close) побайтно (на слиянии с #699)
CHECK: FASTBT_V5_N=200 pytest tests/test_fast_parab.py -k kub -s
EXPECT: passed, parab-вызовов ядра > 0
EVIDENCE: pending — см. отчёт

G4 — мутации ловятся: AF не растёт при новом экстремуме; стоп бара u по бару u; гэп по уровню
CHECK: pytest tests/test_fast_parab.py -k mutatsiya -q; плюс ручная правка ядра в чистом NUMBA_CACHE_DIR
EXPECT: тесты проходят (мутации эталона ловятся); с ядерной мутацией 1200-тест падает
EVIDENCE: все три ядерные мутации валят тест (проверено вручную, правка возвращена)

G5 — флаг покрытия YADRO_PARAB_GOTOV
CHECK: pytest tests/test_fast_parab.py -k pokryto -q
EXPECT: passed
EVIDENCE: passed
