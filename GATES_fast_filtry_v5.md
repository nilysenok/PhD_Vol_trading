# GATES — Ф-сл4: фильтры v5 и ранги в numba

G1 побайтно на синтетике и данных ≤ 2021 (17 тикеров)
CHECK: python3 -m pytest tests/test_fast_filtry_v5.py -q
EXPECT: 61 passed
EVIDENCE: 61 passed in 1.85s

G2 текущий день не входит; мутации № 21, 22 ловятся
CHECK: python3 -m pytest tests/test_fast_filtry_v5.py -q -k "mutatsiya or bez_tekushchego"
EXPECT: 4 passed
EVIDENCE: входит в прогон G1 (4 теста зелёные: мутант-эталон с днём d не равен numba-версии)

G3 замер до/после (малая проверка, 17 тикеров ≤ 2021, один процесс, лучшее из 3)
CHECK: pytest scratchpad/zamer.py -s
EXPECT: числа до/после
EVIDENCE: calc_adx ×17: 25,1 → 0,7 мс (×34); maska_atr_expand ×17: 38,9 → 6,8 мс (×5,7); rang srednee 17×1500: 208 → 130 мс (×1,6)
