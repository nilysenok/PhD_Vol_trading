# GATES V6-10: стопы и тейки пивота S5, стоп VWAP S6 (движок)

Все CHECK — `python3 -m pytest tests/test_exec_e_piv.py -k <ключ>` (NUMBA_CACHE_DIR свежий), EXPECT: passed.

G1. «Выкл.» (piv_sl, piv_tp = None) = v5 побайтно. -k vykl. EVIDENCE: 1 passed; регрессия tests/test_exec_e*.py test_fast_v5*.py test_fast_etap1*.py: 747 passed, 10 skipped.
G2. Стоп next/half/буфер, тейк P/R/half, лонг и шорт. -k "stop or tejk". EVIDENCE: passed (все 29 в файле: 29 passed).
G3. Стоп VWAP, лонг и шорт, полоса выше цены как в v4. -k vwap. EVIDENCE: passed.
G4. Уровни дня сигнала (не d+1), гэп по open, инвариант [low; high]. -k "urovni or gep". EVIDENCE: passed.
G5. Мутации ловятся (d+1; сторона; буфер; гэп по уровню; нижняя граница). -k mutatsiya. EVIDENCE: 5 passed (тесты-ловцы).
G6. Быстрый слой: ne_znaet = [piv_sl, piv_tp], NotImplementedError, fast_etap1.run = эталон. -k bystryj. EVIDENCE: passed.
G7. py_compile ок; `cargo xtask check` и `scripts/finish.sh` в этом репозитории отсутствуют (xtask/ нет) — ABANDON: G7-xtask, нет инструмента.
