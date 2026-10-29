# GATES_P4 — лист П4 сетки v5: оси Л5 в быстрый слой (29.10.2026)

Владение: `fastbt/fast_exec.py`, `fastbt/fast_etap1.py`, `tests/test_fast_v5_ekviv.py`, `GATES_P4.md`.
Медленный путь (`exec_e.run`) — эталон; быстрый = медленный побайтно (допуск 0).

G1. Синтетика: быстрый ≡ медленный побайтно для каждой перенесённой оси (ed_s atr/pct, ed_t atr/pct, chand), ≥ 200 точек × 3 ленты на ось.
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k sintetika`
EXPECT: все passed, 0 failed
EVIDENCE: pending

G2. Реальные бары train4 (≤ 2019): то же, ≥ 200 точек × 3 тикера на ось.
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k realnye`
EXPECT: все passed, 0 failed
EVIDENCE: pending

G3. #488: любой режим, неизвестный быстрому слою (поле Params вне списка ядра, новое поле, лишний вход run), уходит на медленный путь; мутация «старый список исключений» ловится.
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k "neizvestn or mutaciya"`
EXPECT: все passed
EVIDENCE: pending

G4. Мутации ядра ловятся: ATR дня d в Chandelier, единица стопа = σ̂ при ed_s=atr, храповик снят.
CHECK: `python3 -m pytest tests/test_fast_v5_ekviv.py -q -k mut_yadro`
EXPECT: все passed
EVIDENCE: pending

G5. Регрессии быстрого слоя и Л5.
CHECK: `python3 -m pytest tests/test_fast_etap1_smoke.py tests/test_exec_e_v5_ed.py tests/test_v6_b.py tests/test_rodnoy_uroven.py tests/test_fast_etap1_equiv.py -q`
EXPECT: 0 failed, кроме прямо названных в отчёте (тест старого контракта «ed_s → медленный путь»)
EVIDENCE: pending

G6. Замер с/точку до/после на 50 точках train4 (оси Л5).
CHECK: `python3 -I <scratchpad>/zamer_p4.py`
EXPECT: печать «до», «после» с/точку
EVIDENCE: pending

G7. tt=3 (ATR-trailing, engine_trend): уже в numba (`trend_run`, коммит 88cb970e) — ручной гейт: переносить нечего.
CHECK: `grep -n "tt == 3" fastbt/engine_trend.py`
EXPECT: строки 48, 61 внутри `@njit trend_run`
EVIDENCE: pending
