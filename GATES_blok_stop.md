# GATES — БЛОКЕР #596: стоп в баре входа вне [low; high] (29.10.2026)

G1. Тесты расследования без xfail проходят.
CHECK: python3 -m pytest -q tests/test_rassl_tf_s1.py
EXPECT: 5 passed
EVIDENCE: 5 passed in 0.39s (29.10)

G2. Тест-свойство инварианта: ≥ 1000 случайных путей, все режимы (ed_s/ed_t, chand, k_e, chast, parab, лимит, стоп-заявка, tf 1h), исполнений вне [low; high] — 0.
CHECK: python3 -m pytest -q tests/test_ispolnenie_v_bare.py
EXPECT: все passed
EVIDENCE: 14 passed in 5.09s; N_PUT = 1000 путей дневных режимов и 1000 сессий tf 1h

G3. Мутация «стоп в баре входа исполнен по уровню S» ловится тестом-свойством и проверкой в коде (ValueError).
CHECK: python3 -m pytest -q tests/test_ispolnenie_v_bare.py -k mutaciya
EXPECT: passed
EVIDENCE: 2 passed, 12 deselected

G4. Существующие тесты исполнения зелёные.
CHECK: python3 -m pytest -q tests/test_exec_e*.py tests/test_fast_etap1_smoke.py tests/audit_fastbt/test_14_stop_take_bar.py tests/audit_fastbt/test_15_gap.py tests/audit_fastbt/test_04_commission.py tests/test_tf_ispolnenie_v5.py
EXPECT: 0 failed
EVIDENCE: 679 passed in 2.29s

G5. Быстрый слой зеркально, побайтно = медленный.
CHECK: python3 -m pytest -q tests/test_fast_etap1_equiv.py tests/test_fast_v5_ekviv.py (последний — после слияния #599)
EXPECT: 0 failed
EVIDENCE: 19 passed in 156.90s (после слияния origin/kanon-v22-pereschet)

G6. Масштаб: исполнения вне [low; high] на опубликованных конфигурациях train4 до правки.
CHECK: исходники fastbt с origin/kanon-v22-pereschet + новый tests/test_fast_v5_ekviv.py: pytest -k "realnye or sintetika"
EXPECT: число вне бара
EVIDENCE: до правки: синтетика 1786 исполнений вне [low; high] эталона; реальные бары train4 (17 тикеров) — 429 пар (точка, тикер) с нарушениями. 1h, 20 точек (docs/otchety/2026-10-29_rassl_tf_s1.md): S1 49 008, S2 27 652, S6 24 814. ОТСТУПЛЕНИЕ: якорь E / A_strogaya / 50 точек zh047 отдельным masshtab.py не мерились.
