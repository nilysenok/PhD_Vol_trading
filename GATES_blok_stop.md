# GATES — БЛОКЕР #596: стоп в баре входа вне [low; high] (29.10.2026)

G1. Тесты расследования без xfail проходят.
CHECK: python3 -m pytest -q tests/test_rassl_tf_s1.py
EXPECT: 5 passed
EVIDENCE: pending

G2. Тест-свойство инварианта: ≥ 1000 случайных путей, все режимы (ed_s/ed_t, chand, k_e, chast, parab, лимит, стоп-заявка, tf 1h), исполнений вне [low; high] — 0.
CHECK: python3 -m pytest -q tests/test_ispolnenie_v_bare.py
EXPECT: все passed
EVIDENCE: pending

G3. Мутация «стоп в баре входа исполнен по уровню S» ловится тестом-свойством и проверкой в коде (ValueError).
CHECK: python3 -m pytest -q tests/test_ispolnenie_v_bare.py -k mutaciya
EXPECT: passed
EVIDENCE: pending

G4. Существующие тесты исполнения зелёные.
CHECK: python3 -m pytest -q tests/test_exec_e*.py tests/test_fast_etap1_smoke.py tests/audit_fastbt/test_14_stop_take_bar.py tests/audit_fastbt/test_15_gap.py tests/audit_fastbt/test_04_commission.py tests/test_tf_ispolnenie_v5.py
EXPECT: 0 failed
EVIDENCE: pending

G5. Быстрый слой зеркально, побайтно = медленный.
CHECK: python3 -m pytest -q tests/test_fast_etap1_equiv.py tests/test_fast_v5_ekviv.py (последний — после слияния #599)
EXPECT: 0 failed
EVIDENCE: pending

G6. Масштаб: исполнения вне [low; high] на опубликованных конфигурациях train4 до правки.
CHECK: masshtab.py (scratchpad) на снимке базы, S1–S6: якорь E, A_strogaya, 50 точек zh047
EXPECT: число вне бара по каждой конфигурации
EVIDENCE: pending
