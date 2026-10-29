# GATES — Ф-сл2: ядро fast_exec для осей S3/S4 (stop_zayavka+urovni, atr_m, T, reentry_den, chast_1r/chast), kodir, 29.10

## G1 — ядро = exec_e.run побайтно на ≥ 1000 синтетических путей с этими осями
CHECK: python3 -m pytest tests/test_fast_v5_s34.py -q -k "sintetika or krajnie or osi_menyayut"
EXPECT: passed, 0 failed
EVIDENCE: 3 passed (1200 путей: 600 стоп-заявок + 600 лимитных с T/reentry_den; kanal, part, trail, повтор в день ≥ 20 путей каждый)

## G2 — на 50 точках S3 и S4 (бары train4 ≤ 2019, AFLT и SBER) побайтно
CHECK: FASTBT_BARS1M=<D25/bars1m> python3 -m pytest tests/test_fast_v5_s34.py -q -k realnye
EXPECT: passed, 0 failed
EVIDENCE: 1 passed (200 прогонов = 2 тикера × 2 класса × 50; ≥ 60 из 100 торгуют > 20 сделок в каждом классе)

## G3 — мутации ловятся: заявка исполнена касанием; стоп в баре входа по уровню S; reentry в тот же бар
CHECK: python3 -m pytest tests/test_fast_v5_s34.py -q -k mutatsiya
EXPECT: passed, 0 failed
EVIDENCE: 3 passed (расхождений ≥ 3 у каждой мутации эталона; без мутации — равны)

## G4 — замер S3/S4 до/после (малая проверка, Ж-135, один процесс, AFLT train4, 20 точек)
CHECK: FASTBT_BARS1M=<D25/bars1m> python3 -m pytest tests/test_fast_v5_s34.py -q -k zamer -s
EXPECT: ядро быстрее эталона
EVIDENCE: S3 эталон 0,179 → ядро 0,001 с/точку (×124); S4 0,025 → 0,001 (×47) — потолок: без накладных `flat` (кэш дней) и без остального оценщика

## G5 — старые тесты быстрого слоя зелёные (kernel обратно совместим)
CHECK: python3 -m pytest tests/test_fast_etap1_smoke.py tests/test_fast_v5_ekviv.py tests/test_fast_etap1_equiv.py tests/test_stop_zayavki.py -q
EXPECT: 0 failed
EVIDENCE: 477 passed; 1 failed — tests/test_stop_zayavki.py::test_mutatsiya_gap_po_urovnyu_lovitsya (слой эталона, fast_exec не вызывает; красный и до правки — мутация gap упирается в инвариант #610). Не мой файл.

## Реестр
закрыто 5 из 5 (G5 — с названным красным вне владения)
