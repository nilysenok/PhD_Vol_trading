# GATES — wf-L3: докачка 04–27.02.2026 (10-минутки) и 1-минуток 2025–02.2026, kodir, 30.10.2026

Данные — в scratchpad `.../scratchpad/wf_dannye/L3/` (в git не идут). Таблицы: `.../wf_dannye/L3_tools/tablica_parquet.csv`, `tablica_1m.csv`; проверки: `proverka.py`, `proverka_1m.py` там же.

## G1 — тесты загрузчика (синтетический ответ ISS, без сети)
CHECK: python3 -m pytest tests/test_dokachka_wf.py -q
EXPECT: 10 passed
EVIDENCE: 10 passed in 0.31s (код 0)

## G2 — мутация «не обрезает после 2026-02-28» ловится
CHECK: python3 -m pytest tests/test_dokachka_wf.py -q -k "mutaciya or posle_28"
EXPECT: 2 passed
EVIDENCE: 2 passed, 8 deselected in 0.28s (код 0)

## G3 — parquet (28 акций, 10 индексов/RVI/CNY в moex_iss, 11 в final_v6): старая часть равна, хвост позже, ничего после 28.02, новых дублей нет
CHECK: python3 .../L3_tools/proverka.py
EXPECT: staroe_ravno=True, hvost_ok=True, novye_dubli=0 у всех 60 файлов
EVIDENCE: выполнено, все 60 True/True/0; tipy_te_zhe=False только у двух CNYRUB_TOM (volume Int64: ISS отдаёт null)

## G4 — 1-минутки 2025 и 01–02.2026, 17 тикеров
CHECK: python3 .../L3_tools/proverka_1m.py
EXPECT: 34 файла, ok=True у всех (заголовок, max begin < 2026-03-01, rows и sha256 из .done, без пропусков и дублей)
EVIDENCE: файлов 34, ok 34 из 34; общих дней 2025 — 301, 2026 — 50

## G5 — USD000UTSTOM после 11.06.2024
CHECK: curl ISS candles interval=10 from=2024-06-12 till=2026-02-28
EXPECT: нет рыночных данных
EVIDENCE: только 2026-02-16, цены 70,5…76,25, value=null, volume=null — не дописано; ось usdrub — через CNYRUB_TOM (решение дирижёра 30.10 14:20)

## G6 — rate limit ≤ 2 запросов/с
CHECK: РУЧНОЙ — число запросов в секунду нельзя проверить командой после факта; процессов ≤ 2 (пауза 1,0 с), на время третьего (indeksy) пауза 5 с
EVIDENCE: пауза и число процессов заданы в коде и скриптах запуска

## Реестр
закрыто 6 из 6 (G6 ручной); ABANDON: 0; pending: 0
