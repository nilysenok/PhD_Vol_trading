# GATES — Э1-в, режим `plato` (топ-3 по плато), T1, 27.10

Ветка `iter-plato` от `iter-e1-s1s3` + слияние `iter-e1-s4s6`. Толкование 27.10 19:35 (`docs/resheniya/tolkovaniya.md`, Ж-031 Э1-в).
Код: `fastbt/sweep_plato.py` (новый: режим), `fastbt/sweep_pts.py` (новый: уровни, канон точки, параметры с эффектом — вынесено из
`sweep_run_iter.py` из-за лимита 200 строк), правки `sweep_run.py` (флаги), `sweep_run_iter.py` (штраф Optuna, диспетчер).
`sweep_drivers.py` не правится. Числа дыма на реальных данных — механика, не выводы: до исправления знака в `sweep_post.py:162`
(ветка `fix-znak-sweep-post`) все цели S1–S6 неверны.

## G1 — одиночный пик не выбирается, плато выбирается
CHECK: python3 -m pytest tests/test_sweep_plato.py -q -k "pik or mut_plato"
EXPECT: пик (7,7) выигрывает по цели, но не по плато; мутация «плато = цель точки» возвращает пик
EVIDENCE: 13 passed (tests/test_sweep_plato.py целиком)

## G2 — соседи ±1 по параметрам с эффектом достраиваются, идут в ledger и в N; повтор не раздувает N
CHECK: python3 -m pytest tests/test_sweep_plato.py tests/test_sweep_plato_run.py -q -k "sosedi or povtornyi or skvoznoi"
EXPECT: ledger растёт на число новых соседей; N в таблице = len(ledger); мутация «соседей нет» — 0 проб
EVIDENCE: 27 passed (три файла tests/test_sweep_plato*.py)

## G3 — предел активности (сделок/год ≥ 0,5 × база tune 2018) в plato и штраф −inf в Optuna
CHECK: python3 -m pytest tests/test_sweep_plato.py tests/test_sweep_plato_run.py -q -k "predel or shtraf or nedopustimye"
EXPECT: точка с 1 сделкой отсечена; Optuna с недостижимым пределом: пробы в ledger, в таблице пусто; мутации падают
EVIDENCE: см. G2 (27 passed)

## G4 — Optuna ищет только по параметрам с размахом ≥ 0,05 из торнадо
CHECK: python3 -m pytest tests/test_sweep_plato_run.py -q -k "effect or s_effektom or vsya_setka"
EXPECT: w и z (размах < порога) в пробах постоянны; мутация «вся сетка» — w бегает
EVIDENCE: 27 passed

## G5 — неактивные параметры сведены к базе в ключе (условные блоки; ATR-ключи при √RV); дубль по канону берётся из ledger
CHECK: python3 -m pytest tests/test_sweep_plato.py -q -k "uslovnyi or atr_klyuchi or kanon or mut_fix or mut_sqrt"
EXPECT: сосед по неактивному параметру = сама точка (пропуск); мутации (fix без сведения, INERT_SQRT пуст) раздувают N
EVIDENCE: 13 passed (tests/test_sweep_plato.py)

## G6 — факторный план на двух узлах: оба узла без центра
CHECK: python3 -m pytest tests/test_sweep_plato_run.py -q -k "dva_uzla or odin_uzel"
EXPECT: 3×3×2 = 18 точек; один узел — ValueError; мутация «центр дублирован» ломает счёт
EVIDENCE: 27 passed

## G7 — `gamma_hold`: причина нулевого размаха
CHECK: python3 -m pytest tests/test_sweep_plato_rm.py -q
EXPECT: параметр доходит до ядра (при max_hold=5 на реальных SBER/LKOH/AFLT S1 сроки 5→4/3 и т.д.); при базовом max_hold=20 (S1/S2) и 30 (S3/S4) временной стоп не успевает сработать — не дефект кода; мутация «gamma не передан» ловится
EVIDENCE: 2 passed (test_sweep_plato_rm.py: тест на коде + мутация)

## G8 — дым на реальном train 2018/2019, S1–S6 (механика; числа — не выводы)
CHECK: FASTBT_MAIN=$PWD python3 -m fastbt.sweep_run fastbt/designs/sweep/S{k}_setka_e1b.yaml --mode plato --out <копия docs/tables/sweep_e1 в scratchpad> --jobs 2
EXPECT: плато, ≥ 3 соседей, предел активности = 0,5 × база (S4 5,14; S5 6,83; S6 8,70), ≤ 60 с на стратегию
EVIDENCE: S1 2 с (244 соседних точек, проб в ledger 190→396), S2 3 с, S3 4 с, S4 6 с, S5 21 с, S6 5 с; топ-10 кандидатов каждой с 22–29 соседями
