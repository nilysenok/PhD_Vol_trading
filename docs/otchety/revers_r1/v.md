# R-в: овернайт против внутридневной части (реверс раунд 1, справочно, без выбора)

Статус: код готов, `train4` воспроизводит образец #487 (`docs/tables/overnajt/razlozhenie.csv`, допуск 1e-9, «СОВПАЛО»). Числа на `test1` (2022–2024) не считались: замок Z2 закрыт, лист R-P (период `test1` в `etap1`/`a_strogaya`) не слит, IMOEX зоны Z2 не указан.

Запуск после снятия замка и слияния R-P: `FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 scripts/revers/r1_v.py --period test1 --imoex <IMOEX 2021–2024 зоны Z2>` → `docs/tables/revers_r1/v_overnajt.csv`.
Абзац с числами — после прогона.
