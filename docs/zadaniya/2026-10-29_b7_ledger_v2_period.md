# Б7: ledger читает и пишет период В2 (2015–2019) — kodir, 29.10 16:35
Бюджет ~35м · ≤ 100т. Общая часть: `docs/zadaniya/2026-10-29_obshchee_16.md`. Ветка `ledger-period-v2`.

## Цель
`fastbt/ledger_v2.py`: `GODY = 2016–2019`, `NSEG = 16` — `Pisatel21` падает на датах 2015, `Chitatel` отвергает `by_year` вне 2016–2019 (находки #608, #609, #616). Период В2 (Ж-113; `period_v2.START` = 2014-08-26, окно с 2015-01-01) поэтому идёт только в ledger 2.0, а сводка (`scripts/server/svodka_ledger.py`, #616) и обработка (`fastbt/svod_server_v5.py`, #615) на реальном каталоге В2 откажут. Сделать: годы и сегменты ledger берутся из периода в manifest (train4 — как сейчас, побайтно; v2 — 2015–2019); `Chitatel` читает оба; драйвер (`fastbt/progon_sobol.py` `proverit_ledger`) перестаёт отказывать v2 в 2.1 — только если правка ledger это делает корректным.

## Владение
`fastbt/ledger_v2.py`, `fastbt/progon_sobol.py` (только `proverit_ledger` и передача периода в ledger), новый `tests/test_ledger_period_v2.py`.

## Гейты (минимум)
G1 тест: синтетический ledger v2 с датами 2015 — 2.1 пишется и читается (падает до правки). G2 train4: ledger 2.1 побайтно прежний (шарды и manifest без `sozdan`). G3 `Chitatel.by_year`/`okno` на v2 возвращает 2015 корректно (известный ответ). G4 сводка `svodka_ledger.py` и `svod_server_v5.py` на синтетическом каталоге v2 (через `progon_server` на синтетике) — без отказа. G5 мутация «год 2015 отрезан» ловится. G6 `pytest tests/test_ledger_v2.py tests/test_ledger_v21.py tests/test_ledger_v21_poz.py tests/test_ledger_manifest_gonka.py tests/test_progon_sobol.py tests/test_progon_sobol_v5.py tests/test_progon_server.py tests/test_svodka_ledger_v5.py tests/test_svod_server_v5.py` зелёные.
