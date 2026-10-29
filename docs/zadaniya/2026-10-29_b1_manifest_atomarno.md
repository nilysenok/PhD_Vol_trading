# Б1: гонка записи manifest.json в ledger — kodir, 29.10 16:10
Бюджет ~25м · ≤ 80т. Общая часть: `docs/zadaniya/2026-10-29_obshchee_16.md`. Ветка `fix-manifest-atomarno`.

## Цель
`fastbt/ledger_v2.py` `Pisatel.__init__` пишет `manifest.json` через `write_text` без tmp+rename: несколько процессов на первых кусках одной стратегии читают полупустой файл → `JSONDecodeError` (воспроизведено автором #609: 4 из 25 прогонов). Та же гонка — в `fastbt/progon_sobol.py` `schet` (процессы стартуют на одной стратегии разом). На сервере 90+ процессов. Починить: атомарная запись (уникальный tmp + `os.replace`), чтение без гонки; поведение и формат manifest побайтно прежние.

## Владение
`fastbt/ledger_v2.py`, `fastbt/progon_sobol.py` (только место записи/чтения manifest), новый `tests/test_ledger_manifest_gonka.py`.

## Гейты (минимум)
G1 тест-гонка (N процессов/потоков пишут и читают manifest одной стратегии, ≥ 50 повторов) падает до правки. G2 после правки 0 падений на 200 повторах. G3 manifest побайтно прежний (без `sozdan`). G4 `pytest tests/test_ledger_v2.py tests/test_ledger_v21.py tests/test_ledger_v21_poz.py tests/test_progon_sobol.py tests/test_progon_sobol_v5.py tests/test_progon_server.py` зелёные. G5 мутация «вернуть write_text» роняет тест G1.
