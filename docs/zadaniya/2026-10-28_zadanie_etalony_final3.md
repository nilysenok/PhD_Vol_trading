# Задание kodir — пересчёт эталонов s9 `final3` и паспорта A после дивидендов (Ж-059 п. 2, `pravki.md` п. 11, Д6)

**Цель.** После правки дивидендов (PR #370, ветка `pravka-dividendy`) — новые эталоны `docs/tables/strat_dev/s9_*_final3.csv` на коде PR, старые — в архив с хэшами и причиной; оба теста `tests/test_s9_scope.py::test_dev_reproduces_final3_tables` и `::test_test_path_on_dev_years` зелёные. Затем паспорт A (train) и строка «A (все семейства)» было/стало.

**Правило информационного множества (обязательно, толкование [Ж-059 п. 2] в `docs/resheniya/tolkovaniya.md`).** Эталоны содержат годы 2020–2021 (validation, закрыта до одного взгляда в конце). Пересчёт механический: **значения за 2020–2021 не выводить на экран, не печатать, не пересказывать, не сравнивать глазами**; в отчёт — только sha256 файлов, число строк, «тест зелёный/красный». Любое сравнение «было/стало» и паспорт A — **только 2018–2019**. Если генератор печатает метрики по годам — перенаправлять вывод в файл в рабочем каталоге и не читать его.

**Что сделать.**
1. Найти, чем порождены `s9_*_final3.csv` (команда `python3 -m fastbt.s9 … --scope dev --sigma-h1 final3` или аналог — по `fastbt/s9.py`, `tests/test_s9_scope.py`, журналу `docs/` с «final3»); назвать команду в отчёте.
2. Архив: `arhiv/final3_do_dividendov_2026-10-28/` — старые файлы + `SHA256SUMS.txt` + `PRICHINA.md` (Ж-057 п. 1, Ж-059 п. 2: дивиденды в PnL, сигналы на скорректированном ряду; коммит до правки `117f4b4`). Каталог `arhiv/` не отслеживается git — архив кладётся в основную копию **только** командой `cp` из твоей копии после окончания, путь назвать; в git — только `PRICHINA.md` и `SHA256SUMS.txt` в `docs/tables/strat_dev/arhiv_final3_do_dividendov/`.
3. Пересчитать эталоны на коде ветки `pravka-dividendy` (кэш s9 пересобирается — ключ изменился; диск свободен ~139 ГБ).
4. Паспорт A (train 2018, 2019, train; Шарп, mSR, Сортино, MDD, оборот; комиссия 0,04 %) и строка «A (все семейства)» — без дивидендов / D·0,87 / полный D — в `docs/tables/diag/pasport_A_dividendy.csv` (шапка: статус «разработка 2018–2019, после правки дивидендов, потолок — даты dohod.ru сверены у 3 из 14», коммит кода, хэш входов). Если в каком-либо году train |Δ mSR A| > 0,2 — первой строкой отчёта (Ж-059).

**Владение.** Ветка `etalony-final3-div` **от `origin/pravka-dividendy`**, PR с базой `pravka-dividendy` (не `kanon-v22-pereschet`). Файлы: `docs/tables/strat_dev/s9_*_final3.csv`, `docs/tables/strat_dev/arhiv_final3_do_dividendov/{PRICHINA.md,SHA256SUMS.txt}`, `docs/tables/diag/pasport_A_dividendy.csv`, `docs/otchety/2026-10-28_etalony_final3_div.md`. Код и тесты не менять. FINAL-документы не трогать.

**Гейты** (`GATES.md`, `templates/GATES.md`): G1 архив: sha256 старых файлов в `SHA256SUMS.txt` = sha256 файлов на `117f4b4`; G2 оба теста `test_s9_scope` зелёные на ветке (вывод pytest); G3 `pytest tests/test_dividendy*.py tests/test_etap1_run_yakor.py -q` зелёный; G4 `pasport_A_dividendy.csv` есть, только годы 2018/2019/train; G5 в отчёте нет ни одного значения за 2020–2021 (`grep -E "2020|2021"` по отчёту — только в тексте правила/имён, не в числах).

**Запрещено.** Данные ≥ 2022, `test_locked/`, `.env`, `--scope test`; печать/чтение метрик 2020–2021 (см. правило выше); `pytest tests` целиком, `fastbt/tests/test_core.py`, `tests/test_a_weights_future.py`; `ln`/симлинки, `scratchpad/main/`; `scripts/finish.sh` (git/gh вручную); heredoc, `sed -i`, `python -c`.

**Отчёт.** ≤ 10 строк: первая строка — Δ mSR A > 0,2 или «нет»; реестр «закрыто N из M»; расход. Остальное — в файл отчёта.

**Лимит.** ≤ 3 процесса Python; ~60 мин; порог передачи 300k. Правила: `.claude/rules/polnota.md`, `pravki.md`, `dannye.md`, `otchety.md`.
