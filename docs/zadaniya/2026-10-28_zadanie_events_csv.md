# Задание kodir — `data/events.csv` и загрузчик событий point-in-time (Ж-070 п. 2)

**Блок владельца** (`docs/resheniya/resheniya.md`, Ж-070):
> календарь событий data/events.csv — тикер, тип, дата объявления, дата события, источник; запись в REESTR; загрузчик отдаёт событие на день d, только если дата объявления ≤ d; тест с подложенным событием, объявленным позже

Толкование — [Ж-070] в `docs/resheniya/tolkovaniya.md`.

## Входы
- `docs/issledovaniya/2026-10-28_sobytiya_2016_2021.csv` (#392): ЦБ, IMOEX, FORTS — 449 строк.
- Дивиденды — дата объявления и дата реестра / `last_cum_date`. Читать **только через API `fastbt/dividendy.py`**. Файл не менять: его сейчас правит другой агент, который добавляет 2015–2016.

## Что сделать
1. **`data/events.csv`.**
   - Колонки: `ticker, tip, data_obyavleniya, data_sobytiya, istochnik`.
   - Только строки с `data_sobytiya` < 2022.
   - Собирается скриптом `scripts/sobrat_events.py` из входов выше. Руками csv не писать.
   - SHA файла записывается в шапку отчёта.
2. **Загрузчик `fastbt/sobytiya_pit.py`.**
   - `sobytiya_na(d, ticker=None) -> list` отдаёт событие на день d, только если `data_obyavleniya ≤ d` (Д3).
   - События с тикером `*` относятся ко всем тикерам.
   - Чтение ≥ 2022 — отказ (мутация № 17).
   - `digest()` — хэш файла, для ключей кэша (Д5).
   - **К `sobytiya_e1` не подключать**: это отдельная задача итерации 7. В отчёте назови, как устроен `sobytiya_e1` сейчас и что потребуется для подключения.
3. **Запись в `docs/otbor/REESTR_dannyh.md`** — только дописать раздел:
   - источник;
   - типы;
   - полнота;
   - «не учитывались: нет данных» (MSCI, отчётность, SPO, байбэки, публикация календаря ЦБ);
   - статус ЦБ: «объявление = решение, консервативно».

## Владение
- `data/events.csv`
- `scripts/sobrat_events.py`
- `fastbt/sobytiya_pit.py`
- `tests/test_sobytiya_pit.py`
- `docs/otchety/2026-10-28_events_csv.md`
- `docs/otbor/REESTR_dannyh.md` — только дописывание

Ветка `events-pit` от `origin/kanon-v22-pereschet`.

## Тест (`pravki.md` п. 12: тест падает без правки)
- Подложенное событие объявлено d+1 → на d не отдаётся, на d+1 отдаётся.
- Мутация «сравнение по `data_sobytiya` вместо `data_obyavleniya`» ловится.
- Мутация «`<` вместо `≤`» ловится.
- Мутация «тикер `*` не раздаётся всем» ловится.
- Мутация «чтение ≥ 2022» ловится.

## Гейты
- **G1.** Тест зелёный, четыре мутации ловятся. Вывод pytest — в тело PR.
- **G2.** `data/events.csv`: число строк по типам = числу во входах; дат ≥ 2022 нет (`grep`).
- **G3.** Скрипт воспроизводит csv побайтно: второй запуск даёт тот же SHA.
- **G4.** Раздел REESTR дописан.

## Запрещено
- Данные ≥ 2022, `test_locked/`, `.env`, метрики.
- Правка `fastbt/dividendy.py`, `fastbt/sobytiya_e1.py`.
- `pytest tests` целиком, `fastbt/tests/test_core.py`, `tests/test_a_weights_future.py`.
- `ln`, `scratchpad/main/`, `scripts/finish.sh` (git/gh — вручную), heredoc, `sed -i`, `python -c`.
- Запись вне владения.

## Отчёт и лимиты
**Отчёт:** не длиннее 10 строк, реестр «закрыто N из M», расход.

**Лимит:** не больше 2 процессов Python, около 35 минут, порог передачи 300k токенов.

Правила: `.claude/rules/polnota.md`, `testy.md`, `mutations.md`, `dannye.md`, `pravki.md`.
