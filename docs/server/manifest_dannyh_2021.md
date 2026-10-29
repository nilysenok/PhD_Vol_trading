# Манифест переноса данных ≤ 2021-12-31 (Ж-131 п. 2)

Статус: разработка, до аудита. Скрипт `scripts/server/manifest_perenos.py`, список `scripts/server/perenos_spisok.tsv`, тест `tests/test_manifest_perenos.py`. Принято: `manifest_dannyh_2021.tsv` (путь, sha256, байты, max_data, читатель); отвергнуто: `manifest_dannyh_2021_otvergnuto.tsv` (причина).

**Итог:** принято 145 файлов, 1039,7 МБ, max дата внутри 2021-12-30; отвергнуто 91; исключено по пути 0 (в список `test_locked` и `test1_2022_2024` не включены, скрипт отсекает их до чтения). Max дата считается по содержимому (колонки дат, одна колонка на чтении), не по имени; `retrieved_at` (дата скачивания) за дату данных не считается. Имя с 202x (2022–2029) не читается, сразу в отвергнутые (51 файл `bars1m_2022…2024`).

**Кто что читает (принято):**
- `data/fastbt/daily/ohlcv_daily_1850.parquet` — `fastbt/data.py` (все сигналы S1–S6, Соболь v5, калибратор v5, tf_os_v5, fazy).
- `data/regen/D25/bars1m/*_{2014…2021}.csv` (136 файлов, 1037,7 МБ) — `fastbt/bars5m.py` (5-минутки семьи E; 2014 — прогрев В2 с 06.2014).
- `data/regen/ch4_dev/*.parquet` (5) — σ-ряды гл. 4 (`data.py` vpred; E-семья, validation 2020–2021 ≤ 2021 — входит).
- `data/events.csv` — `sobytiya_pit.py`; `data/iss_mix_2015_2019/IMOEX*.csv` — `fazy.py`.

**Отвергнуто — находка (нужные расчётам файлы содержат 2022+):**
1. `moex_discovery/data/raw/candles_10m/*.parquet` (28 файлов, до 2026-02): источник `daily_1850.build` и `bars_agg` (ТФ для tf_os_v5). Дневной ряд 18:50 уже собран и принят; но агрегация ТФ из 10-минуток на сервере невозможна без усечённой копии.
2. `data/corporate_actions.csv` (до 2025-11): `data.corp_actions`. 3. `data/calendar/cbr.csv` (до 2026-12, график заседаний), `torgovye_dni_moex.csv` (до 2024-12), `expirations`, `fomc`, `cpi_us`, `imoex_rebal*` (до 2026): ставка ЦБ, календарь, события. 4. `cpi_ru.csv`, `dividends_actual/expected.csv` — пусты (заголовок). 5. `candles_10m_2014_2015/sverka.csv` — нет дат. 6. `bars1m_2022…2024` (51) — по имени, не читались.

**Предложение (не сделано; решает дирижёр):** отдельным шагом усечённые копии с фильтром `date <= 2021-12-31` на чтении: candles_10m (parquet), corporate_actions, cbr, торговый календарь, expirations, fomc, imoex_rebal; после усечения — прогон этого же скрипта (файл попадёт в манифест, только если max ≤ 2021). Ставка владельца `fastbt/rates/key_rate_cbr_vladelec.csv` и таблицы дивидендов `docs/issledovaniya/*dividendy*.csv` едут через git; их max даты этим скриптом не проверены (вне `data/`), нужна проверка. ❓ IMOEX за 2020–2021 в `iss_mix_2015_2019` нет (max 2019-12-30): для фаз на validation нужен другой источник ≤ 2021. ❓ σ-ряды `ch4_dev` начинаются с 2018 — для train4 (2016–2019) σ 2016–2017 из них не берётся.
