# Верификация входов теста главы 4 (26.10.2026, 12:40)

**Все 30 проверок — OK, НЕ OK нет.** Тест (`scripts/ch4_test_run.sh`, 26.10 10:32–10:38, коммит `25b76aa`) читал: σ_pred h=1 = √rv_pred финала главы 3 (D22_h1), σ_pred h=5 = √rv_pred D12_h5, HAR-J = √har_corr теста главы 3. Все три совпадают побитно на всех строках 2017–03.02.2026. Код теста по хэшам совпадает с FINAL_strategies. Ставка ЦБ фактическая, комиссия 0,05%, маска `[0]`, все шесть исправлений включены. Замечаний семь (раздел 3), на числа теста ни одно не влияет.

Поручение владельца 12:35, п. 4. Полного блока «11:15» не было, проверка шла по списку дирижёра (7 пунктов). Тест не запускался, ничего не пересчитывалось. Читались только файлы данных и кода, а таблицы теста — только на NaN в столбцах p. Ветка `ch4-verif-vhodov` от `kanon-v22-pereschet` (`f84342c`). Код и конфиги на `25b76aa` и на `f84342c` одинаковы: `git diff --stat 25b76aa HEAD -- fastbt scripts/ch4_test_run.sh …` пуст. `fastbt/` в основной копии, из которой шёл тест, совпадает с коммитом: `diff -rq -x __pycache__` пуст. Сеть не использовалась.

Скрипты проверки (только читают):
- `scripts/analysis/verif_vhodov_gl4_sigma.py` — п. 1–3;
- `scripts/analysis/verif_vhodov_gl4_dannye.py` — п. 4;
- `scripts/analysis/verif_vhodov_gl4_kesh.py` — п. 6.

## 1. Таблица проверок

| № | Проверка | Ожидание | Факт | Итог |
|---|---|---|---|---|
| 1.1 | `vpred` в `fastbt/designs/final/{main,d_binary,d_cont}.yaml` | `data/regen/sigma_pred_h1_final3_full.parquet` | во всех трёх — `/Users/…/PhD_Vol_trading/data/regen/sigma_pred_h1_final3_full.parquet` | OK |
| 1.2 | SHA-256 σ_pred h=1 | `5bed7be9eb7ebcfb0466271e3f06670c13bda380d06be1db53828c787b6b0dbb` (FINAL, поле 1) | `5bed7be9eb7ebcfb0466271e3f06670c13bda380d06be1db53828c787b6b0dbb`; файл изменён 26.10 в 10:00, то есть до теста | OK |
| 1.3 | Источник σ h=1 — тест финала главы 3 | `test_locked/final3/final_h1_test.parquet`, SHA `4e56f625…f4e7d0` (FINAL_DESIGN гл. 3, поле 10), дизайн `top5\|V0_XGB\|B-sm\|L0` | SHA `4e56f6252a48cb47f6027a0ff86c1f197102a5e02f61123265287b119cf4e7d0`, `design` = `top5\|V0_XGB\|B-sm\|L0`, `h` = 1 | OK |
| 1.4 | σ h=1 = √rv_pred побитно (float32, сортировка date, ticker — как в фазе build `final3.py`) | все строки 2017–02.2026 | 39 947 из 39 947 побитно (`view('u4')`), внешнее соединение без лишних строк, NaN 0. По годам 2017…2026: 4284 / 4318 / 4284 / 4250 / 4335 / 3972 / 4318 / 4348 / 5379 / 459 — все равны. Повтор кода сборки в памяти: `DataFrame.equals` = True | OK |
| 2.1 | HAR-J h=1: SHA | `406008a7…df93` (FINAL, поле 2) | `406008a7c6e701c2f6e5c5c3e5ba469a7da195674949066bcfe9872115f4df93` | OK |
| 2.2 | HAR-J = √har_corr теста главы 3 побитно (`har_corr` — эталон HAR-J с поправкой sm, FINAL_DESIGN гл. 3, раздел 3) | все строки | 39 947 из 39 947 побитно, по всем годам 2017–2026 | OK |
| 3.1 | Откуда σ h=5 у C1 W504 p70 | `fastbt.s9` | `s9.SOURCES["hyb_h5"]` = `data/regen/sigma_pred_h5_final2.parquet`, столбец `sigma_pred_h5`, с конфигом `horizons: [h5]` (`cfg5` в `_job`); точка теста `TEST_C = ("C1:W504_p70", "hyb_h5")`. В `fastbt.final_test` h=5 нет | OK |
| 3.2 | σ h=5: SHA и равенство D12_h5 | SHA `673b3d02…571d`; = √rv_pred `final_h5_test.parquet` (SHA гл. 3 `46949913…`, `wrap2\|V0_XGB\|B-sm\|L0`) | SHA `673b3d02d2af765c9f20033a6d21451b2122ca4ed73f8f50db27c38f3211571d`. Источник: SHA `4694991372bd95d3e2a7a091c2548444d77050ca4c75cc40170f5bd78b8652fc`, `design` = `wrap2\|V0_XGB\|B-sm\|L0`, `h` = 5. Побитно 39 951 из 39 951 по всем годам | OK |
| 4.1 | Источник и период баров | `ohlcv_daily_full.parquet` (FINAL 8.1) | `moex_volatility/results/final/strategies/data/ohlcv_daily_full.parquet` (`fastbt.data.OHLCV`; `FASTBT_DATA_ROOT` в скрипте теста не задаётся). SHA `5453fb8f…a400`, файл от 23.02.2026. 17 тикеров, 09.06.2014–03.02.2026, 50 906 строк. Свечи 10 мин ISS свёрнуты по дате (`docs/fastbt_spec.md`, п. 1). NaN 0, дублей 0, close ≤ 0 нет, high < low нет, нулевого объёма нет | OK |
| 4.2 | Совпадение календаря с данными главы 3 (`features_pit_v3.parquet`, SHA `8c51bfa1…` = FINAL_DESIGN гл. 3, поле 4) | одинаковые пары тикер × день | 50 906 из 50 906 пар в обоих файлах, лишних нет | OK |
| 4.3 | Таблица торговых дней 8.1 FINAL_strategies | воспроизводится | воспроизведена полностью. Даты: 252 / 254 / 252 / 250 / 255 / 234 / 254 / 256 / 318 / 27. Пары: 4284 / 4318 / 4284 / 4250 / 4335 / 3972 / 4318 / 4348 / 5383 / 459. Мин. и макс. дней на тикер, выходные даты (0/3/0/0/1/0/0/3/65/6), даты и строки σ — как в FINAL | OK |
| 4.4 | Нет 28.02–23.03.2022 | баров нет | баров 28.02–23.03.2022 — 0. Последний бар — 25.02 у всех тикеров. Первый после разрыва — 24.03 (14 тикеров) и 28.03 (LSRG, MTLR, OGKB) | OK (замечание 3б) |
| 4.5 | Пары бар × σ на 2017–2026 | бар без σ — только 4 субботы 2025 (FINAL 8.1) | у 39 947 пар есть и бар, и σ. Без σ ровно 4 бара: AFLT 16.08.2025, MOEX и VTBR 29.03.2025, RTKM 12.04.2025. σ без бара нет | OK |
| 4.6 | Цены баров против close главы 3 | разные агрегации, совпадения не требуется | close равен в 59% строк. До 2019 — 98–100%, с 22.06.2020 (запуск вечерней сессии) — 3–14% по годам, медиана расхождения 0,2–0,3%. Корреляция дневных лог-доходностей 2017–2026 — 0,926 | OK (замечание 3в) |
| 5.1 | Ставка ЦБ фактическая (`fix_rate`) | `msp.rate_cbr()` по решениям ЦБ | `key_rate_cbr.parquet` (SHA `46f22134…8c57`, до 06.02.2026). Ступени 2022–2025 совпадают с решениями ЦБ: 9,5 → 20 (28.02.2022) → … → 21 (28.10.2024) → … → 16 (22.12.2025). Сверено по памяти, без сети. В журнале теста средняя ставка 10,618 / 9,945 / 17,5 / 19,175 / 16,0 (2022…2026), `rate_cbr` даёт ровно эти числа. Восстановленная ставка (`rate_restored`) дала бы в 2025 году 21,0 | OK |
| 5.2 | Комиссия 0,05% | `comm` = `msp_comm` = 0,0005 | 0,0005 во всех трёх yaml, в `Config` по умолчанию, `s9.COMM`, `chapter4_a` (`cfg.msp_comm`). Проверка по журналу `chapter4_a` (маска `[0]`): комиссия = оборот/день × дни × 0,0005. 2022: 3,18% × 234 × 0,0005 = 0,37% (в журнале 0,37). 2023: 3,58% × 254 × 0,0005 = 0,45% (в журнале 0,45) | OK |
| 5.3 | Маска `[0]` | везде, кроме строки «до/после утечки» | `dev.make_config` ставит `mask_old=False` для `final_test` и всех этапов С9. `chapter4_a` берёт `Config()` (маска `[0]`), маску `[1]` — только второй вызов с `--leak-mask-old`, ради `tabl_4_a_utechka`. В журнале у основных строк вариант «маска [0]» | OK |
| 5.4 | `fix_c_s12_val` | валидация C у S1/S2 по полным позициям A | `approach_cd.run_c`: обрезка A до тестового года только при `not cfg.fix_c_s12_val`; флаг `true` | OK |
| 5.5 | `fix_b_median` | медиана σ в B только по прошлому | `approach_b`: `lim = year_bounds(cal_end_year(Y))[0]`. В тесте `val_end` = 2021, значит только бары до 01.01.2022 (окно заморожено) | OK |
| 5.6 | `fix_d_units` | σ в D в годовых единицах | `masks._d_gates` и `approach_cd.d_cont_scales`: σ·√252. σ_pred в дневных единицах (медиана 0,0172), так что двойного перевода нет | OK |
| 5.7 | `fix_rate` | ставка по решениям ЦБ | `report.rate_for`, `final_test.tables`, `margin` ставят `rate_cbr` при `fix_rate`. В `s9`, `chapter4_a`, `decomp` — всегда `rate_cbr`. Числа — п. 5.1 | OK |
| 5.8 | `fix_sharpe` | Шарп = mean/std избыточной доходности | `msp.sharpe(excess)` = mean·√252 / std(ddof=1) — для всех таблиц `final_test`, С9 и A | OK (замечание 3г) |
| 5.9 | `fix_c_fill` | нет допустимой конфигурации C → C = A | `run_c`: `if v41 or cfg.fix_c_fill: out[Y] = ("A", …)`. В тестовых позициях `data/fastbt/final_test_main.parquet` (10:32:11) по 510 группам тикер × стратегия × год (17 × 6 × 5) строк C столько же, сколько A, пропусков C, B, D нет | OK |
| 5.10 | Шесть `fix_*` в тестовом прогоне | все `true` | все шесть `true` в трёх yaml и в `Config` по умолчанию. `final_test` и С9 (`_job`: `dev.make_config(main.yaml)`) берут их из yaml, `chapter4_a` — из `Config()` | OK |
| 6.1 | Что видно в журнале `ch4_test_run.log` | конфиги и σ теста | `tabl_4_final_vybor`: `fastbt/designs/final/{main,d_binary,d_cont}.yaml`. С9: «σ-источники (тест): sigma_naive_rvw_test.parquet rvi_1850_test.parquet», кэш `cache_682e4ce9eb.pkl`. Журнал С9 `data/fastbt/s9/run_log_final3_test.json` (10:38:01): `"sigma_h1": ".../sigma_pred_h1_final3_full.parquet"`. Явного списка прочитанных σ-файлов у `final_test` в журнале нет | OK |
| 6.2 | Какие σ прочитал кэш С9 (по ключу кэша) | `final3_full` + h5 + HAR-J | ключ пересчитан из хэшей файлов. С `final3_full` (и с побитно равным ему `final3`) — `682e4ce9eb`, как в журнале. С `final2` было бы `ec152bcaef`, с `FINAL` — `67e8d527ee`. Значит, кэш теста собран из `sigma_pred_h1_final3_full`, `sigma_pred_h5_final2` и `sigma_pred_h1_harj_final2` в их нынешнем виде, при `code_hash` `1d3b7a57…` и нынешнем `main.yaml` | OK |
| 6.3 | Какой σ прочитал `final_test` | `vpred` из yaml = `final3_full` | косвенно, но однозначно. Сверка С9 в журнале: «C: Шарп пайплайн −0.8265 / движок −0.8265, max\|Δexcess\| 3.17e-10». Пайплайн — позиции `final_test_main.parquet` того же прогона, движок — кэш на `final3_full` (п. 6.2). На final2 в разработке эта сверка расходилась на 0,14 (−0,47 / −0,61, FINAL, раздел 3). HAR-J передан явно (`--harj-vpred`). Пул `spawn` создаётся после `os.environ["FASTBT_VPRED"]` для каждого дизайна отдельно | OK |
| 6.4 | Нет чтения final2 (старый гибрид h=1, `sigma_pred_h1_final2.parquet`) | не читается | в журнале «final2» не встречается ни разу. В тестовой ветке `s9.build_sources` выходит до чтения `sigma_pred_h1_final2_steps.csv` и seed-файлов D12_h1, seed-источники из `SOURCES` удалены (`set_scope`). Ключ кэша исключает final2 (п. 6.2). Слово «final2» есть только в именах файлов h=5 и HAR-J, содержимое которых — тест главы 3 (п. 2–3, замечание 3е) | OK |
| 7.1 | Хэши кода из FINAL (поле 9) на `25b76aa` | 17 файлов | все 17 совпадают: `s9.py` `a66fa476…86adb5`, `final_test.py` `d57e7ef5…409cdc4a`, `chapter4_a.py` `16d24307…f5588c856`, `decomp.py` `9b5f0472…45932e1`, `bench.py` `3b64ca86…8688fc`, `config.py` `3b02417b…49888`, `dev.py` `af55b948…74cd3`, `approach_b.py` `9d52b774…c8bea2`, `approach_cd.py` `d37cad5c…e326e`, `turnover.py` `a576b468…8fce4e`, `ch4_test_run.sh` `8cb2f6a2…38affb`, `test_a_weights_future.py` `6e3cbff3…0b891e`, `main.yaml` `4920ddec…c9d782f`, `d_binary.yaml` `3ba60fc1…62fdd`, `d_cont.yaml` `63882b4a…80e089`, `sigma_pred_h1_final3_full.py` `7e4e3a58…8f9296`, `sigma_pred_h1_v2.py` `9f86f3c2…7da163db` | OK |
| 7.2 | `fastbt.cache.code_hash` (весь пакет, кроме тестов) | `1d3b7a57…860c10` | `1d3b7a5720a45c6d335e818b0df44630ea860c10`. Версии: Python 3.9.6, numpy 2.0.2, pandas 2.3.3, numba 0.60.0, scipy 1.13.1 — как в поле 10 | OK |

Итого 30 строк — 30 OK, НЕ OK нет. В подсчёт не входят: замок (`FINAL_DESIGN_strategies.md` создан в `09685fb`, предке `25b76aa`, после этого не менялся) и согласованность окон (`SCOPES["test"]`: годы 2022–2026, `val_start` 2018, `val_end` 2021; σ до 03.02.2026, поэтому B/C/D не урезаны до 2025 — строки «σ_pred до …» в журнале нет).

## 2. Влияние на результаты теста

НЕ OK нет, поправлять числа теста не нужно. Тест шёл на тех σ, коде и параметрах, что зафиксированы в FINAL_strategies.

## 3. Замечания (на числа теста не влияют)

- **а.** Шапка и врезка `FINAL_DESIGN_strategies.md` после переименования остались прежними: «ЧЕРНОВИК (не снимает замок…)» и «`FINAL_DESIGN_strategies.md` не создавался, `--scope test` не запускался». Это противоречит факту: файл создан (`09685fb`) и снял замок для теста. Править формулировку — решение дирижёра. Вносить правку в FINAL после теста — вопрос протокола.
- **б.** В FINAL 8.1 сказано: «Закрытие биржи 25.02–24.03.2022 в данных отсутствует». На деле бары 25.02 и 24.03 есть, нет 28.02–23.03 (у LSRG, MTLR, OGKB — до 25.03, первый бар 28.03). Разрыв 25.02 → 24.03 в 27 календарных дней указан верно.
- **в.** Close баров главы 4 — последняя 10-минутная свеча дня с вечерней сессией (`fastbt_spec`, п. 1). Close в `features_pit_v3` главы 3 построен иначе и с 22.06.2020 систематически отличается (медиана 0,2–0,3%, максимум 20,7% — 24.02.2022). Календарь совпадает полностью, σ к барам привязана по дате без сдвига. В стратегиях close из `features_pit_v3` не используется (С9 берёт из него только `rv_w` и `close_RVI`). Стоит оговорить в тексте: доходности стратегий считаются по закрытию дня с вечерней сессией. Дивиденды не корректируются: строки `question` в `data/corporate_actions.csv` не применяются, доходности ценовые. Это известное ограничение, а не вход теста.
- **г.** Флаг `fix_sharpe` нигде в коде не читается. Шарп всегда считается как mean/std избыточной доходности (`msp.sharpe`), то есть исправление включено безусловно и выключить его флагом нельзя. Для теста это ожидаемое поведение.
- **д.** В журнале 1 737 предупреждений `RuntimeWarning` (divide by zero / overflow / invalid в matmul) из `fastbt/s9_stats.py:89` (`_hac`, бутстрап Ледуа–Вольфа). Ошибок и трассировок нет, прогон завершился («готово 10:38:11»). В основной таблице `s9_osnovnye_final3_test.csv` p не определён только у строки A — базы сравнения. Разбирать, есть ли NaN p у других строк справочных таблиц (P2: 6 из 8), — отдельная задача, не проверка входов.
- **е.** Имена `sigma_pred_h5_final2.parquet` и `sigma_pred_h1_harj_final2.parquet` исторические: по содержимому это побитно ряды теста главы 3 (final3), п. 2–3. Чтобы не путать с отвергнутым `sigma_pred_h1_final2`, в тексте и подписях лучше ссылаться на SHA.
- **ж.** `fastbt.data.VPRED` вычисляется при импорте. Главный процесс `final_test` импортирует `fastbt.data` до установки `FASTBT_VPRED`, но σ читают только процессы пула `spawn`, а они создаются после установки переменной (`workers` = 10). При `--workers 1` главный процесс взял бы σ по умолчанию (`vpred_aligned.parquet`). Тест шёл с `--workers 10`, сверка п. 6.3 это подтверждает. На будущее хрупко.

## 4. Команды

```
# 1.1, 7.1–7.2
grep vpred fastbt/designs/final/*.yaml
git diff --stat 25b76aa HEAD -- fastbt scripts/ch4_test_run.sh scripts/analysis/sigma_pred_h1_final3_full.py scripts/analysis/sigma_pred_h1_v2.py tests/test_a_weights_future.py   # пусто
diff -rq -x __pycache__ fastbt /Users/nilysenok/Desktop/Старое/PhD_Vol_trading/fastbt                                                                                  # пусто
for f in fastbt/{s9,final_test,chapter4_a,decomp,bench,config,dev,approach_b,approach_cd,turnover}.py scripts/ch4_test_run.sh tests/test_a_weights_future.py \
         fastbt/designs/final/{main,d_binary,d_cont}.yaml scripts/analysis/sigma_pred_h1_{final3_full,v2}.py; do shasum -a 256 $f; done
python3 -c "import fastbt.cache as c; print(c.code_hash())"
# 1.2–3.2
shasum -a 256 data/regen/sigma_pred_h1_{final3_full,final3,harj_final2,FINAL,final2}.parquet data/regen/sigma_pred_h5_final2.parquet data/regen/test_locked/final3/*.parquet
python3 scripts/analysis/verif_vhodov_gl4_sigma.py          # ИТОГ {'h1': True, 'harj': True, 'h5': True}
# 4.1–4.6
python3 scripts/analysis/verif_vhodov_gl4_dannye.py
# 5.1 (ставка): средняя по календарным дням, как в margin.sharpe_vs_rate
python3 -c "import pandas as pd; from fastbt import msp; r=msp.rate_cbr(); [print(y, round(r.reindex(r.index.union(d:=pd.date_range(f'{y}-01-01',f'{y}-12-31'))).sort_index().ffill().bfill().reindex(d).mean()*100,3)) for y in range(2022,2027)]"
# 5.4–5.10
grep -n "fix_c_s12_val\|fix_b_median\|fix_d_units\|fix_c_fill\|fix_rate\|fix_sharpe\|mask_old" fastbt/*.py
# 6.1–6.4
grep -vE "RuntimeWarning|G = y" docs/otchety/ch4_test_run.log | grep -nE "yaml|parquet|pkl|сверка"; grep -c final2 docs/otchety/ch4_test_run.log   # 0
cat /Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/fastbt/s9/run_log_final3_test.json
PYTHONPATH=. python3 scripts/analysis/verif_vhodov_gl4_kesh.py   # final3_full -> cache_682e4ce9eb.pkl
```
