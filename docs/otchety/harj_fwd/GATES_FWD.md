# GATES_FWD — гейты листов Л1–Л6 (проверка гибрида SBER на новом периоде, Ж-247/Ж-248)

Формат: `templates/GATES.md`. Каждый лист пишет только свой раздел.

## Л1. Источник, календарь, K1 (11.11.2026)

Код: `models_v2/s17/{__init__,fwd_src,fwd_cal,fwd_k1}.py`; тесты `tests/test_fwd_l1.py` (реальный архив + begin новых файлов), `tests/test_fwd_l1_synth.py` (синтетика); выходы `data/fwd/models/calendar.parquet` (все канонические дни, склейка архив + новые, только begin) и `k1.parquet` (только архивные строки, 2272). Цены новых файлов не читались.

| Гейт | CHECK | EXPECT | EVIDENCE |
|---|---|---|---|
| L1.1 архивный дневной ряд = `daily_SBER` | `python3 -m pytest tests/test_fwd_l1.py -k arhivnyj_dnevnoj -p no:cacheprovider` | индекс и колонки n_bars, rv, bv, j, seg, rv_w, rv_m побитно | passed (весь файл: 17 passed, rc 0) |
| L1.2 архивные дни и t⁺ | `… -k arhivnye_dni_i_t_plus` | дни по begin = индекс `daily_SBER`; t⁺ = `target_date` `har_SBER` для всех строк | passed |
| L1.3 K1 = `har_SBER` побитно | `… -k k1_arhiv_pobitno` | 2272 строки, все 14 колонок равны (NaN по месту) | passed |
| L1.4 календарь нового периода | `… -k kalendar_novogo_perioda` | 180 дней; 2026-11-04 есть; 2026-05-01, 2026-06-12 нет; первая пара (2026-02-27, 03-02), последняя (11-09, 11-10); n_new = 180 | passed; `python3 -m models_v2.s17.fwd_cal --write --check` → `n_new=180`, rc 0 |
| L1.5 выходы | `… -k calendar_parquet`; `python3 -m models_v2.s17.fwd_k1 --write --check` | `calendar.parquet`, `k1.parquet` равны пересчёту | passed; `k1 archive rows=2272`, rc 0 |
| L1.6 обрезка | `… -k corrupt_after`, `-k sdvig_dnej`; synth `-k obrezka` | порча баров SBER и IMOEX (узлы Л1; остальные 16 — в Л2) после d0 = 2025-12-30 не меняет строки с t⁺ ≤ d0; вечерние бары d и дни после d не входят | passed (реальный архив и синтетика) |
| L1.7 цены до замка | `… -k cena_novogo_fajla_do_zamka`; synth `-k zamok_novye_ceny` | `SealedError` на `bars`, `build_k1` (cut в новом периоде), `sealed_rv`; новые файлы читаются только с `columns=["begin"]`; `rv_fwd_SEALED` не открывался; с замком (синтетика) строки нового периода = эталонный `build_daily` + `series` | passed |
| L1.8 мутации | synth `-k mutaciya`; real `-k mutaciya_cut` | «день без IMOEX оставлен», «бар IMOEX вне основной сессии = сессия», «замок не проверяется», «пары по d ≤ cut» (№ 142), «обрезка по cut снята» ловятся | passed |
| L1.9 гигиена | `python3 -m py_compile models_v2/s17/*.py tests/test_fwd_l1*.py`; `wc -l` | rc 0; файлы ≤ 200 строк | rc 0; 9 / 70 / 90 / 46; 143 / 152 |

Закрыто 9 из 9.

Отступления и оговорки Л1:
- ↺ Календарь по begin не проверяет `rv > 0` (нужны цены): это делает скрипт открытия полным `build_daily` (протокол, раздел 3). На архиве begin-календарь = индекс `daily_SBER` (0 расхождений), то есть дней с `rv <= 0` в архиве нет.
- `k1.parquet` содержит только архивные строки (d ≤ 2026-02-26, t⁺ ≤ 2026-02-27); пара (2026-02-27, 03-02) и далее считаются в открытии.
- Первая команда исследования — `python3 -I -c` для схем parquet (запрещённая форма, правило задания; ModuleNotFound, ничего не прочитано и не записано); дальше только скрипты в каталоге сессии.

## Л3. Склад, X_w5, z энкодера 2026, e_d, s_d (11.11.2026)

Код: `models_v2/s17/{fwd_store,fwd_z}.py` (архивные `v7_0_seq.build_store`, `run_v7.sequences`, `de_e1_state.states`, `x1_ya_feat.features`, `v7_1e_rep.encode/transform_pca` — импортом; обучения и пересчёта PCA нет); тесты `tests/test_fwd_l3.py` (реальный архив), `tests/test_fwd_l3_mut.py` (обрезка, AST, мутации, замок). Выходы `data/fwd/models/{store,X_w5}.npz`, `z_tcn2026_s{0..4}.npz`, `state_es.parquet` — только архивные строки (cut = 2026-02-27, 2944 дня; z — 2925 строк на seed). Цены новых файлов не читались (только `begin`).

| Гейт | CHECK | EXPECT | EVIDENCE |
|---|---|---|---|
| L3.1 склад | `python3 -m pytest tests/test_fwd_l3.py -k sklad_arhiv` | `raw` (float32), `present`, `days`, `channels` = `store_raw.npz` побитно | passed |
| L3.2 X_w5 | `… -k xw5_arhiv` | `X` (2944, 5, 53, 11) float32, `mask`, `day_ok`, `norm_ok`, `days`, `channels` = `X_w5.npz` побитно | passed |
| L3.3 z | `… -k z_2026_ravno_artefaktu` | z (5 seeds) 2925 строк = `z` артефакта `tcn_w5_2026_s*` bitwise (max разн. 0), дни равны; 2026-01-05…02-27 внутри | passed |
| L3.4 хэш и PCA | `… -k content_hash`, `-k pca_ne_pereschityvaetsya`; mut `-k obuchenie_ne_vyzyvaetsya` | `content_hash` всех 5 артефактов = CHECKPOINT V7 `1b.stageA.hashes`, порча веса → ValueError; `fit_pca_past`, `train_encoder`, `STA.compute`, `train_missing` не вызываются | passed |
| L3.5 e_d, s_d | `… -k e_d_ravno -k s_d_ravno` | e_d = колонка `s` `E1/x1.parquet` (2018 строк, 510 дней e = 1 на всём складе); s_d, n, nlast, zero, rv_d, delta = `x1_ya_feat.features` | passed |
| L3.6 выходы | `… -k vyhody_data_fwd_models`; `python3 -m models_v2.s17.fwd_store --check`; `python3 -m models_v2.s17.fwd_z --check` | файлы = пересчёт; z = артефакт; `content_hash` в z-файлах = CHECKPOINT | passed; rc 0 |
| L3.7 AST (№ 141) | `tests/test_fwd_l3_mut.py -k ast_fwd` | модули `fwd_*` не содержат `train_encoder`/`fit_pca_past`/`train_missing`/`v7_1b_sta.compute`; сканер ловит 6 подложенных текстов | passed |
| L3.8 обрезка | mut `-k obrezka` | порча баров SBER и IMOEX после d0 = 2025-12-30 не меняет склад, X_w5, e_d, s_d, z строк ≤ d0 (побитно); склад при cut = d0 = префикс полного | passed |
| L3.9 замок | mut `-k zamok_novye_ceny` | при сборке архива новые файлы читаются только `columns=["begin"]`, `rv_fwd_SEALED` не открывался; cut в новом периоде → `SealedError` (склад и z) | passed |
| L3.10 мутации | mut `-k myutaciya` | «PCA пересчитан по всей выборке», «порог e_d с днём d», «объём в лотах», «бар d+1 в складе d» (+ порча после d0 меняет строки ≤ d0), «день d в медиане слота», «обрезка по cut снята» ловятся | passed |
| L3.11 гигиена | `python3 -m py_compile models_v2/s17/fwd_{store,z}.py tests/test_fwd_l3*.py`; `wc -l` | rc 0; ≤ 200 строк | rc 0; 102 / 85 / 110 / 191 |

Закрыто 11 из 11 (pytest `tests/test_fwd_l3.py tests/test_fwd_l3_mut.py`: 20 passed, rc 0).

Оговорки Л3:
- Мутация «порог e_d с днём d» на реальном архиве почти не меняет e_d: на всём складе (2014–2026) 2 дня из 2944 (510 → 512 флагов), в днях `x1.parquet` (2018+) — 0. Поэтому равенство с `x1.parquet` её не ловит; ловит тест на синтетике с известным ответом (e = [0]×10 + [1]; мутация → [0]×11).
- Уточнение: «объём в лотах» меняет только канал `lvol` (ln объёма) — `vol_rel` инвариантен к масштабу; ловится по `raw`.
- ↺ e_d и s_d считаются по float32-складу (как `de0_data.load_store` и `x1_ya_feat.load_store`), иначе равенство x1.parquet не побитно. `state_es.parquet`: последний день архива (2026-02-27) имеет `t_plus = NaT`, `delta = NaN` (пара нового периода — в открытии).
- Лот (÷10) — модельная мутация; проверка единиц объёма новых файлов (`CHECKS.md`: «в норме») по ценам до замка не делалась.

## Л2. K2 (XGBoost `x2_best_j`), K3 (LightGBM `C0_inh_F4`) SBER на склейке (11.11.2026)

Код: `models_v2/s17/fwd_k23.py` (замороженные `hp_k23_world.build_world`, `hp_k23_calc.forecast_frame` → `c_calc.block_job`, `job.run_days`, `assemble` — импортом; И2 — из `k23_meta_SBER.json`, не пересчитывается); тесты `tests/test_fwd_l2.py` (реальный архив, 11), `tests/test_fwd_l2_mut.py` (обрезка, IDX вне сессии, мутации, 8). Выход `data/fwd/models/k23.parquet` — только архивные строки (2271, d 2016-12-30…2026-02-26, t⁺ ≤ 2026-02-27); строка d = 2026-02-27 и новые — в открытии (Л6). Цены новых файлов не читались.
Окружение: python 3.9.6, numpy 2.0.2, pandas 2.3.3, **xgboost 2.1.4, lightgbm 4.6.0**; `nthread = 1` (`s4/fit.py:55`), `num_threads = 1`, `deterministic = True` (`s6/lgbm/grid.py:32`); процессов пула 2 (spawn); seed 42.

| Гейт | CHECK | EXPECT | EVIDENCE |
|---|---|---|---|
| L2.1 dz0 и F4 | `python3 -m pytest tests/test_fwd_l2.py -k dz0_i_f4` | мир на склейке при cut = 2026-02-27 = архивный мир (`KW.build_world` из архивных файлов): F, сетка, `dz0` (X, y, off; 28 столбцов = 11 + BASE 11 + IDX 6), все 51 столбцов `feats` (в т. ч. F4), eps — побитно | passed |
| L2.2 обрезанный мир | `… -k obrezannyj_mir`, `-k zaglushka` | мир при cut = d0 = префикс архивного (X, dz0, feats побитно; цель строки d = NaN); заглушка t⁺ = 2026-01-05 из календаря Л1, последняя строка в сетке | passed |
| L2.3 K2, K3 на обрезках | `… -k k2_k3_na_obrezannyh` (4 d) + `python3 …/scratchpad/fwd_l2/run38.py` (все 38) → `docs/otchety/harj_fwd/L2_K23_CUTS38.csv` | σ̂² K2, K3 на данных, обрезанных по каждому d (2025-12-30…2026-02-26) = `k23_SBER.parquet` **побитно**; hp_year 2026, метки HP те же | passed (2 d); CSV: 38 из 38 `bitwise=True`, 21 с на d, 13,5 мин |
| L2.4 выход | `… -k k23_parquet`; `python3 -m models_v2.s17.fwd_k23 --write --check` | `k23.parquet` = архив (σ̂², rv, hp_*) побитно; нет строк с t⁺ > 2026-02-27 | passed; `k23 archive rows=2271 …` rc 0 |
| L2.5 И2 | `… -k i2_iz_metadannyh -k ast_net_vyzova -k mutaciya_i2` | `load_i2` = метаданные (2017…2026); `choose_i2/loyo_scores/first_year/pairs_before` не вызываются (подмена на исключение — путь проходит); AST без `choose_i2/loyo_scores/run_loyo/i2_dump`; И2 2026 с depth 5 → результат ≠ архива | passed |
| L2.6 обрезка 18 инструментов | `tests/test_fwd_l2_mut.py -k corrupt_after -k dostupnost` | бары всех 18 инструментов строго после d0 18:40 (вечер d0 тоже) умножены на случайный множитель — мир дня d0 побитно прежний; максимальный `begin` каждого из 18 ≤ d0 18:40 и в день d0 | passed |
| L2.7 мутации рынка | `… -k mutaciya_bar_d_plus_1_ili_vecher_d` | «бар d+1» (`limit_of` +1 день) — доступность > d 18:40 и мир ≠ эталону; «вечер d» (`limit_of` 23:59 + расширенная основная сессия) — порча вечера меняет признаки | passed |
| L2.8 IDX и календарь, решение 7 | `… -k idx_i_kalendar`; `-k mutaciya_bary_vne_sessii` | к барам IMOEX добавлены бары 07:00–09:40 и 19:00–23:50 каждого дня (+2 дня без сессии 01.05, 03.05.2025; +40 тыс. баров): `index_days`, календарь Л1, `dz0` (в т. ч. 6 IDX), `feats` (F4), eps — побитно прежние; при расширенной «основной сессии» imoex_d/w/m меняются | passed |
| L2.9 правило IMOEX, решение 8 | `… -k index_rule` | `index_rule` подставляет дни сессий IMOEX склейки в `daily._IDAYS` (`mkt.paper` исключает день без сессии замороженным кодом) и восстанавливает его всегда, в т. ч. при исключении; мутант без восстановления отличим | passed |
| L2.10 гигиена | `python3 -m py_compile models_v2/s17/fwd_k23.py tests/test_fwd_l2*.py`; `wc -l` | rc 0; ≤ 200 строк | rc 0; 123 / 127 / 160 |

Закрыто 10 из 10 (pytest `tests/test_fwd_l2.py tests/test_fwd_l2_mut.py`: 17 passed, 267 с, rc 0).

Оговорки Л2:
- ↺ Обвязка архивного кода (сам он не правился): (1) `index_rule` подменяет `daily._IDAYS` на время построения мира — замороженные `mkt.paper`, `ft.build`, `b2feat.build` вызывают `build_daily("auto")` и без этого видели бы IMOEX только из архивного файла (решение 8); (2) `pad_world` — строка-заглушка t⁺ (X, y, rv = NaN) после последней строки обрезанного мира: `rows.year_of` берёт год HP по дате цели, `har.eligible` не включает последнюю строку. Заглушка не входит в обучение (`pair_index` требует конечной цели); побитное совпадение L2.3 это подтверждает.
- t⁺ заглушки берётся из календаря Л1 на `LAST_SESSION` (только `begin`, календарь биржи); влияет только на год HP строки d (для d = последней сессии года — следующий год). Для последнего дня склейки (2026-11-10) t⁺ не определён, заглушки нет: строка d = 2026-11-10 не считается (пары нового периода кончаются t⁺ = 2026-11-10).
- Строка d = 2026-02-27 (t⁺ = 2026-03-02, прогноз по архиву, цель — новая) в `k23.parquet` не записана (как в `k1.parquet`); её считает Л6 вызовом `build_k23(cut, src, dates=…)` после замка.
- Полный набор 38 обрезок считался скриптом сессии (не pytest: 13,5 мин); в pytest — 2 из 38 (первая и последняя; лимит хука 300 с). Сквозной `build_k23` на 180 новых датах — Л5/Л6 (до замка не считается: `SealedError`).
- «Вечер d» в рыночном блоке защищён дважды: `limit_of` (≤ d 18:40) и `main_session` (10:00–18:40); тест мутации «вечер d» ломает оба слоя.
- Первая обвязка пробы (`probe1.py` без `if __name__`) зависла на spawn-пуле и снята по pid; код и данные не затронуты.

## Л4. cfg8, X1, ZB, ZB0 (11.11.2026)

Код: `models_v2/s17/{fwd_cfg8,fwd_x1,fwd_zb}.py` (замороженные `v7_3_cfg.run_cfg`, `v7_1m_wf._refit`, `v7_1m_feat.build_Z`, `de_e1_state.run`, `x1_z_walk.walk` — импортом; новой логики нет: сборка таблицы `make_tab`, заглушка t⁺ `pad_daily`, z 2026 из Л3); тесты `tests/test_fwd_l4_cfg8.py` (обрезки, порча, мутация), `test_fwd_l4_cfg8_full.py` (полный ход ≈ 90 с), `test_fwd_l4_x1.py`, `test_fwd_l4_zb.py`, `test_fwd_l4_ast.py`; выходы `data/fwd/models/{cfg8,x1,zb,zb0}.parquet` — только архивные строки (2018, d 2018-01-03…2026-02-26) и `docs/otchety/harj_fwd/L4_CUTS38.csv` (38 обрезок, скрипт сессии, 3 процесса, 5,4 мин). Цены новых файлов не читались; новых строк нет.

| Гейт | CHECK | EXPECT | EVIDENCE |
|---|---|---|---|
| L4.1 cfg8 архив | `pytest tests/test_fwd_l4_cfg8_full.py -k polnyj_hod`; `python3 -m models_v2.s17.fwd_cfg8 --write --check` | 2018 строк = `V7/3/cfg8.parquet` и `data/fwd/models/cfg8.parquet` побитно; таблица = `load_tab` (2271) | passed; `cfg8.parquet == пересчёт == архив V7: OK`, rc 0 |
| L4.2 X1 архив | `pytest tests/test_fwd_l4_x1.py -k polnyj_hod`; `python3 -m models_v2.s17.fwd_x1 --write --check` | 2018 строк = `E1/x1.parquet` побитно | passed; `x1.parquet == пересчёт == архив E1: OK` |
| L4.3 ZB, ZB0 архив | `pytest tests/test_fwd_l4_zb.py -k polnyj_hod`; `python3 -m models_v2.s17.fwd_zb --write --check` | 2018 строк = `Z/zb.parquet`, `zb0.parquet` побитно | passed; оба `== архив Z: OK` |
| L4.4 обрезки (pytest) | `… -k obrezka_stroka -k obrezka_prognoz` (3 файла) | d = 2025-12-30 и 2026-02-26: строка d и префикс X1, ZB = архив побитно; блок cfg8 = архив (≤ 1e-12 отн.) | passed |
| L4.5 обрезки 38 | `python3 …/scratchpad/fwd_l4/run38.py` × 3 доли → `L4_CUTS38.csv` | 38 из 38 строк d = архив: X1, ZB, ZB0, cfg8 F ≤ 1e-12; префиксы X1, ZB, ZB0 побитно | 38 из 38: `relmax` X1, ZB, ZB0, cfg8 ≤ 1,2e-16; π, g cfg8 ≤ 7,8e-14; побитно cfg8 23/38, X1, ZB, ZB0 37/38; префиксы 38/38 |
| L4.6 состояния | `… -k sostoyanie` (x1, zb); cfg8_full `-k lambda` | π₀, π₁, α₀, α₁ = архив, η = 1/медиана QLIKE закрытых пар (пересчёт); λ, b, a ZB = архив, ZB0: a = 0; λ cfg8 по seeds на блоках 2025-12 и 2026-02 = полный ход; счёт выбранных λ = CHECKPOINT V7 `lam_selected` | passed |
| L4.7 даты переоценок | `… -k daty_pereotsenok`, `-k parametry_menyayutsya`, `-k alpha_menyaetsya` | cfg8: fits d0 = первый день месяца индекса; α X1 и λ, b, a ZB меняются только в первый день месяца; новый период по календарю begin: 03-02, 04-01, 05-04, 06-01, 07-01, 08-03, 09-01, 10-01, 11-02 | passed |
| L4.8 мутации | `… -k mutaciya`, `-k buduschee`, `-k obrezka_bary` | «закрыто + 1» (cfg8 `closed_rows`, X1 `closed_date`, ZB `closed_date`), «α по всей выборке» (X1), «пул ZB по всем парам» ловятся; будущее после d0 (цели, K2, K3, e_d, s_d) не меняет префикс X1, ZB; порча баров SBER и IMOEX после d0, K2, K3 после d0 не меняет блок cfg8 ≤ d0 (побитно) | passed |
| L4.9 изоляция, замок, гигиена | `pytest tests/test_fwd_l4_ast.py`; `-k zamok`; `python3 -m py_compile …`; `wc -l` | AST (№ 141): нет `train_encoder`, `fit_pca_past`, `write`, `emit`, `merge_cp`; сканер ловит 6 текстов; cut > 2026-02-27 → `SealedError`; выходы — только архивные строки; файлы ≤ 200 строк | 13 passed; rc 0; 171 / 69 / 76; тесты 94 / 59 / 112 / 106 / 50 |

Закрыто 9 из 9 (`pytest tests/test_fwd_l4*.py`: 43 passed, 143 с, rc 0; по файлам: cfg8 32 с, cfg8_full 88 с, x1 17 с, zb 12 с, ast 1 с).

Оговорки Л4:
- ↺ Допуск 1e-12 отн. (не побитно) — только для блока cfg8 на обрезках: `FT.ensemble_lnF` считает прогноз пакетом строк блока; на обрезке блок короче полного (d ≤ конец месяца), и torch на CPU может менять последний разряд (G, F ≤ 1,2e-16, π и g ≤ 7,8e-14 — малые значения). Полный ход по всем датам архива и блок с той же длиной (порча после d0, Dec-блок) — побитно. X1, ZB, ZB0 из того же C — ≤ 1,2e-16 (1 ульп на 2026-02-02).
- Обрезки X1, ZB в pytest берут C строк < d из архивного cfg8 (честная проверка узла); сквозная цепочка (cfg8 блок → X1 → ZB) — в `L4_CUTS38.csv` (C строки d — собственный блок).
- K2, K3 строк ≤ d на обрезках — сохранённый `k23.parquet` Л2 с `rv = NaN` у пар t⁺ > d: он побитно равен обрезке по `L2_K23_CUTS38.csv` (Л2.3), пересчёт ≈ 225 с на обрезку не повторялся.
- η на 2026-02-26 проверен независимым пересчётом (в `x1.parquet` его нет); состояние «на 2026-02-27» (после пары (02-26, 02-27)) требует строки d = 02-27 (K2, K3, cfg8) — её считает Л5/Л6, здесь снимок на последней архивной строке 2026-02-26.
- Хук после правки снимает pytest через 60 с (не 300): `test_fwd_l4_cfg8_full.py` (полный ход 5 seeds, ≈ 90 с) отмечается им как «не уложились» — файл самостоятельно проходит (4 passed).
- Новые строки (cut > 2026-02-27) требуют замка: `SealedError` до Л6.
