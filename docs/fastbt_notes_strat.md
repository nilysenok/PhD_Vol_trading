# Заметки к fastbt: что известно о коде стратегий гл. 4 (ветка strat-zamer, 25.10.2026)

Пути от `moex_volatility/scripts/` основной копии (только чтение). Лог замеров: `data/strat_zamer/logs/` (не в git). Песочница `data/strat_zamer/sbx/` (копии скриптов и входов, в `.gitignore`).

## 0. Что делал и в каком состоянии
- Песочница и обвязка: `scripts/analysis/zamer_setup.sh`, `zamer_keyrate.py` (восстановление `cbr_key_rate.parquet`, исходного файла `/Users/nilysenok/Desktop/PhD/...` на Маке нет), `zamer_unit_a.py` (единица stage3), `zamer_unit_v4.py` (единица v4 + ворота по `daily_positions_pre2026.parquet`), `zamer_downstream.sh` + `zamer_gate_msp.py` (stage5 + бутстрап из сохранённых позиций), `zamer_fast.py` (numba-версия `calc_sharpe_comm`, ТОЛЬКО для stage3, проверена до 0.0 на 6 стратегиях SBER 2023), `zamer_fast_a.py` (joblib-раннер stage3, недописан, не использовать).
- ВАЖНО: пайплайн stage3 (`stage3_walkforward.py`) НЕ является источником печатей A/C/D (см. сообщение аудита): печати из `strategies_walkforward_v4.py`. Все замеры stage3 ниже годятся только как грубый порядок величины; движок надо строить на v4.
- Остановлено: все фоновые процессы `zamer_*` сняты (`pkill`). Полный прогон не запускался.
- Точные сокращения 1–4: сделан только п. 4 (ответ ниже) и частично п. 3 (кэш сигналов уже есть в коде); пп. 1–2 не делались.

## 1. Какой код реально делает печати (v4)
- Точка входа `strategies_walkforward_v4.py:main` (1583), по тикерам `process_ticker_v4` (1017), `multiprocessing.Pool` `N_WORKERS=min(cpu_count,8)` на ТИКЕРЫ (1635). Внутри тикера: цикл tf (daily, hourly) -> стратегия S1..S6 -> A, B, C, D подряд.
- Импортирует `strategies_walkforward.py` (движки бэктеста, сигналы, сетки, C-маска, D-шкала), `s5_rerun.py`, `s5s6_rerun.py` (S5/S6; для daily-пути `dispatch_signals` берёт `gen_s5_signals`/`gen_s6_signals` из strategies_walkforward — проверить, какие версии реально используются, если fastbt покрывает S5/S6).
- Комиссия в v4: `COMM_DAILY=0.0040`, `COMM_HOURLY=0.0035` (0,40% и 0,35% на сторону, v4:117-118), а не 0,05%. Комиссия вычитается в `calc_sharpe_v4` (359), `calc_metrics_v4` (375), `compute_max_dd` (416): `net = pos*lr - |diff(pos, prepend=0)|*commission`. Ставки на кэш (key rate / FDR) в v4 НЕТ: она появляется позже в `stage5_msp.py` (`build_portfolio_returns`: fdr = max(0,1-exposure)*rf*cal_days; margin_cost при exposure>1; excess = total - rf*cal_days; рф из `cbr_key_rate.parquet`, ffill/bfill; комиссия 0,0005 inst там вычитается из `daily_gross_return` позиций).
- Позиция `pos[t]` — состояние на баре t по close[t]; доходность следующего бара `log_ret[t]=log(close[t+1]/close[t])`, `daily_gross_return = pos*log_ret`.

## 2. Маска (утечка) — где в коде
- `_year_bounds(dates, Y)` (`strategies_walkforward.py:2013-2018`) возвращает `(индекс 1 янв Y, индекс 1 янв Y+1)`.
- A (`approach_a_v4`): `_, train_end = _year_bounds(dates, test_year)` (v4:494) -> окно обучения до 1 янв Y+1, т.е. включает год Y; `test_end` = `_year_bounds(dates, test_year+1)[1]` (502-503) (бэктест гонится до конца года Y+1 и берётся срез года Y далее при записи).
- B, C, D: `val_start = _year_bounds(dates, 2020)[1]` (= 1 янв 2021, а не 2020!), `val_end = _year_bounds(dates, test_year)[1]` (= 1 янв Y+1). То есть валидация = 2021…Y включительно (содержит тестовый год).  Строки: B v4:660-662, C ~840-848, D ~907-915. Выбор горизонта σ (B/C/D) на той же маске val_mask.
- Исправление маски [0]: заменить на `[0]` — граница 1 янв Y. Для B/C/D: val_start=_year_bounds(dates, 2020)[0] нужно решить отдельно (владелец: период разработки 2017–2021; валидация следующего года).

## 3. Устройство A (v4:471-642)
1. `cached_sigs[sp]` и `cached_strength[sp]` — сигналы и «сила» считаются один раз на набор сигнальных параметров в рамках года (внутри цикла по test_year -> пересчитываются 4–7 раз; их можно вынести за цикл по годам: от года не зависят).
2. Фаза 1 (грубая сетка): RM-сетка прореживается: >200 -> `[::3]`, >20 -> `[::2]`, иначе вся. Для каждой пары (сигнал, RM) `dispatch_bt_atr_v3` (строки 1930-1966) на `end_idx=train_end`, затем `calc_sharpe_v4`, `compute_max_dd`, `compute_exposure` (последние три — по маске обучения; каждый вызов делает `positions[mask]`, `np.diff`, `np.std` -> узкое место в Python).
3. Фильтр `sh>0 and max_dd>-0.30 and exposure>5`; если пусто — топ-10 по Шарпу; затем top50.
4. Фаза 2 (уплотнение): соседи ±1, ±2 индекса RM-сетки (`get_rm_neighbors`, 444) для top50 (одна и та же сигнальная конфигурация), dedup по (sp, rm), top-20 (переменная названа top30, срез `[:20]`).
5. Фаза 3: для top-20 x 24 комбинации execution layer (`EXEC_GRID_DAILY`: consec_entry [1,2], min_hold [0,5], cooldown [0,10], min_strength [0,0.3,0.5]) — `execution_layer` (211-278, чистый Python-цикл по барам для consec/min_hold/cooldown), Шарп по обучению; top-10.
6. Ансамбль: top-10 прогоняются до `test_end` (log_trades=1), `execution_layer`, голосование: `mag = mean(|pos|)`, `dir = sign(sum(pos))`, `ensemble = (mag>0.5)*dir` (v4:~620-625). Бинарные позиции.
- Ответ на «двухэтапный перебор»: КОД делает грубая сетка -> уплотнение (v4:518-576), как «coarse-to-fine»: грубо — прореженная RM-сетка по ВСЕМ сигналам, уплотнение — соседи RM только для top50. Полной сетки нет. В тексте диссертации раздел 2.3.3 найти не удалось (файла текста у меня нет): сверять надо владельцу. stage3_walkforward.py (не источник печатей) гонит ПОЛНУЮ сетку RM для top-30 сигналов.

## 4. Движки бэктеста (numba @njit, `strategies_walkforward.py`)
- `bt_contrarian_v3` (888-1010): цикл `for t in range(warmup, n)`. Вход (983-1005): при `cp==0 and cooldown_rem==0`, `sig_arr[t]==±1`: `ep=close[t]`, ATR `atr14[t]` (nan->0), `sl=close∓sl_mult*a`, `tp=close±tp_mult*a`, held=0. Выход (910-981): брейк-ивен (917-923: если (close-ep)>be_trigger*ea то sl=ep), стоп (low<=sl / high>=sl; SL проверяется ПЕРЕД TP на том же баре), тейк (high>=tp), partial_exit (не используется в daily-сетке), затем сигнальный выход `exit_arr[t]`, затем временной стоп `held>=max_hold` (held++ в начале шага). Перезаход на том же баре после выхода разрешён (cp==0 проверяется после блока выхода) если cooldown не задан. Кулдаун после SL: `cooldown_bars`. Цена исполнения: стоп/тейк НЕ по уровню, а позиция просто становится 0 на этом баре; `pos[t]=cp` записывается ПОСЛЕ обработки выхода/входа (бар выхода — позиция 0; бар входа — позиция ±1 с доходностью log_ret[t]).
- `bt_trend_v3` (1013-1146, трейл fixed_atr/chandelier/parabolic), `bt_range_v3` (1149-1268, ADX-фильтр, time_decay), `bt_range_split_v3` (1271-1398), и их `*_vpred_v3` версии для B (1401-1715): стоп/тейк в единицах σ_pred (`sigma`), `gamma_hold` (адаптивный max_hold от σ/медианы), `k_be`.
- Диспетчеры: `dispatch_bt_atr_v3` (1930), `dispatch_bt_vpred_v3` (1969): B использует `base_hold`=20 (daily)/40 (hourly) для Contrarian и 30/60 для Trend; cooldown_bars по умолчанию 5.

## 5. B (v4:645-759)
Нет отдельного поиска сигналов: берёт `a_params` (top-10 сигнальных конфигураций A за год) и exec-параметры лучшей (первой) комбинации A. Перебор `B_GRIDS_V3[cat]` (Contrarian: 5·5·5·3·3 = 1125; Trend 6·7·5·3·3 = 1890; Range 1125) — для каждой точки 10 прогонов `dispatch_bt_vpred_v3` + `execution_layer` + голосование + `calc_sharpe_v4` на val_mask. Комиссия та же 0,40% (v4). Лучшая по Шарпу на val идёт на тест.

## 6. C (v4:762-895)
`_c_grid_search_v4`: перебор C_THRESHOLDS(6) / C_RANGE_BANDS(7) x C_LOOKBACKS(3) x C_HORIZONS(3) x C_DIRECTION(3) x term_filter(2; пропуск при h22) x hysteresis(2). Маска `_build_c_mask_v3` (strategies_walkforward.py:2222-2280): `pctrank = pd.Series(sigma).rolling(lookback, min_periods=max(50, lookback//5)).rank(pct=True)` (дорого: pandas rolling rank — 0,98 с из 1,56 с стадии C на единицу), Trend: pctrank>thr; Contrarian: pctrank<1−thr; Range: lo<=pctrank<=hi; hysteresis через `_apply_hysteresis` (numba); direction rise/fall по `sigma[t]-sigma[t-5]`; term_filter по отношению σ_h22/σ_h1. Позиция A обнуляется вне маски. Отбор по Шарпу на val_mask при ограничениях `_quick_exposure_trades`: exposure ≥ 5% и сделок/год ≥ 10; если пусто — запас 3% / 6; если снова пусто — позиции A как есть. Ранг pctrank можно считать один раз на (σ, lookback) — не на каждую точку сетки.

## 7. D (v4:898-983)
Бинарный затвор: `scale = _compute_d_scale_v3(sigma, dp)` (strategies_walkforward.py:2431): ewm-сглаживание, floor/cap, `(target_vol/σ)^γ` или обратная волатильность c rolling-средним; `min(scale, max_lev)`; `gate = scale >= gate_th`, `gated = a_pos*gate`, Шарп на val. Сетка: 5·3·3·2·2·3·3 = 1620 x 4 порога + обратная 3·3·3·3 = 81 x 4. Размер позиции после затвора НЕ масштабируется (только 0/1), значит w не входит в перебор (инвариантность к размеру позиции в D тривиальна).

## 8. Профиль (cProfile, v4, S1 SBER, daily, только тестовый год 2022, 1 поток, Мак под нагрузкой load≈13–17)
Время единицы (S1, SBER, один год): A=17,3 с, B=0,7, C=1,6, D=0,8 (всего 20,8 с вкл. cProfile-накладные; без профилировщика оценка ×0,6). Стадия A доминирует (≈83%).
Топ-10 по tottime в `approach_a_v4` (функции опознаны по числу вызовов/строкам; имена в логе усечены):
1. `bt_contrarian_v3` (numba-диспатч), 960 вызовов, 3,38 с (0,0035 с/вызов) — единственная чисто-численная часть;
2. `np.diff` 157 360 вызовов, 1,34 с (2,49 с cum) — внутри `calc_sharpe_v4`/`compute_max_dd`;
3. `compute_max_dd`-подобная функция, 77 960 вызовов, 1,22 с (4,09 cum);
4. `calc_sharpe_v4`, 78 440 вызовов, 1,15 с (5,43 cum);
5. `np.var/_var` 78 440 вызовов, 1,03 с (std ddof=1);
6. `ufunc.reduce` 311 938 вызовов, 0,84 с;
7. `execution_layer`-подобная функция, 77 990 вызовов, 0,77 с (0,81 cum) [число вызовов совпадает с `compute_exposure`/`execution_layer` — уточнить по строке];
8. `cumsum` 77 960, 0,71 с;
9. `np.broadcast_to` 157 360, 0,68 с;
10. функция с 77 990 вызовами, 0,59 с (1,90 cum).
Вывод: ~80% времени A — Python-обвязка метрик вокруг 960 численных бэктестов (≈5,4 с cum на Шарп + 4,1 на max_dd), а не сам бар-цикл; 960 бэктестов на единицу дают ≈3,4 с, то есть ≈3,5 мс на бэктест 1500–3000 баров. Переписать метрики (Шарп, max_dd, exposure) одним numba-проходом по маске даст ×5–10 по A без изменения логики; цикл `execution_layer` (чистый Python, 24 комбинации x 20 x годы) — следующий. C: 63% времени — pandas `rolling.rank`. D: мелкие numpy-вызовы.
- stage3 (не источник печатей) на тех же единицах: A 1–13 с на (стратегия, SBER, 2023), B 0,03, C 0,6, D 0,1; с numba-Шарпом A: 1,0–7,7 с (×1,3–2,5), побитовое совпадение позиций A–D.

## 9. Воспроизведение сохранённых МСП (из stage5 по позициям, 8 с)
`zamer_downstream.sh`: stage5 по `wf_positions_{A..D}.parquet` в песочнице даёт Шарп МСП A/B/C/D = 1,3500 / 1,4654 / 2,4229 / 2,5488 против сохранённых 1,3153 / 1,4654 / 2,5099 / 2,5357. B совпадает до 0.0 (ряд пересчитан `fix_msp_b.py`, комиссия вычтена); A, C, D расходятся (макс. |Δ дневной excess| 1e-4…8e-4 на ~90% дней): сохранённые ряды A/C/D построены без вычета комиссии (согласуется с сообщением аудита), `stage5_msp.py` вычитает её. Это не ошибка движка, а различие версий stage5; воротами для fastbt служат позиции, не МСП A/C/D из сохранённых CSV.
- `cbr_key_rate.parquet` восстановлен (2016–2021 из `/Users/nilysenok/Desktop/Старое/disser/data/key_rate_cbr_2016_2025.csv`, 2022–2025 обратным счётом из `MSP_A_inst.csv`: rf=(total−excess)/cal_days*365); ряд 2022–2025 в CSV диссертации расходится с использованным (2025: 0,21 против 0,20/0,17), значит исходный файл ставки другой — взять новый у владельца.

## 10. Ворота по позициям v4 (маска [1])
Ворота `zamer_unit_v4.py` реализованы; на единице S1 SBER 2022 позиции и в новой версии, и в `daily_positions_pre2026.parquet` нулевые (234 из 234 совпали, ничего не доказывает). Нужны активные единицы (например S3/S4 SBER, 2023) — не прогонялось. `daily_positions_pre2026.parquet` — 4 905 677 строк, колонки date, strategy (имя «S1_MeanRev»), tf, ticker, approach, test_year, position, daily_gross_return.

## 11. Входы/файлы параметров, читаемые пайплайном (для аудита)
v4: `data/ohlcv_daily_full.parquet`, `ohlcv_hourly_full.parquet`, `vpred_aligned.parquet` (md5 0284dee1… = `_BACKUP`; `vpred_aligned_harj.parquet` другой). Сетки зашиты в коде v4/strategies_walkforward.py. stage3: дополнительно `walkforward_v4/narrowed_grid_v2.json` (сетка A, в `main` загружается, но `walkforward_A` её не использует — полная сетка из `SIGNAL_GRIDS`), `test_phases/FINAL_best_configs.json` (читается, но используется только в отключённом `walkforward_A_DISABLED`). stage3c: `walkforward/wf_positions_A.parquet`, ohlcv, vpred. stage5: `wf_positions_{A..D}.parquet`, ставка. Файлов с параметрами, подобранными на тесте, вне `FINAL_best_configs.json`/`narrowed_grid_v2.json` не обнаружено, выводов не делаю.

## 12. Сложность (для оценки fastbt)
Daily A: сигналов x RM: S1 ≈ 4 000·? (см. лог stage3: S4 36x396, S5 432x640, S6 126x640); v4 фаза 1 берёт прореженную RM-сетку. Основной объём вычислений — число бэктестов (960 на единицу S1 на год) x цикл по барам ≈ 1500–3000. Целевой fastbt: один njit-проход на (сигнал, RM) с метриками внутри (Шарп/maxDD/exposure по маске без выделения массивов), `prange` по комбинациям, execution_layer в numba, rolling rank в numba (rank по окну 126/252/504) с кэшем на (σ, lookback).

## 13. Объём контекста
Мой контекст уже велик (оценка > 150 тыс. токенов): движок fastbt лучше отдать свежему агенту с этими заметками.
