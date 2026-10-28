# Handoff R2, агент L — 26.10, 17:00 по Маку (решение владельца 17:00)

Ветка `r2-B-nogi`, база `kanon-v22-pereschet`. Протокол — `docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md`: v3, в server_r2 есть отсылки к v4. Строка «L». Подробный журнал — `docs/otchety/dev_log_r2_L.md`. Отчёт — `docs/otchety/2026-10-26_r2_L.md`.

## Зона L (решение владельца 16:25 и решения дирижёра)

- `scripts/analysis/r2_nogi.py` — ноги кластера 2. Модуль драйвера L.
- `scripts/analysis/r2_calib.py` — калибровка: кластер 3, кластер 12 (scale_har и вероятностный слой), calib.update. Модуль драйвера L3.
- Не L:
  - кластер 14 → P (`r2_panel.py`), упрощённая панель L удалена;
  - кластер 15 → R (`r2_refit.py`);
  - кластер 11 (τ, struct) → M. Код τ и CJ в `r2_nogi` остался как справка для M.
- Всё — версия d0 с эмбарго h, шаги 1–6, годы ≤ 2021. Ядро — `r2_core.py` (S), драйвер — `r2_driver.py` (S).

## Сделано (все PR слиты)

- **#175** — ноги кластера 2 и калибровка (кластеры 3, 12) на `r2_core`.
  - API: `fit_predict` (для E), `leg_ln` / lnall (для B), `params.csv` в формате D.
  - Контроли: ноги = ряды Э `*_emb_d0` (≤ 1,4e-16); калибровка по умолчанию = Б0.1 sm R1 (≤ 4e-16).
  - Боевой счёт ног: 71 дизайн по `models/designs/r2/r2_L_designs_v1.yaml`, 202 ряда, ошибок 0. Кэш — `data/regen/R2/cache/<hash>/`, сводка — `data/regen/R2/L/schet_L_nogi.csv`.
- **#183** — панель только у P: `leg.panel ≠ off` → ValueError. Добавлен `r2_calib.fn_v0_panel` — калибровка ряда P.
- **#189** — контракт драйвера: `run --designs --h --seeds --steps --workers` и `leak` у L и L3; schema_v2; `leg.regressors` списком — сетка.
  - Дым через `r2_driver.Runner`: 155 эконометрических дизайнов (цепочки L и L→L3), итоговый ряд у всех, детекторы OK.

## Открытые вопросы (❓ — умолчание действует)

1. Проверка сдвига (п. 2.6) блокирует RVI_anchor h=5/22 и MIDAS h=22.
   - Профиль корреляций плоский (разница ≤ 0,008), RVI выровнен по дню d, утечки нет.
   - RVI-anchor+s² по протоколу — кандидат V0.
   - Умолчание: блок по протоколу. Решение — дирижёр или владелец.
2. ln RV IMOEX у F (features_r2_v2) определён иначе, чем в R1 (`harx.t_rvm_imoex`): медиана |Δ| 0,057. Умолчание: в S2 — определение R1.
3. Определения, которых нет в схеме: MIDAS θ-сетка, HAR_perlead, factor, `sm_pooled` (= все прошлые шаги), форма s² по состоянию (`s̄²·clip(f/f̄, 0,25; 4)`), c_qlike аддитивный (как Д26 c), prob scale_har. Записаны в `models/designs/r2/r2_L_nogi.yaml` с ❓.
4. В `r2_L_designs_v1` есть HARJ_qmle и HARX_S4_qmle. estimator — ключ R; в драйвере L их считает с default, а qmle — R.
5. `neprimenimye_faktory: []` у L и L3: L входит во все цепочки, объявлять неприменимое нельзя — фактор выключится у всех.

## Фоновые процессы

Нет. На 16:58 процессов `r2_nogi` и `r2_calib` нет. Последние логи — в scratchpad сессии; итоговые файлы:
- `data/regen/R2/L/dym/*.csv` — дым ног, калибровки, fit_predict, params, сверка F;
- `data/regen/R2/L/dym_driver/dym_driver_L.csv` — дым через драйвер;
- `data/regen/R2/L/schet_L_nogi.csv` — боевой счёт ног.

## Что дальше (для свежего агента)

- Ждать запуска драйвера: он сам вызовет `r2_nogi.py run` и `r2_calib.py run` по дизайнам `models/designs/r2/driver_v1/`.
- При сбоях смотреть `R2/S/driver/run/state_h{h}.json`.
- Проверочные команды:
  - `python3 scripts/analysis/r2_nogi.py kontrol` — 9/9;
  - `r2_calib.py kontrol` — 3/3;
  - `r2_nogi.py dym` / `r2_calib.py dym` / `r2_nogi.py dym_driver`.

## Запреты (протокол, раздел 10, и правила сессии)

- Ничего за годы ≥ 2022 не читать и не писать: `r2_core.load_d0` ≤ 2021, `write_series` / `check_no_future`. `test_locked/` не читать.
- Не читать:
  - таблицы и отчёты теста: `docs/tables/*_test*`, `docs/tables/dlya_teksta/`, `2026-10-26_test_glava3/4.md`, `2026-10-26_proverka_laga_harj.md`, `zhurnal_pervogo_progona_gl3.md`, `final_test_run.log`;
  - постановку владельца `2026-10-26_r2_klastery_vladelec.md`;
  - план `2026-10-25_pereschet.md`.
  - Раскрытие: Э при поиске определений видел цифры теста в выдержке #143 (журнал Э).
- Не менять `final3.py`, `gibrid_v17.py`, `gibrid_designs.py`, `final_test.py`, `harx.py`, `features_pit_v3`. Общие модули соседей (`r2_core`, `r2_driver`, `r2_panel`…) только импортировать.
- `r2_nogi.load` / `trmask` / `_perturb` импортирует ВЫР-1 — их не менять.
- Без `sed -i`, heredoc, `python3 -c` для записи файлов; без `rm -rf`; `.env` не читать; сети нет (кроме `git push` и `gh pr`).
- Счёт на Маке: до 10 воркеров, num_threads = 1. PR — в `kanon-v22-pereschet`, в `main` не сливать.
