# Ж-211 SBER h=1 (hybrid_final) — kodir, 06.11.2026. Статус: ВЫПОЛНЕНО, все стадии прошли, EVIDENCE в GATES_HF.md заполнен (12/12), не закоммичено; ниже — старая передача (история). Не коммитить. Процессов <= 6; python -c/heredoc/& запрещены (скрипт-файлы, run_in_background, `python3 -W ignore -m models_v2.s7.run_hf --stage X`).

## Сделано (до «коммит есть» разрешено: реестр и диагностика I без обучения)
- `docs/otchety/hybrid_final_v1/GATES_HF.md` — HF1–HF12 (сетки c, λ; правила выбора; тест взаимодействия; уровни 1/2; N), гейты G0–G11. EVIDENCE: все pending (G0 и G3 выполнены стадиями, но EVIDENCE заполнить после коммита).
- Реестр: `ALL_TRIALS.csv`, `CHECKPOINT.json` (стадия reg): Ж-204 106 + Ж-206/210 182 + пилот 189 (уникальные прогнозы INVENTORY p1/p2/p3/pb/fix) => N_prior = 477; N_final = 477 + новые (<= 537). Новых trials пока 0.
- Диагностика I (стадия diag, exit 0): `HAR_RESIDUAL_DIAGNOSTICS.csv` (2271 стр.), `HAR_ERROR_STRUCTURE.csv`, `YEARLY_STRUCTURE.csv`, `HAR_RIDGE_BY_YEAR.csv`.
- Код `models_v2/s7/`: hf_ledger (register с лимитами), hf_data, hf_diag, hf_rules (apply/cfg_grid/wf_select/ema_weights/rate_limit_weights), hf_stage_reg, run_hf (диспетчер; модули остальных стадий ещё НЕ написаны: hf_bound, hf_inter, hf_wstage, hf_feat, hf_regstage, hf_final). Тесты `tests/test_v2_hf_rules.py` (13 passed; λ=0 побитно HAR, clip с обеими границами, мутации «закрыто+100», «клип без знака», «EMA на будущем весе»).

## Находки диагностики I (потолок, история просмотрена)
- Почему 2017–18: по годам QLIKE(HAR)/QLIKE(XJ): 2017 0,98; 2018 0,89 (HAR ЛУЧШЕ XJ); 2019 1,06; 2020 1,18; 2022 1,39. Ex post вес HAR по году: 0,55 / 1,00 / 0,25 / 0,00 / 0,05 / 0,00. Корреляция ln-ошибок HAR/XJ стабильна 0,92–0,98 — структура совместных ошибок не менялась; менялась относительная точность (XJ учится с историей, в 2017–18 мало пар). Выигрыш A1_roll_126 над XJ: 2017 −0,0046, 2018 −0,0281, далее ≈ 0 (±0,002); простое среднее HAR/XJ в 2018 хуже HAR (−0,011): выигрыш 2018 — перенос веса на HAR (roll), а не комплементарность.
- Остаток HAR: автокорреляция e лаг 1/5/22 = 0,005/−0,043/−0,006; ridge LOYO ΔQLIKE: const +0,0030 (только смещение вредит), F3 −0,0060, F5 −0,0035, F4 −0,0014, F1 +0,0048, F2 +0,0018, F6 +0,0082, F1–F6 −0,0031 (повтор Ж-210).
- Недооценка: после доходности <= −2 % mean e +0,23 (t 5,3), ΔQLIKE HAR−XJ +0,051; резкий рост RV (>0,7 ln к RV_w) — e +0,94, ML лишь чуть лучше (1,08 vs 1,33 QLIKE); 2022: доля 5 дней в |ΔQ| 72 %.

## Следующие шаги (после «коммит есть»; порядок GATES_HF)
1. `bound` (hf_bound.py): читать `data/regen/xgb_lgbm_full_v1/e/b_B1q_s42.parquet`, `b_B2q_s42.parquet` (d, sigma2, g); har из comp; R2: 4 c × 2 модели (register("bounded", cfg)), `R2_wf` на cfg_grid(lams=(1.0,)) [c ∈ сетка ∪ ∞ ∪ 0]; R3: λ ∈ {0,25;0,5;0,75} × 2 (register("residual_xgb"/"residual_lgbm")), `R3_wf` на cfg_grid(cs=(), lams=LAM_GRID), `R23_wf` на cfg_grid(). Проверка: apply(c=None, λ=1) ≈ stored sigma2 (1e-12), λ=0 побитно HAR. Выход RESIDUAL_XGBOOST.csv / RESIDUAL_LIGHTGBM.csv; ряды — `data/regen/hybrid_final_v1/series_bound.parquet`. m = hf_rules.closed_of(rv, d, td).
2. `inter`: V1–V6 (HF8), pit.past_bucket, OLS+NW Вальд, Холм по 12; при отсутствии подтверждения — `ABANDON HF-G8` в GATES (первой строкой отчёта), 0 trials режимов.
3. `weights`: цели rules.weights('roll_126'/'roll_252'); ema_weights h ∈ {21,63}, rate_limit r ∈ {0,005;0,02}; H7 тройка; 10 trials register("dyn_weights"); плато соседей.
4. `feat`: bfit.run на наборах S345/S34/S35/S3 (feats_b2, группы accepted.json; `W2={**W,"dz":design.Dz(X,dz.y,dz.off)}`, hpy_of из st_b.hpy_of, kind xq/lq, procs 4; X = fg-матрица по ВСЕМ строкам F: `feats[cols].to_numpy()`), абляция лучшего (<= 3), R23_wf на лучшем; register("residual_xgb"/"residual_lgbm"); ~30–50 с на ряд.
5. `final`: H1–H7, DM/бутстрап A/B/C (evalrows.vs, stats_d.boot_delta+block_idx, stats_d.holm_n с N=hf_ledger.n_final()), годы/периоды HF10, уровни HF11, таблица п. 17, главный тест обрезки (G2) по всем новым рядам, MCS необязателен, документы G9. Мутации темы дописать в тесты: Холм N, пороги режима по всей выборке, лимит trials; стадии bound/weights/feat/inter/final покрыть тестами на синтетике.
6. EVIDENCE во всех гейтах, затем отчёт <= 500 слов (таблица п. 17 SBER h=1, L1/L2 по гибридам, ответ «почему 2017–18», trials, команды, реестр гейтов).

## Нарушения/замечания
- Один пустой heredoc (`python3 - <<EOF` без кода) выполнен по ошибке — результата нет; больше не использовался.
- Хук post-edit гоняет tests/test_v2_hf_rules.py после каждой правки (ожидаемо).
- Отступления: V5 = f6_mkt_chg (относительный рыночный уровень; уровень IMOEX RV не сохранён); пилот Ж-200/202 — по INVENTORY (189), не по реестру trials; наборы признаков выбраны по EX1 (потолок). В `mutations.md` строки для Ж-211 не добавлял (файл вне владения) — внести писарю: λ=0≠HAR, клип без нижней границы, закрыто+k в выборе (c, λ), EMA/лимит на будущем весе.
