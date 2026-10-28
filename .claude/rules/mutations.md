# Обязательные мутации по темам проекта (канон v23, §5 п. 1)

Правило владельца (`proverka.md` п. 1): каждый PR, меняющий пайплайн, содержит тест, который падает на подложенной ошибке и проходит на правке. Ниже — минимальный перечень по темам; PR, затрагивающий тему, обязан подложить соответствующую мутацию (если тема его касается). Новая найденная ошибка — новая строка в этом файле в тот же день (`docs/kanon/Kanon_universalnye_dopolneniya.md`, §8).

Источники строк: план `docs/zadaniya/2026-10-25_pereschet.md`, постановки v3 (§6) и E (§8: `docs/zadaniya/2026-10-27_semya_E.md`), тела PR и отчёты проверяющих (`docs/otchety/`). PR — номера из `kanon-v22-pereschet`; «поймана» означает: тест с подложенной ошибкой падает на мутации. Тест-образец — `файл::тест` от корня репозитория; его можно копировать как шаблон.

| № | Мутация (что подложить) | Тема | PR, где поймана | Тест-образец |
|---|---|---|---|---|
| 1 | σ̂ сдвинута на +1 день (используется прогноз, известный завтра) | сдвиг входа | #271 (шаг 4); раньше #234/#235 (S0), аудит #237 | `tests/test_s9_shag4.py::test_sdvig_sigma_plus_odin_lovitsya`; `tests/audit_fastbt/test_01_sigma_align.py::test_dev_align_never_uses_future_rows` |
| 2 | Касание уровня (low = L) засчитано как исполнение лимита | касание уровня | #290 (в работе) | `tests/test_exec_e.py` после слияния #290; ближайший по теме до слияния — `tests/audit_fastbt/test_14_stop_take_bar.py::test_take_only_no_cooldown` |
| 3 | Тейк и стоп в одном баре — порядок перевёрнут | порядок событий в баре | #252 (аудит FA2a) | `tests/audit_fastbt/test_14_stop_take_bar.py::test_both_levels_in_bar_long_stop_wins` |
| 4 | Гэп через стоп/тейк исполнен по уровню, а не по открытию | гэп | #252 (аудит FA2a) | `tests/audit_fastbt/test_15_gap.py::test_gap_down_through_stop_recovers_inside_bar` |
| 5 | Комиссия выхода не взята на последнем баре позиции (граница окна, Н5) | комиссия на границе | #252 (xfail) → #254 (исправление) | `tests/audit_fastbt/test_04_commission.py::test_open_position_at_window_end_pays_entry_and_exit` |
| 6 | Окна отбора (B, C, D, быстрый слой, обучение A) не платят выход на конце окна | комиссия на границе (окна отбора) | #259 (+ #261: тесты на три непокрытых места) | `fastbt/tests/test_ov_window_exit.py::test_sharpe_window_pays_exit` |
| 7 | Ключ кэша не меняется при подмене σ / баров / корпсобытий | ключ кэша (FA2b) | #249 (поймана как xfail) → #250 (исправление) | `tests/audit_fastbt/test_23_cache_keys.py::test_s9_key_tracks_sigma` |
| 8 | Ранг C считается по окну с текущим днём (Н1) | текущий день в ранге | #238 / #246 (аудит FA) | `tests/audit_fastbt/test_02_c_rank.py::test_main_c_mask_rank_excludes_current_day` |
| 9 | к2 со сдвигом 0 ≠ исходный ряд | контроль к2 | #265 | `tests/test_s9_kontrol.py::test_lag0_gives_original_mask_and_engine_sharpe` |
| 10 | к3 оставляет S5/S6 (должен обнулять ровно их) | контроль к3 | #260 (код к3), #265 (файл теста) | `tests/test_s9_kontrol.py::test_k3_mutation_s6_left_caught`; `tests/test_s9_shag0.py::test_mutation_k3_with_s5_left` |
| 11 | Справочный выбор при 0,05 % берётся из данных 0,04 % | комиссия, выбор при другой ставке | #269 (шаг 3′, коммит 897352e) | `tests/test_s9_shag_db_run.py::test_run_ref_choice_differs_from_main_and_mutation_caught` |
| 12 | ddof=0 вместо ddof=1 в основной колонке Шарпа | одно определение метрики | #279 | `tests/test_s9_shag_dsharpe.py::test_mutation_ddof0_in_main_column_caught` |
| 13 | Δ Шарпа взята из бутстрапа ЛВ, а не как разность табличных Шарпов | одно определение метрики | #279 | `tests/test_s9_shag_dsharpe.py::test_mutation_lw_delta_caught` |
| 14 | N в DSR подменён на 1 (DSR обязан вырасти) | N в поправке | #287 (метрики E, слит 27.10) | `tests/test_metriki_e.py::test_mutaciya_p12_N_v_edinicu_dsr_dolzhen_vyrasti` |
| 15 | Цель (RV на горизонте h) с окном, заходящим за 31.12.2021 (утечка RV 2022) | цель за границей периода | шаг 4: #271 → #283, до слияния (коммит d62825e) | `tests/test_s9_shag4_boy.py::test_cut_target_mutatsii_lovyatsya` |
| 16 | Ставка на 2018–2021 вытянута назад константой (`bfill`) | константа в данных | находка #236; исправление и тест #267 | `tests/test_stavka.py::test_restored_rate_backfilled_constant_is_caught` |
| 17 | Чтение 5-минутных баров за пределами периода | чтение за границей периода | #288 | `tests/test_bars5m.py::test_mutatsiya_chteniya_za_predelami_lovitsya` |
| 18 | E-const (σ̂ константой) считается с текущим днём | текущий день в σ̂ | #289 | `tests/test_sigma_e.py::test_mutaciya_const_s_tekushchim_dnem_lovitsya` |
| 19 | Топ-10 по обороту текущего года вместо прошлого | текущий год в отборе | #289 | `tests/test_likvidnost_e.py::test_mutaciya_top10_tekushchego_goda_lovitsya` |
| 20 | Знак позиции умножен повторно после слоя исполнения (шорт → лонг) | знак позиции | #326 → #328 | `tests/test_sweep_post_znak.py::test_mutaciya_dvoynoy_znak_lovitsya` |
| 21 | Квантиль силы `q_sily` считается с текущим днём | текущий день в отборе | #308 / #310 | `tests/test_sweep_post_mut.py::test_mutp_01_q_sily_s_tekushchim_dnem_ili_budushchim` |
| 22 | RV для уровней √RV берётся с текущим днём | текущий день в σ | #319 | `tests/test_rm_sqrt_rv.py::test_mutatsiya_rv_s_tekushchim_dnem_lovitsya` |
| 23 | Уровень √RV строится по σ дня d+1 | сдвиг входа | #319 | `tests/test_rm_sqrt_rv.py::test_mutatsiya_sigma_d_plus_1_lovitsya` |
| 24 | Ledger пишет пробу на любом периоде или при любой комиссии | N в поправке | #308 / #310 | `tests/test_sweep_post_ledger.py::test_mut_ledger_pishet_na_lyubom_periode` |
| 25 | Пересчёт при другой комиссии перезаписывает SR пробы | N в поправке | #304 / #306 | тест: см. файл `tests/test_sweep_mutacii.py` ❓ (точного теста на перезапись SR нет) |
| 26 | Повторное открытие validation без удвоения N | N в поправке | #304 / #310 | `tests/test_sweep_mutacii.py::test_mut_24_povtornoe_otkrytie_ne_udvaivaet_N` |
| 27 | Плато заменено пиком | отбор | #310 | `tests/test_sweep_modul_mut.py::test_mutm_07_plato_zamenyeno_pikom` |
| 28 | Шорты A выброшены в E (E0 при q = 0 ≠ A) | E0 ≡ A | #311 | `tests/test_sem_e.py::test_e0_q0_equals_full_a_with_shorts` |
| 29 | B без условия 3 (к2) | отбор B | #314 | тест: см. файл `tests/test_konveyer_bcd.py` ❓ (точного теста нет; ближайший `test_k2_lag_zero_caught`) |
| 30 | Замок открытия теста ставится после чтения данных | открытие | #313 / #320 | `tests/test_otkrytie_gl4.py::test_zamok_do_chteniya` |
| 31 | D начисляется в день записи вместо дня гэпа (дивиденды) | дивиденды, день начисления | #370 | `tests/test_dividendy_pnl.py::test_mutatsiya_den_zapisi_vmesto_dnya_gepa_lovitsya` |
| 32 | Стоп через отсечку не сдвинут на D | дивиденды, стоп | #370 | `tests/test_dividendy_ispolnenie.py::test_mutatsiya_stop_ne_sdvinut_lovitsya` |
| 33 | Сигналы на реальном ряду вместо скорректированного (подключение в загрузчиках) | дивиденды, загрузчики | #370 | `tests/test_dividendy.py::test_mutatsiya_M3b_load_ticker_bez_adjust_ohlc_lovitsya`; `tests/test_dividendy.py::test_mutatsiya_M3c_load_ticker_nf_bez_adjust_ohlc_lovitsya` |
| 34 | Дата отсечки +1 / −1 день против дня ценового гэпа | дивиденды, дата отсечки | #400 | `tests/test_dividendy_gap.py::test_mutatsii_lovyatsya` |
| 35 | Событие, объявленное после d, отдаётся на d (point-in-time) | point-in-time | #395 | `tests/test_sobytiya_pit.py::test_obyavleno_pozzhe_ne_otdayotsya` |
| 36 | Свечи ≥ 2022 записаны после ответа ISS (без обрезки) | граница 2022 при выгрузке | #391 | `tests/test_vygruzka_vselennaya.py::test_mutation_no_clip_caught` |
| 37 | Контроль (б) заворачивает события за концом train в окно train | граница train | #380 | `tests/test_zh051_iter_t.py::test_mutaciya_bez_otbrasyvaniya_hvosta_lovitsya` |
| 38 | Комиссия не в ключе ledger / не в `digest` | ключ кэша | #411 | `tests/test_klyuchi_kesha.py::test_ledger_commission_mutation_caught`; `tests/test_klyuchi_kesha.py::test_digest_commission_mutation_caught` |
| 39 | Каталог numba-кэша без хэша кода (устаревший скомпилированный код) | ключ кэша | #411 | `tests/test_klyuchi_kesha.py::test_numba_cache_dir_mutation_caught` |
| 40 | Режим ФДР (`fdr`) не в ключе кэша / ledger | ключ кэша, режим | #398 | `tests/test_fdr_vnutri_dnya.py::test_mutatsiya_fdr_ne_v_klyuche_lovitsya` |
| 41 | Ворота «сделок ≥ 30» по `prinyato` вместо реальных сделок | ворота отбора | #380 | `tests/test_zh051_iter_t.py::test_mutaciya_vorota_po_prinyato_lovitsya` |
| 42 | Правило выбора внутри CSCV подменено на argmax (PBO не той процедурой, Ж-080) | PBO, правило выбора | в работе, ветка `pbo-pravilo-kod` | `tests/test_pbo_pravilo.py` после слияния |
| 43 | Подмена функции движка (`E.run_e1`) на уровне модуля диагностического скрипта: остаётся после импорта и ломает остальные тесты процесса | изоляция подмен | #509 (найдена 28.10 перепроверкой дирижёра: 70 failed после `test_revers_r1_b`) | `tests/test_revers_r1_b_izolyaciya.py::test_posle_importa_run_e1_ishodnyj` |

Слиты #288, #289; их мутации внесены в строки 17–19. Остальные мутации постановки E §8 (исполнение по барам дня d вместо d+1; E0 при q = 0 ≠ A побайтно; сила сигнала по d+1; комиссия при стопе внутри дня; ставка на капитал, освободившийся внутри дня) обязательны для PR семьи E с первого дня и вносятся в таблицу строками, когда появятся тесты.

Правила пользования:
- проверяющий подкладывает мутацию в отдельной копии, не в основном дереве (`testy.md` п. 2);
- тест без мутации, которая его роняет, считается не доказывающим (`polnota.md` п. 10);
- строка таблицы без PR/теста-образца — пробел: закрывается в ближайшем PR по теме.
