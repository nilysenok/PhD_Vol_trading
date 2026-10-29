# Проверка пересчёта реверса р.1 на A_zashch (коммит `c3d88564`), проверяющий, 29.10.2026; оформил писарь по тексту проверяющего

Вердикт: НЕ СЛИВАТЬ до подтяжки (блокеры — тесты и шапки; числа подтверждены).

Блокеры
1. Из заданных тестов падают 6 (6 failed, 41 passed, 1 skipped; env Z2). 4 в `tests/test_revers_r1_yakor_be.py` (строки 37, 93, 117 и `sverka_a_cd`) читают `docs/tables/revers_r1/b_razlozhenie.csv` и `e_po_fazam_test1.csv` и ждут строки `yakor` и ряд A; таблицы заменены, прежние — в `_do_a_zashch/`; тесты перенаправить на `_do_a_zashch/`. 2 в `tests/test_revers_r1_a_zashch.py` (`test_test1_baza_ravna_stroke_a_cd`, `test_comm0_vyshe_i_div0_menyaet_tolko_tsenu`) берут кэш по общему указателю `cache_latest_test1.txt` (строка 21), а не по ключу; указатель смотрит на кэш без дивидендов (11:47:30) → Sharpe −0,8205 вместо −0,9146; результат зависит от того, кто последним собирал кэш.

Вопросы 1–7
1. ДА. Sharpe A_zashch (плата 0) = −0,9145861306377668 побайтно в `b_razlozhenie.csv`, `e_po_fazam_test1.csv`, `zh_sigma_zatvor_a_zashch_test1.csv`, `a_avtokorr_test1.csv`; по годам 2022/2023/2024 −1,0316521211005965 / −0,642936482139735 / −1,4071524631849406 — как в `a_cd_test1.csv`. Плата 5 % (−1,0480) совпадает между `e_po_fazam`, `v_overnajt_a_zashch`, `a_avtokorr`.
2. ДА. Ночь + день = валовая во всех 7 строках `v_overnajt_a_zashch.csv`; `r1_v_a_zashch.py:79` сверяет с портом до 1e-12. День — open и close дня d+1, позиция решена на close d, заглядывания нет. N = lr − I, сумма верна по построению; проверок определения разложения кроме `test_vnutridnevnaya_izvestnye_otvety` нет.
3. ДА. Затвор использует σ̂ ≤ d (`r1_zh_zashch.py:47-53`); к2 — 20 повторов; p по конечным: 5/18 = 0,2778 при 17 конечных; NaN не подменяются (mSR 2024 пуст; `test_p_znachenie_nan_ne_podmenyaetsya` зелёный).
4. ДА. `div0` оставляет позиции и убирает дивиденд из lr; `fdr_t0` = база (`fdr.py:53-54`: `noch_iz_pos` для t0 = |pos|, база с exposure = |pos|) — следствие модели; `fdr_t1` строго ниже базы.
5. ДА по формулам (`fastbt/metriki_e.py:108-142`, Льюнг–Бокс сверен со statsmodels, `tests/test_avtokorr.py` зелёные). Синтетика при φ = ±0,3. Утверждение дирижёра верно с поправкой: Sharpe нед./дн. ≈ 0,98/√VR(5), множитель (дней/недель)/5 = 744/154/5. Факт / 1/√VR: A_fix 1,287 / 1,313; A_strogaya 1,282 / 1,308; A_zashch 1,122 / 1,143; TEST_C 1,291 / 1,316; D 1,040 / 1,060.
6. НЕТ. Метка «справочно, 2022–2024» есть везде; коммита и хэшей входов нет в `b_razlozhenie.csv`, `_net/b_razlozhenie_net_a_zashch.csv`, `e_po_fazam_test1.csv`, `v_overnajt.csv`, `zh_..._k2_test1.csv`. В `zh_sigma_zatvor_a_zashch_test1.csv` и `a_avtokorr_test1.csv` имя `sigma_hyb_h1_dev.parquet` с двумя разными sha256 без пояснения.
7. ДА. Дат ≥ 2025 в таблицах нет; в коде граница 2025-01-01 (`zona.check`, `read_bars`); `citat_sigma` режет σ̂ по 2024-12-31.

Не удалось: мутации в своей копии (правка заблокирована хуком) — запущены встроенные мутации тестов (сдвиг дня в листе в, лаг ACF, VR без деления на 5, знак VR) — зелёные; мутация листа ж «σ̂ дня d+1» есть в тесте (`test_zatvor_ne_zavisit_ot_budushchego_i_mutaciya_sdviga_lovitsya`), своей не гонял.

Исправления — отдельным PR кодера (тесты на `_do_a_zashch/`, кэш по ключу, шапки); своя ошибка дирижёра — таблицы б и е закоммичены без прогона зависящих тестов (`docs/resheniya/tolkovaniya.md`, 29.10).
