Стоит поверх #679 (V6-12b), тот — поверх #672 (V6-12a). База PR — kanon-v22-pereschet.

## Что сделано
Срез листа V6-12c: подключены пять осей исполнения `lestnica`, `gap`, `t_vyhod`, `vyhod_lim`, `f_neisp` (fastbt/eval_v6_ispoln.py), они открыты в `canon_v6.GOTOVY` по осям (токены `V6-12:<ось>`, механизм #679). Свёртки canon_v6: `piv_buf` без стопа `piv:next|half` — канон (замечание Н4 #677; тейк `piv:R` буфер не задаёт); `time_decay = da` без `tau` или при `tejk ≠ net` — выброс (None).
Лесенка: две ноги по w/2 (лимит на k_e и на k_e + lestnica), как `EvaluatorV2._ladder`, ряды складывает `sum_rows`; k_e и вес восстанавливаются и при сбое.
Несовместимое (движок отказал бы ValueError) даёт NeSchitana: `vyhod_lim` с `t_vyhod ≠ close`; gap/vyhod_lim/lestnica на стоп-заявке.

## Файлы
fastbt/eval_v6_ispoln.py (новый), fastbt/eval_v6.py (+подключение), fastbt/canon_v6.py (V12 + две свёртки); тесты: tests/test_eval_v6c.py, tests/test_eval_v6c_real.py (новые), правка двух тестов под свёртки — tests/test_kub_v6.py (piv_buf), tests/test_v6_nez_kuby.py (time_decay, порог числа точек); GATES_v6-12c.md.

## Итог проверки
Синтетика: `pytest tests/test_eval_v6c.py tests/test_eval_v6b.py tests/test_kub_v6.py tests/test_v6_nez_kuby.py tests/test_klyuchi_kesha_v6.py` — 147 passed (после слияния свежей базы). py_compile — ок. Реальные данные (train4, ≤ 2021, 3 тикера, 1 процесс, свежий NUMBA_CACHE_DIR, `SDF_TYAZHELYE=1 pytest tests/test_eval_v6c_real.py`): 30 passed, 8 skipped («оси нет в кубе Sk»), 120 с — «выкл.» = v5 побайтно на 20 точках каждого Sk (6), каждая ось исполнения меняет результат (24), мутации «лесенка без /2» (ФДР t0d отказывает) и «ось не доходит до оценщика» ловятся. Условия: vyhod_lim считается только при t_vyhod = close (якорь куба — open2, #Ж-137 п. 6, несовместимо), у тренда t_vyhod виден только при tau. Rust-команды проекта не применимы.
уровень проверки: код
окно данных: train4 (≤ 2021) · честная половина: нет (проверка реализации, не цифра)

## Что не сделано и почему (ABANDON, первой строкой отчёта)
Таймбокс 50 тыс. токенов превышен ещё при чтении кода — лист разрезан:
- G5 пивот S5 / VWAP S6 и `pol_puti`: значения `stop_nach`/`tejk` — расширения осей v5, а токен GOTOVY — на всю ось, поэтому `piv:*`/`vwap:*` нельзя открыть без `pol_puti` (оба в одних осях). Нужны: хук `_tm` в `EvaluatorV5._rows` (там `m_tikera(..., True)` зашит), `PivUrovni` из сырья, `k_s = net` при пивотном стопе, NeSchitana на `risk_sigma`/трейл/безубыток при `piv_sl`.
- G6 оси тренда на стоп-заявке (bezubytok, stop_v_plyus, pir, T, reentry, smena — в кубах S3/S4 это новые оси раздела `ispolnenie`, их надо вернуть в точку v5 перед `razbor`; значения `pct:/atr:/step:` у `stop_nach`/`tejk`/`trail`, лист V6-11).
- G7 `tf` — по-прежнему ValueError «ещё не подключена».

## Вопросы
Свёртка `time_decay` выбросом (по заданию) отбрасывает ~70 % точек куба вместо ~40 % (было: только несовместимые по canon_v5): `tejk = net` — одно значение из многих. Схлопывание `time_decay → net` сохранило бы точки. Решение дирижёра/владельца класса B.

## Канон
- гарантии (ИНВАРИАНТЫ с именем теста): лесенка — две ноги по w/2 (`test_lestnica_dve_nogi_po_polovine`); k_e восстанавливается при сбое (`test_lestnica_vosstanavlivaet_k_pri_sboe`); Соболь не даёт «не подключена» (`test_sobol_variruet_novye_osi_bez_ne_podklyucheno`); свёртки (`test_time_decay_vybros_bez_tau_ili_pri_tejke`, `test_piv_buf_svertka_h4`)
- цифры `≡`: —
- `↺` (только отступления от справки или задания): ↺ — time_decay: выброс при tejk ≠ net/без tau по заданию меняет долю выброшенных точек куба ~40 → ~70 % · откат: схлопывание в «net» (29.10)
- задачи: закрыть №: оси исполнения lestnica/gap/t_vyhod/vyhod_lim/f_neisp; остаток V6-12c: пивот/pol_puti, оси тренда, tf

🤖 Generated with [Claude Code](https://claude.com/claude-code)
