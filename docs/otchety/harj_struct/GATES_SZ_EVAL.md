# GATES — S-Z-EVAL (оценка и отчёт S-Z), кодер, 12.11.2026 (записано до первой правки кода, 01:4x по `date`; EVIDENCE дописан 01:5x)

Выбор до счёта: объявленная диагностика «нулевое распределение процедуры» — S-RCKZ, синтетика `z_walk_syn` (не правится), 10 шумов, seeds 0–9; считается отдельной командой `z_eval_null --run` до оценки, в условие Z-DEC не входит.
Порядок записи: `S_Z_NULL.csv` (01:45:22) → ключ `S.Z.eval` → таблицы с p (01:47) → `S_Z_REPORT.md`.

## ZE.G1 — оценка
CHECK: `python3 -W ignore -m models_v2.s18.z_eval --check`
EXPECT: QLIKE = независимой формуле на 2198 датах без обрезки; Σ вкладов лет = Δ (10⁻¹²); NW-8; B = 10 000; Холм N = 855 / 3415, m = 6; H-Z1, H-Z2, условия разд. 7.3 (два набора, 0,415 / 0,50 / 1,00 / 0; K = 22) и Z-DEC выведены из таблиц; ключ `S.Z.eval` раньше таблиц; sha256 выходов до и после равны
EVIDENCE: `N Холма 855 / 3415, m = 6; H-Z1 True, H-Z2 True; Z-DEC 3: S-RCKZ не прошёл: У5-S|к S-RCK (+0.0118). S-RCK остаётся кандидатом…`; `S-Z eval --check: ok`, rc 0 (QLIKE сверен с `v4_ntp_state.qlike`; бутстрап и Холм — независимый пересчёт; sha до/после внутри `--check`). Подложенные ошибки (мои, в памяти, не тесты): N = 6, Z-DEC = «1» без подтверждения, знак Δ, H-Z2 переставлен, обрезка дат в основной Δ — все пойманы `problems`.

## ZE.G2 — отчёт
CHECK: `python3 -W ignore -m models_v2.s18.z_doc --check`
EXPECT: числа `S_Z_REPORT.md` = таблицам (пересборка побайтно); первая строка — «исследовательский результат на раскрытой истории, не OOS; механизм выбран после просмотра истории»; первая таблица — K1 / K2 / K3 / B_mix / X1 / W5-G / S-RCK / S-RL1 / S-RCKZ / S-RZ / S-RCKZs с QLIKE, датами, n и Δ; в каждой таблице границы дат и n; формулировки разд. 7.5
EVIDENCE: `S-Z doc --check: ok`, rc 0 (пересборка = файл; порядок рядов; «подтверждено» / «лучше» только в формулировках 7.5 и тексте Z-DEC; «эквивалент» отсутствует).

## ZE.G3 — нулевая диагностика
CHECK: `python3 -W ignore -m models_v2.s18.z_eval_null --check`
EXPECT: `S_Z_NULL.csv`: 10 шумов (seeds 0–9); сводка = пересчёту из строк; в отчёте рядом с реальными Δ(S-RCKZ к S-RCK) и Δ(S-RCKZ к S-RCKZs); правило 5–95 %
EVIDENCE: `z_eval_null --check: ok`, rc 0. seeds 0–2 совпали с Z2.G7 (−0,0967 / +0,3180 / −0,1053 %). Нуль: среднее +0,0219 %, СКО 0,2780, 5 % −0,3029, 95 % +0,4331; доля θ ≠ 0 — среднее 0,587. Разд. 3 отчёта.

## ZE.G4 — тесты (пишет тестировщик, не автор)
CHECK: `python3 -m pytest tests/test_s18_z_eval*.py -q`
EXPECT: все прошли
EVIDENCE: ABANDON: ZE.G4 вне владения кодера (задание: «Тесты test_s18_z_eval* не писать: их пишет тестировщик»); не выполнен, не закрыт.

## Регрессия
CHECK: `python3 -W ignore -m models_v2.s18.z_walk --check`; `python3 -m pytest tests/test_s18_z_*.py -q`; `shasum -a 256 data/regen/harj_struct/Z/*`
EXPECT: ok; выходы `Z/*.parquet` не изменены (sha до и после)
EVIDENCE: `z_walk --check: ok`, sha до/после равны (4 файла: e8284586…, 4cd9345d…, 35ab7b3a…, 915533ee… — rckz, rckzs, rz, zmaps); `pytest tests/test_s18_z_*.py`: 103 passed, 1 failed — `test_s18_z_reg.py::test_checkpoint_without_s_keys_is_refused_before_any_write` XPASS(strict): тест ожидал дефект z_reg, который теперь исправлен (мой код `z_reg` не трогает; 21 passed / 1 xfailed в моём прогоне 01:44 — файл изменён другим листом).
