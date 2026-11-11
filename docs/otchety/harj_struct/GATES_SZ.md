# GATES — спецификация S-Z до счёта (Ж-258), arhitektor, 12.11.2026

Своя ошибка: файл записан после первых проверочных прогонов (01:06), не до них. Прогоны только читали данные; EVIDENCE — из повторного прогона 01:05–01:06 по `date`.
`<sp>` = `/private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/5cfa4f4e-a264-412c-a614-0a8af6830acb/scratchpad/s_z`. Скрипты запускаются из корня проекта командой `python3 -W ignore <sp>/<скрипт>.py`: `-I` здесь не работает, numpy лежит в пользовательском site-packages.

## G1 — S_Z_SPEC.md существует, разделы 1–12
CHECK: grep -c "^## " docs/otchety/harj_struct/S_Z_SPEC.md
EXPECT: 12
EVIDENCE: 12 (01:06)

## G2 — обязательные элементы поручения в спецификации: контроль S-RCKZs, N 855 / 3415, вложение Н2, предсказание
CHECK: grep -c -E "S-RCKZs|N = 852 \+ 3 = 855|3409 \+ 2·3 = 3415|\(Н2\)|## 9\. Предсказание" docs/otchety/harj_struct/S_Z_SPEC.md
EXPECT: ≥ 5
EVIDENCE: 18 (01:06)

## G3 — артефакты Stage A целы; z на всех датах D_M до 2026-02-27 и на всех сдвигах 252
CHECK: python3 -W ignore <sp>/avail.py
EXPECT: content_hash ≠ CHECKPOINT V7: 0 из 45; нет z (годы < 2026): 0; нет z(s−252): 0; в артефакте 2026 нет 179 дат 2026 F
EVIDENCE: «content_hash != CHECKPOINT V7: 0 из 45»; «нет z по всем seeds (годы < 2026): 0»; «нет z(s-252) по всем seeds (все годы): 0»; «2026 | 2925 | 2014-07-01 | 2026-02-27 | 4 | 2198 | 179 | 0 | 1.0»; «дат D_M после 2026-02-27 (2026 F): 179, d 2026-03-02..2026-11-09»; rc 0 (01:05)

## G4 — z на 179 датах 2026 F восстанавливается без обучения (веса и PCA артефакта 2026), совпадает с артефактом и PIT
CHECK: python3 -W ignore <sp>/fwd2026.py
EXPECT: у каждого seed нет z на 0 из 179 дат; max|z − артефакт| = 0 на 2925 архивных днях; PIT max 0 на 3 датах
EVIDENCE: «seed 0 … нет z на датах 2026 F: 0 из 179; конечны: True; архивных дней общих 2925, max|z - артефакт| = 0.000e+00» (так же seeds 1–4); «PIT: d 2026-03-02 / 2026-07-07 / 2026-11-09 — … = 0.000e+00»; «замок открытия существует (Src.is_open, содержимое не читалось): True»; rc 0 (01:05)

## G5 — вложение θ_Z = 0 определено и выполняется побитно на сохранённом S-RCK (без обучения)
CHECK: python3 -W ignore <sp>/nest.py
EXPECT: C == rck.C True; F == rck.F True; X1_R == rck.X1_R True; S-RZ: F == w5g.F True
EVIDENCE: «S-RCKZ при θ_Z = 0: C == rck.C побитно: True (max отн. 0.00e+00)»; «F == rck.F побитно: True; X1_R == rck.X1_R: True»; «S-RZ при θ_Z = 0 (C = cfg8): F == w5g.F побитно: True»; 59 месяцев θ ≠ 0, 1208 дней; rc 0 (01:06)

## G6 — сдвиг контроля по сессиям отличается от сдвига по строкам D_M (основание мутации)
CHECK: python3 -W ignore <sp>/nest.py
EXPECT: в D_M пропущены сессии; совпадение дат < 1946 из 1946
EVIDENCE: «сессий SBER в [2018-01-03; 2026-11-09]: 2220; дат D_M: 2198; пропущенных сессий в D_M: 22»; «… 1694 из 1946» (01:06)

## G7 — основание обрезки z̃ ±3: распределение только входа, без RV
CHECK: python3 -W ignore <sp>/ztilde.py
EXPECT: max |z̃| дней прогноза ≫ 3; доля > 3 ≈ 1 %
EVIDENCE: «|z~| дней прогноза: max 44.12; доля > 3: 0.0091; > 4: 0.0053»; «|z~| внутри пулов: max … 29.82; медиана max 20.57; доля > 3 в пуле: медиана 0.0145, max 0.0239»; 75 активных месяцев с 2020-08-03 (01:06)

## G8 — ни одного прогноза S-Z не посчитано
CHECK: ls data/regen/harj_struct
EXPECT: A C CM D L R (каталога Z нет)
EVIDENCE: A C CM D L R (01:06)

## G9 — N программы до регистрации S-Z
CHECK: grep -n -A 4 '"S.L2.registered"' docs/otchety/harj_x1/CHECKPOINT.json
EXPECT: "n": 852, "n_wide": 3409
EVIDENCE: «"n": 852», «"n_wide": 3409» (01:06); CHECKPOINT.json и ALL_TRIALS.csv не изменялись

## G10 — записаны только файлы владения
CHECK: find docs/otchety/harj_struct docs/otchety/harj_x1 models_v2 data/regen/harj_struct tests -type f -mmin -40
EXPECT: только docs/otchety/harj_struct/S_Z_SPEC.md и docs/otchety/harj_struct/GATES_SZ.md
EVIDENCE: «docs/otchety/harj_struct/S_Z_SPEC.md», «docs/otchety/harj_struct/S_LINE_FINAL.md», «docs/otchety/harj_struct/GATES_SZ.md»; rc 0 (01:06). **С EXPECT не совпало:** `S_LINE_FINAL.md` — итог линии по Ж-258 пп. 1–3, его ведёт дирижёр. Архитектор этот файл не открывал и не писал. EXPECT не учёл параллельного автора. Гейт не закрыт; закрывает его G11.

## G11 — инструменты записи архитектора применялись только к файлам владения (ручной гейт: авторство файла командой не проверяется)
CHECK: РУЧНОЙ — авторство записи файлом не выражается; проверка по журналу вызовов Write/Edit этой сессии
EXPECT: Write/Edit — только S_Z_SPEC.md, GATES_SZ.md и `<sp>/*.py`
EVIDENCE: Write — `<sp>/avail.py`, `fwd2026.py`, `ztilde.py`, `ztilde_rob.py`, `nest.py`, `docs/otchety/harj_struct/S_Z_SPEC.md`, `GATES_SZ.md`; Edit — `<sp>/ztilde.py`, `ztilde_rob.py`, `S_Z_SPEC.md`, `GATES_SZ.md`. Других записей нет (01:07)

## Реестр
закрыто 10 из 11; ABANDON: 0; pending: 0; не совпало с EXPECT: 1 (G10 — чужой файл S_LINE_FINAL.md, см. G11)
