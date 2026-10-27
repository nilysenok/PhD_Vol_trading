# Журнал R2, агент A (сборка теста, второе открытие, FINAL_DESIGN_round2)

Протокол v3 — `docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md`. Ветка `r2-A-sborka` от `kanon-v22-pereschet` (f527d69, перебазирована на b6edba1 после PR #161). Время — по `date` на Маке.

| № | Дата-время | Что делали |
|---|---|---|
| 1 | 2026-10-26 15:55 | старт: прочитаны протокол v3, заготовки О (`r2_sborka_test.py`, `final_test_r2.py`), `r2_loyo.py` (factors, gate_E), отчёты ВЫР-1/ВЫР-2, журналы `dev_log_r2_*`, поля 0, 6, 6а FINAL_DESIGN |
| 2 | 2026-10-26 16:00 | `r2_sborka_test.py` под v3 + `--selftest` (синтетика) — OK; grep раздела 8 — 4 совпадения, все с `# R2-guard` |
| 3 | 2026-10-26 16:05 | `final_test_r2.py` в версии d0 + `--selftest` — OK; `r2_A_sborka.yaml` (шаблон спецификации и SHA замороженных) |
| 4 | 2026-10-26 16:10 | шаблон `FINAL_DESIGN_round2.md`; после него `final_test_r2 --selftest` снова OK (шаблон блокирует открытие до замка) |

## Что прочитано (вне test_locked) и откуда SHA замороженных

- SHA тестовых файлов ВЫР-1 — `data/regen/vyr/vyr1/lock_{etalon,f1,f5,f22}.csv`: журналы записи (file, rows, steps, sha256), чисел нет.
- SHA HAR-CJ d0 — `docs/otchety/2026-10-26_vyr2.md`.
- SHA кадра первого теста — `scripts/analysis/test_dop.py` (`SHA`).
- Формат файлов ВЫР — по коду `vyr_etalony_finaly.py` и `vyr_spravochnye.py`. Сами файлы не открывались.
- `test_locked/` не открывался и не листался.
- Не читались: таблицы и отчёты теста, `final_test_run.log`, план `2026-10-25_pereschet.md`, постановка владельца, §§ 12–14 FINAL_DESIGN.
- Запускались только `--selftest` обоих скриптов. Синтетика — во временном каталоге.

## Решения и толкования

- ↺ «Параметры после 2021 заморожены на шаге 6 либо по правилу t−1». К тесту шага k (год 2016 + k) применяется оценка по val шагов ≤ k, как в `r2_loyo.factors`. При k ≥ 7 это val ≥ 2022, поэтому берётся оценка шага 6. Обе формулировки дают одно и то же. В спецификации допустимы только ключи шагов 1–6, ключ > 6 — assert. Константы (гиперпараметры, выбранные на 2017–2021) — скаляром. Откат: правка `par` (26.10).
- Ворота E — формат `r2_loyo.gate_E` (PR #161): `pobeditel.pobeditel`, `V0`. Если победил V0, требуется правило `single` с одной компонентой. ❓ A-1 (для O): design_sha в gate_E нет — A пишет design_sha и design_yaml с SHA в спецификации сам.
- `final_test_r2`: все h считаются всегда.
  - Без победителя гибрид = F_h(d0). На h=22 это эталон (б) d0, и `svodka` test_dop2 пишет «финал = эталон»; DM на h=22 переопределяется, только если победитель R2 есть.
  - Старая строка гибрида первого прогона в кадр не входит.
- ↺ Сверка с первым тестом сужена до строк, не менявшихся в d0: V0 XGBoost / LightGBM (raw и с поправкой), «Лучшая одиночная бустинга», n и дни. Эталоны и гибрид в d0 другие по построению. Откат: `SVERKA` PR #147 (26.10).
- До замка проверяются:
  - FD без заглушек;
  - SHA 12 замороженных файлов по `r2_A_sborka.yaml`;
  - файлы R2: нет rv_actual, число строк = кадр.

  Отказ на этих проверках не тратит открытие. После замка — СТОП, если цель эталонов ВЫР-1 не совпала с целью кадра.
- ❓ A-2 (для дирижёра): HAR-X с поправкой в раскладке (`3_raskladka`, `final_test.CHAIN`) есть только в кадре R1, d0-ряда на тесте нет. Умолчание — ряд R1 с пометкой в FD.

## Эмбарго s² и s² последовательно при открытии (26.10, после PR #165)

Ъ Дирижёр 26.10: «(1) Калибровка теперь с эмбарго (`calib.s2_embargo = on`) … для победителя R2 и для эталона (б) d0 — симметрично. (2) Решение владельца „s² последовательно при открытии“: на шагах 7–10 s² у эталона, F_h(d0) и победителя берётся по правилу t−1, с эмбарго. Механически это делает только твой сборщик: он берёт OOS-остатки шагов 7–9 из `test_locked`, без метрик, с меткой `# R2-guard-s2`. … эталон (б) для второго открытия нужно пересобрать с эмбарго в s². Прежний `test_locked/vyr/etalon_d0_*` … только справочно.»

Ъ Дирижёр 26.10 (решение владельца 17:35): «эталон В2 берётся в версии с эмбарго в s² (PR #195), флаг входит в хэш дизайна. В FINAL_DESIGN_round2 рядом со строкой об исправлении лага дня d добавь строку об этом исправлении … PR #194 и #195.»

- PR #195 прочитан через `gh api`: `r2_calib.py`, `r2_L_calib.yaml` и отчёт L. В отчёте только QLIKE val 2017–2021. Правило эмбарго взято оттуда: строка пула допустима, если конец окна цели < 1 янв. оцениваемого года (`emb_mask`; `_tendy` — h-й следующий торговый день тикера).
- `r2_sborka_test.py`, режим `s2_pravilo: posledovatelno` (по умолчанию):
  - Тест шага k: s² по val шагов max(1, k−2)..k (sm), годы < 2016 + k, эмбарго. Шаги 1–6 — val-файлы вне test_locked. Шаги 7–10 — `part = val` файлов ВЫР-1 и компонент.
  - Цель тестовой части не загружается. Строки с целью val помечены `# R2-guard-s2` (8 строк).
  - Контроль: без эмбарго код обязан повторить `F_sm` ВЫР-1 (правило R1, `gibrid_bias.s2_est`), иначе СТОП.
  - Пишет `R2/final/etalon_d0_s2emb_h{h}_test.parquet` и `final_d0_s2emb_h{h}_test.parquet`. Сырой F_h(d0) val 1–6 = кэш `F_d0_h{h}_s42_val` / Fv (`vyr1/factors_f{h}.csv`); h=22 = эталон.
  - Режим `zamorozka` (раздел 8: шаги 7–10 = шаг 6) — справочный: цель из test_locked не читается вовсе.
  - selftest — OK:
    - без эмбарго = R1;
    - с эмбарго = независимая сверка (последние h торговых дней тикера до 1 янв.);
    - победитель — сырой итог × F;
    - подложенный `F_sm` — СТОП.
- `final_test_r2.py`:
  - (а), (б) d0 и F_h(d0) берутся из сборки (`R2/final/*_s2emb_*`); сырой (а) сверяется с ВЫР-1.
  - Прежние ВЫР-1 (s² как R1) — справочные строки в таблице периодов. Цель кадра сверяется по ВЫР-1.
  - SHA собранных — `r2_A_sborka.yaml: sobrano`; без них СТОП до замка. selftest — OK.
- ❓ A-3 (для дирижёра): в сообщении «шагов 7–9». По правилу t−1 для теста шага 10 (2026) нужны остатки 2023–2025, то есть шагов 8–10 (как в R1, FINAL_DESIGN поле 2). Умолчание — правило: читаются val шагов 7–10.
- Поддерживается s² вида F = exp(κ·s²): s2_est ∈ sm / smrob / sm_roll1y / sm_roll2y / sm_pooled, κ — константа. Если у победителя κ = fit, s2_state, c_qlike или MZ — СТОП и вопрос: такие режимы потребовали бы признаков или подгонки по 2022+.

## Код (SHA-256, первые 16)

PR #165: r2_sborka_test.py bf32fd46bd6be040 · final_test_r2.py f753e2da3b8a9f73 · r2_A_sborka.yaml 4f3d356a84a10d5d.
Эмбарго s²: r2_sborka_test.py 96bd76cdecc059d0 · final_test_r2.py a2557299f452606e · r2_A_sborka.yaml 5f21844f0e99bdf2.
Вариант 1: r2_sborka_test.py 62f9e401e02e0fdd · final_test_r2.py 8c0753ad697a0409.

## 26.10, вечер — E8 импортом G, FINAL_DESIGN_round2 (вариант 1)

- **E8 в сборке — функциями G** (PR #224). `--proverka-G` на val 2017–2021: h=5 E1/E2 — 240/240, h=22 E1/E1b — 90/90, побайтно (Δ = 0). Читались только ряды val кэша и features ≤ 2021 через `r2_core.load_d0`, без test_locked.
- **Решение владельца — вариант 1:** F_h(d0) на всех h. FINAL_DESIGN_round2 заполнен по воротам драйвера:
  - B–D — 21:05;
  - E на исправленной основе — 21:17;
  - E8 драйвера — 21:20.

  Таблицы G и M вошли в приложение R2. По ним E1 + E8 на h=5 не проходит 2σ на 10 seed, на h=22 — мощность.
- **Сборщик.** h=1 без gate_E принимается, только если gate_D без лучшего дизайна и его SHA есть в спецификации (selftest). Спецификация: `F_h_d0: [1, 5, 22]`, SHA ворот, `status: ждёт «да» владельца`. Сборка не запускалась, test_locked не читался.
- **Своя ошибка:** один `git fetch` до разрешения дирижёра; одна команда с heredoc без записи файла (отказана средой, переписана).

## Сборка теста R2 — старт 2026-10-26 21:35:49

Спецификация models/designs/r2/r2_A_sborka.yaml SHA-256 f6046b6264fc7699d1f3a1038c9e11ea0d76a69363b2cb5d6cd148918a034cea; код r2_sborka_test.py SHA-256 62f9e401e02e0fdd25ac278085eec895ccc5bc11bd85407d29261c4e4e4f4fa7.
Ворота E: h=5 17a4c64cb6bc60ba906e97bc85a7189aa16f3d5b1cd437a37f5061f163c39ff7; h=22 d95552ddd5c0a7a0943e19f1c95236a00dec51d107eab3d9348e515f93792f53.
Сборка на h = []; F_h(d0) на h = [1, 5, 22]; s²: posledovatelno, эталон {'est': 'sm', 'kappa': 0.5, 'emb': True}.
test_locked до этой строки не читался.

Прочитано (тестовая часть — без цели; цель — только val шагов 7–10 для s² и val 1–6 вне test_locked):

| файл | SHA-256 | строк |
|---|---|---|
| data/regen/test_locked/vyr/etalon_d0_h1_test.parquet | 3657f85805fd2be5906892d63d56f709425ee19a1442de9c9cd76d4643556de6 | 39951 |
| data/regen/vyr/etalon_d0_h1_val.parquet [val шагов 1–6] | f10175116fcd9df1bb513a1b54db24a0a164dc1b069693c36a8651b1fdb8af33 | 25755 |
| data/regen/test_locked/vyr/etalon_d0_h1_test.parquet [part = val, шаги 7–10: цель для s²] | 3657f85805fd2be5906892d63d56f709425ee19a1442de9c9cd76d4643556de6 | 18021 |
| data/regen/test_locked/vyr/final_d0_h1_test.parquet | e4c59d2f5a3fbf0a0a1b4269209e736fc93fcbf9d60d5e01f359e8edc088f432 | 39951 |
| data/regen/R2/cache/VYR/F_d0_h1_s42_val.parquet [val шагов 1–6] | e0f316fee705b1a55000f84db40bd2a74eed5174ca022132d5888b2ffe8164e2 | 25755 |
| data/regen/test_locked/vyr/final_d0_h1_test.parquet [part = val, шаги 7–10: цель для s²] | e4c59d2f5a3fbf0a0a1b4269209e736fc93fcbf9d60d5e01f359e8edc088f432 | 18021 |
| data/regen/test_locked/vyr/etalon_d0_h5_test.parquet | d28f791f7cc5d92fe77e07d53133cce97082d6cd1290562ea2c6c7b0f9f892e5 | 39951 |
| data/regen/vyr/etalon_d0_h5_val.parquet [val шагов 1–6] | 9c437706bede8e03f1d91d2a2afcf4020d583f1aff1a1ad7df78da4370d54e5c | 25755 |
| data/regen/test_locked/vyr/etalon_d0_h5_test.parquet [part = val, шаги 7–10: цель для s²] | d28f791f7cc5d92fe77e07d53133cce97082d6cd1290562ea2c6c7b0f9f892e5 | 18021 |
| data/regen/test_locked/vyr/final_d0_h5_test.parquet | ef81c061d85f2ab122b3e9f690d54d46938dde4479f6499aa4218138189077f0 | 39951 |
| data/regen/R2/cache/VYR/F_d0_h5_s42_val.parquet [val шагов 1–6] | 7d0107d998b7e01c045ac983e4b1285b3699fa76f959d988221473bd69e8644b | 25755 |
| data/regen/test_locked/vyr/final_d0_h5_test.parquet [part = val, шаги 7–10: цель для s²] | ef81c061d85f2ab122b3e9f690d54d46938dde4479f6499aa4218138189077f0 | 18021 |
| data/regen/test_locked/vyr/etalon_d0_h22_test.parquet | f29cacef3f99d06362ca267412ed6af27c34aa8b2ad26fac929419e7ddd4ff19 | 39951 |
| data/regen/vyr/etalon_d0_h22_val.parquet [val шагов 1–6] | f3bd53492e97c6075aa5b9b464ed4c77cd7dc0b6f3544e288d546523ed320f76 | 25755 |
| data/regen/test_locked/vyr/etalon_d0_h22_test.parquet [part = val, шаги 7–10: цель для s²] | f29cacef3f99d06362ca267412ed6af27c34aa8b2ad26fac929419e7ddd4ff19 | 18021 |
| data/regen/test_locked/vyr/final_d0_h22_test.parquet | a5c870caf15ed3112cc07651910d5bc1afe5bb5db64bc3236e33551b60783390 | 39951 |

Записано (цели нет):

| файл | SHA-256 | строк |
|---|---|---|
| data/regen/test_locked/R2/final/etalon_d0_s2emb_h1_test.parquet | 1dce250bc913e93a7643094261eda5695288e10f72eeac3d0ae1d7d97f8665fc | 39951 |
| data/regen/test_locked/R2/final/final_d0_s2emb_h1_test.parquet | 5ebcf61a1ccfd0f6c660e03431e306f8fe7719f011a66b469d8431be147c92e4 | 39951 |
| data/regen/test_locked/R2/final/etalon_d0_s2emb_h5_test.parquet | de559a9ed66c892842b049964003509ab338bfbac87768dd5d63bf406041b51d | 39951 |
| data/regen/test_locked/R2/final/final_d0_s2emb_h5_test.parquet | 4c11df7660cb174e9d169abdc67d8fd16286c21aa052351fa36b49e9b63eb887 | 39951 |
| data/regen/test_locked/R2/final/etalon_d0_s2emb_h22_test.parquet | 3341184413c8923a71c0b78bd3e72020ecbaa93cd60a67b7a161a494d302c0fd | 39951 |
| data/regen/test_locked/R2/final/final_d0_s2emb_h22_test.parquet | 059a8e664f490ea3cf9ff4ce25b0db9958c2b074b30c21a7e415e0f945b54c2f | 39951 |

Конец 2026-10-26 21:35:49.
