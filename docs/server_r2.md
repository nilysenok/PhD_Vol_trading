# Сервер второго раунда гибрида (R2)

**Ъ Владелец 26.10, 14:30:** «Арендован сервер. Весь второй раунд гибрида (уровни 1–3, 10 seed; протокол `docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md`) считается только на сервере. Мак — только сборка кода и журналы. Результаты первого раунда не пересчитывать. Закрытая папка `data/regen/test_locked/` на сервер НЕ копируется.»

Файлы:

| файл | что делает |
|---|---|
| `scripts/server.sh` (zsh) | `push`, `setup`, `run`, `status`, `pull`, `stop` |
| `env/requirements_r2.txt` | точные версии пакетов (FINAL_DESIGN §6 и всё, что импортируют `r2_*.py`) |
| `env/setup_server.sh` | окружение на сервере: Python 3.9.6, пакеты, ссылка на путь Мака, Rust 1.93.0, `moex_iss` |
| `env/activate_r2.sh` | активация перед счётом; один поток BLAS, TZ=Europe/Moscow |
| `env/check_env.py` | сверка версий |
| `scripts/server_repro_check.py` | воспроизводимость: D12_h5, seed 42, val шагов 1–6 против рядов Мака, порог 1e-6 |

## 1. Что делает владелец (один раз)

1. **Файл доступа** `~/.config/phd_server.env`. Он лежит вне репозитория и в git не попадает. Права — `chmod 600`. Скрипт читает из него только пять ключей, файл не исполняется:
   ```
   PHD_SERVER_HOST=<адрес>
   PHD_SERVER_USER=<пользователь>
   PHD_SERVER_KEY=~/.ssh/<ключ>
   PHD_SERVER_PORT=22
   PHD_REMOTE_DIR=PhD_Vol_trading      # путь от $HOME на сервере; латиница
   ```
   Переменные окружения с теми же именами важнее файла.
2. **GNU rsync на Маке:** `brew install rsync`. `/usr/bin/rsync` в macOS — это openrsync, скрипт его отвергает. На сервере тоже нужен rsync: `sudo apt-get install -y rsync`.
3. **Системные пакеты на сервере** (Debian/Ubuntu): `sudo apt-get install -y curl tar bzip2 build-essential cmake pkg-config git rsync`.
4. **Ссылка на путь Мака.** В коде первого раунда и R2 зашиты абсолютные пути `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/...`: в `l2_lib.py`, `harx.py`, `gibrid_io.py` и `r2_*.py`. Протокол R2 запрещает менять этот код, поэтому `setup` создаёт на сервере ссылку `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading -> ~/PhD_Vol_trading`. Если sudo спрашивает пароль, это делается руками один раз:
   `sudo mkdir -p '/Users/nilysenok/Desktop/Старое' && sudo ln -s "$HOME/PhD_Vol_trading" '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading'`.
5. **Снятие запрета для дирижёра** — по желанию. Сейчас в `.claude/settings.json` в `deny` стоят `Bash(ssh:*)`, `Bash(scp:*)`, `Bash(rsync:*)`, `Bash(scripts/server.sh run:*)`, `Bash(scripts/server.sh setup)`, `Bash(scripts/server.sh pull)` и `Bash(scripts/server.sh stop:*)`. Чтобы дирижёр запускал R2 сам, уберите из `deny` строки `scripts/server.sh …`. `ssh`/`scp`/`rsync` напрямую можно оставить запрещёнными: скрипт вызывает их сам. Агенты этот файл не правят.
   - Учтите: в `allow` уже есть `scripts/server.sh push` и `scripts/server.sh status`, они остались от шаблона SDF. Как только появится `~/.config/phd_server.env`, агент сможет выполнить `push` без отдельного разрешения. Если это нежелательно, перенесите их в `deny`.

## 2. Первый запуск

```
scripts/server.sh push --dry-run | less       # список файлов, без сети; в конце сводка: число, размер, строк с test_locked = 0
scripts/server.sh push                        # код и данные на сервер, затем проверка, что test_locked на сервере нет
scripts/server.sh setup                       # Python 3.9.6, пакеты, ссылка, Rust, moex_iss (~5–15 мин; лог — data/run_server/setup_*.log)
scripts/server.sh run checkenv                # версии против FINAL_DESIGN §6
scripts/server.sh run repro                   # воспроизводимость (12 обучений)
scripts/server.sh status                      # ждать «repro … завершён, код 0»
scripts/server.sh pull                        # отчёт repro -> data/regen/R2/repro/, логи -> data/run/server/<время>/
```

Счёт R2 начинается только после `repro` с кодом 0, иначе — раздел 5.

## 3. Что передаётся и что нет

- **Код** берётся из копии, где лежит `scripts/server.sh` (можно переопределить через `PHD_CODE_ROOT`). Это `git ls-files --cached --others --exclude-standard` по каталогам `scripts models env moex_iss fastbt tests docs`: отслеживаемые файлы плюс новые неигнорируемые. Код на сервере каждый раз обновляется. В `CODE_VERSION.txt` на сервере записаны ветка, коммит и число незакоммиченных файлов.
- **Данные** берутся из основной копии `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading` (переопределяется через `PHD_LOCAL_ROOT`): `data/regen`, `data/calendar`, `moex_discovery/data/external/moex_iss/indices_10m` (`l2_lib.RVI10`), `moex_discovery/data/final_v6/candles_10m` (`l2_lib.CANDLES`). Дополнительные каталоги задаются через `PHD_PUSH_EXTRA="a b"`. На 26.10, 14:40 это ≈ 22,9 тыс. файлов и 10,8 ГБ.
  - По умолчанию данные идут с `--ignore-existing`: push не затирает то, что сервер уже посчитал, в том числе `data/regen/R2/**`. Флаг `--update-data` разрешает заменять файлы на более новые по времени.
- **`test_locked` исключён тремя слоями.**
  1. `find -prune` по `*test_locked*` при сборке списка.
  2. `grep` по готовому списку: если найдена хоть одна строка, передача отменяется.
  3. `rsync --exclude='*test_locked*'`, а после передачи — `find` на сервере. Если что-то найдено, ошибка и требование убрать.

  Кроме того, `run` не стартует, если на сервере есть что-то с `test_locked` в имени, а `status` об этом предупреждает. Закрыт и второй каталог `data/models_cache/test_locked`: он не входит в список и подпадает под шаблон.
- **pull** забирает только `data/regen/R2/**` и логи `data/run_server/*.{log,exit}`.
  - Файлы, которые на Маке уже есть, не перезаписываются (`--ignore-existing`).
  - Если файл есть на обеих сторонах, но размеры разные, это «КОНФЛИКТ»: он печатается и не трогается.
  - Логи кладутся в новый каталог `data/run/server/<время>/` (`data/run/` в `.gitignore`).
  - `pull --dry-run` печатает списки `new`, `log` и `КОНФЛИКТ`. Ему нужна сеть, чтобы увидеть файлы сервера.

## 4. Задачи R2 на сервере

```
scripts/server.sh run <имя> -- <команда…>      # фон: nohup setsid nice -n 10; лог data/run_server/<имя>.log, pid — .pid, код выхода — .exit
scripts/server.sh status                       # загрузка, ядра, диск, задачи (работает/завершён с кодом), хвосты логов, test_locked
scripts/server.sh stop <pid>                   # TERM всей группе процессов (воркеры multiprocessing тоже); --kill — KILL
```

- Перед командой выполняется `source env/activate_r2.sh`: `python3` берётся из окружения 3.9.6, `OMP/OPENBLAS/MKL_NUM_THREADS=1` (переопределяется через `PHD_THREADS`), `TZ=Europe/Moscow`.
- Параллелизм задаётся `--workers` у самих скриптов. Бустинг и так однопоточный (`l2_lib.fit`: `nthread=1`, `num_threads=1`).
- `stop` снимает только pid из `data/run_server/*.pid`.

Примеры по таблице агентов протокола:

```
scripts/server.sh run E_nogi      -- python3 scripts/analysis/r2_nogi.py <подкоманда> …
scripts/server.sh run B1_f1       -- python3 scripts/analysis/r2_roli_h1.py f1 --workers 5
scripts/server.sh run B2_f5       -- python3 -W ignore scripts/analysis/r2_roli_h5_h22.py f5 --workers 5
scripts/server.sh run B2_ost5     -- python3 -W ignore scripts/analysis/r2_roli_h5_h22.py role --role ost --h 5 --leg <нога> --file <…>
scripts/server.sh run O_level1    -- python3 scripts/analysis/r2_loyo.py level --level 1 --h 1 5 22
```

- Число одновременных задач подбирается так, чтобы сумма `--workers` была не больше числа ядер из `status`.
- Результаты пишутся туда же, куда на Маке: `data/regen/R2/{cache,O,T,U}/`. Это та же папка по ссылке на путь Мака.
- После задачи: `scripts/server.sh pull`, затем журнал агента на Маке (`docs/otchety/dev_log_r2_<агент>.md`).

**Что на сервере не запускается.** Всё, что читает или пишет `test_locked`:

- `r2_sborka_test.py`;
- `final_test_r2.py --scope test`;
- `r2_kesh_h1.py` в режимах, которые пишут или читают `*_test.parquet` в `test_locked`;
- режимы `d24_1.py fixed/tune`, `d17_1.py`, `d16_v9.py`, `wrap_v9.py`, которые пишут тест в `test_locked/<tag>/`.

Скрипты R2 уровня val (`r2_nogi`, `r2_roli_h1`, `r2_roli_h5_h22`, `r2_loyo`) пишут только val шагов 1–6. Если такая запись всё же произойдёт, следующий `run` и `push` остановятся.

## 5. Воспроизводимость (`server_repro_check.py`)

**Что делает.**

- Сначала сверяет SHA-256 входов с Маком:
  - `features_pit_v3*.parquet` и config (FINAL_DESIGN §6);
  - `D12_h5_selected_h5.csv`;
  - сохранённые ряды `D12_h5_{xgb,lgbm}_h5_val.parquet` и `..._wrap2_val.parquet`;
  - код `l2_lib.py`, `d24_1.py`;
  - версии пакетов.
  Если хоть что-то не сходится — выход 1 до обучения.
- Затем пересчитывает D12_h5 seed 42, val шагов 1–6, XGB и LGBM. Это 12 обучений тем же путём, что контроль F_5 агента Б2: `r2_roli_h5_h22.setup/points/fit_all`, код первого раунда без изменений.
- Сравнивает с рядами Мака по (date, ticker, step) и считает:
  - max |Δ ln rv_pred|;
  - q99;
  - max |Δ rv_pred|;
  - долю строк выше порога;
  - max по шагам;
  - best_iteration по шагам против Мака.
- Порог — 1e-6 по ln. При превышении — выход 1.
- Отчёт: `data/regen/R2/repro/repro_D12_h5_s42_<хост>_<время>.{csv,json}` и `…_trees.csv`.
- На Маке (Apple M4 Pro) тот же пересчёт дал 0 (XGB) и 1,8e-15 (LGBM): `data/regen/R2/cache/B2/_control/F5.csv`.
- Метрик качества нет. Шаги 7–9 из сохранённых файлов не читаются (`filters step ≤ 6`).
- Быстрая проверка без обучения: `run repro_in -- python3 scripts/server_repro_check.py --only-inputs`.

**Почему на x86 1e-6 может не держаться.**

- Однопоточность уже стоит: `nthread=1`, `num_threads=1`, `force_col_wise=True`.
- Остаются различия архитектуры:
  - SIMD (NEON против AVX2/AVX-512) и FMA в гистограммах и градиентах;
  - libm (`exp`/`log`);
  - BLAS (у numpy на Маке Accelerate, на Linux OpenBLAS) — в регрессии Хьюбера HAR-J (`cfg huber` у D12_h5), которая идёт в смещение ноги.
- Обычно это даёт 1e-12…1e-9. Хуже, если сдвиг переставляет порог бина или итерацию ранней остановки: тогда расхождение крупное на отдельных шагах.

**Что делать, если 1e-6 не держится.** Порядок — от дешёвого к дорогому; счёт R2 до решения не начинать.

1. **Прочитать диагностику.**
   - Если входы и версии совпадают, best_iteration везде те же и max |Δ ln| < 1e-4, это округление архитектуры, а не ошибка.
   - Если best_iteration разошлись, расходится траектория обучения. Смотрите `max_dln_by_step`.
2. **Потоки.** Проверьте в `.json`, что `OMP/OPENBLAS/MKL_NUM_THREADS=1`; при запуске через `run` это так. Повторите с `PHD_THREADS=1`. Убедитесь, что numpy на сервере использует OpenBLAS, а не MKL: `python3 -c "import numpy; numpy.show_config()"`.
3. **LightGBM `deterministic=True`.** Повторите проверку с `--lgb-deterministic`: обёртка добавляет флаг к `lgb.train` в процессе, код не меняется. Флаг действует вместе с `force_col_wise=True` (уже стоит) или `force_row_wise=True`. Если с ним сервер совпадает с Маком, а без него нет, нужно решение владельца — включать ли флаг во всём R2. Тогда F_h тоже пересчитывается с ним.
4. **XGBoost.** Отдельного флага детерминизма для `hist` на CPU нет: при `nthread=1` он детерминирован на одной архитектуре. Межархитектурную разницу флагами не убрать.
5. **Та же архитектура.** Сервер arm64 Linux (Graviton или Ampere) ближе к Маку: NEON, тот же порядок операций в LightGBM/XGBoost. libm и BLAS всё равно другие. Это дешевле всего, если провайдер позволяет.
6. **Если совпадения нет, а расхождение мало** (best_iteration те же, max |Δ ln| ≲ 1e-4), есть два пути на выбор владельца:
   - (а) **↺ порога**: контроль «seed 42 повторяет сохранённый ряд» в скриптах агентов (`control_tol: 1.0e-7` в `r2_B2_roli_h5_h22.yaml` и аналогах) ослабить до измеренного уровня. Записать строкой `↺` с датой в протоколе (до счёта; на результаты R2 не опирается).
   - (б) **Своя платформа для сравнений** (умолчание, см. ❓ ниже): F_h и все кандидаты R2 на seed 42–51 считаются на сервере. Парные разности для 2σ берутся только между рядами одной платформы. Сохранённые ряды Мака служат только для контроля с ослабленным порогом.

   Скрипты агентов со строгим контролем (`r2_roli_h5_h22.py f5`, `kontrol`) на x86 остановятся с кодом 2 — это ожидаемо, пока нет решения.

### 5а. Фаза A — «сервер принят» (`scripts/analysis/r2_repro_A.py`, протокол v3, раздел 6, строка S)

Это отдельный критерий от проверки ln-прогнозов с порогом 1e-6 из п. 5. Повторяются дизайны R1 на h = 5, seed 42, val шагов 1–6:
- D12_h5 XGB;
- D12_h5 LGBM;
- wrap2 — их среднее (F_5 первого раунда);
- эталон (а) HAR-J d0 с эмбарго (ВЫР-1), пересчитанный через `r2_core`.

QLIKE года считается по фактической RV Мака. Отдельно проверяется, что фактическая RV в пересчёте совпадает с Маком.

Сервер принят, если выполнены оба условия:
- **|ΔQLIKE| / QLIKE ≤ 1e-4** для каждого ряда и каждого года 2017–2021;
- **порядок рангов** четырёх рядов совпадает с Маком по каждому году и по среднему за 2017–2021.

Иначе выход 1, и по протоколу весь R2 считается на Маке.

Выход: `data/regen/R2/S/repro_A.csv` (по годам), `repro_A_ranks.csv` и `repro_A.json` (входы, SHA, платформа). На сервере: `scripts/server.sh run repro_A`, затем `pull`.

**Дым на Маке** — `python3 scripts/analysis/r2_repro_A.py --self`: сохранённые ряды сравниваются сами с собой, а HAR-J d0 пересчитывается. 26.10, 16:11: max rel = 0 по всем рядам и годам, HAR-J d0 пересчётом тоже 0, порядок рангов совпал во всех 6 срезах. Результат — `repro_A_self*.csv`.

## 6. Если pip не нашёл колесо

`setup` ставит только колёса (`--only-binary=:all:`), чтобы не собирать numpy/scipy из исходников с другим компилятором. Если на платформе нет колеса для cp39:
- Linux x86_64 и aarch64 (glibc ≥ 2.28) должны иметь все колёса;
- на старом glibc обновите ОС или возьмите образ Ubuntu 22.04/24.04;
- вариант `xgboost-cpu==2.1.4` вместо `xgboost` (без CUDA) — тот же код CPU, но имя пакета другое; только с записью `↺`.

Если micromamba не находит `python=3.9.6` в conda-forge: `PHD_PY_TOOL=pyenv scripts/server.sh setup` (нужны `libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev`).

## 7. Rust: `moex_iss`

`setup` ставит rustup с toolchain **1.93.0**: это версия Мака (`rustc 1.93.0 (254b59607 2026-01-19)`). Затем выполняется `cargo +1.93.0 build --release --locked --manifest-path moex_iss/Cargo.toml`, бинарник — `moex_iss/target/release/moex_iss`. `aws-lc-sys` (rustls) требует компилятор C и иногда cmake. Для счёта R2 крейт не нужен (он для сверки данных с ISS); пропуск: `PHD_SKIP_RUST=1`.

## 8. ❓ Открытые вопросы (умолчание действует, если нет ответа до 27.10, 15:00)

- ~~❓ Тестовые прогнозы победителей R2 после заморозки~~ — снято протоколом v3 (раздел 6): «Сборка теста (фаза F) — на Маке».
- ❓ **Если `repro` не держит 1e-6** (раздел 5). Умолчание: путь (б) — все ряды для сравнений R2 считаются на сервере, порог контроля в скриптах агентов поднимается до измеренного уровня строкой `↺`.
- ❓ **`allow` содержит `scripts/server.sh push` и `status`** (раздел 1, п. 5). Умолчание: оставить, потому что без `~/.config/phd_server.env` команды не работают.
- Протокол v3 (раздел 6) называет фоновый запуск `scripts/server.sh bg`. Здесь это `run`; синоним `bg` намеренно не заведён. В `allow` стоит `Bash(scripts/server.sh bg:*)`, и такой синоним обошёл бы запрет `run` из `deny`. Решение — за владельцем при снятии запрета.

## 9. `r2_core` — общая основа модулей R2 (агент S)

`scripts/analysis/r2_core.py` используют модули L, B, M, E, O, D: они только импортируют его, общий код R2 при этом не меняется. Настройки — `models/designs/r2/r2_S_server.yaml`, схема — `models/designs/r2/schema_v1.yaml`. Полный API описан в docstring файла.

**Схема и кэш**
- `validate(design)`. Дизайн задаётся вложенным dict, путём к yaml или ключами с точками (`{'boost.role': 'residual'}`). Возвращается полный дизайн: все `default` дописаны, числа нормализованы (`1.0 → 1`). Если что-то не так, выбрасывается `DesignError` со списком всех ошибок:
  - неизвестный ключ;
  - значение вне `values`;
  - изменён `fixed`;
  - в `features` нет G0;
  - нет `h`.
- **Единое правило нормализации с агентом O** (`r2_loyo._n`).
  - Схема читается через `yaml.safe_load`, поэтому `off`/`on` в ней — false/true.
  - В дизайне `on/off/yes/no/true/false` в любом регистре → bool; строка-число → число (`'0.5'` = 0.5, `3.0` = 3); `inf` остаётся строкой.
  - В каноне для хэша bool — `true`/`false`, числа — десятичные строки.
  - `python3 scripts/analysis/r2_core.py check-o` сверяет с O все 3 × 64 дизайна `factor_v1/h{h}.yaml`: схема, уникальность хэшей, `complexity = r2_loyo.slozhnost = meta`, совпадение правила на всех значениях схемы и пробных строках. На 26.10, 16:09 — OK, расхождений 0.
- **Ключ-скаляр, заданный списком**, — это сетка внутри складки. `grid_keys(d)` возвращает такие ключи, `variants(d, keys=[...])` перебирает их варианты; модуль сам указывает, какие сетки к нему относятся. `leg.enet.lambda/alpha` по умолчанию — вся сетка 3 × 3.
- `canonical(d)` строит канонический yaml по шапке схемы: без `design`, с `schema: v1`, числа — десятичные строки, `sort_keys`. `design_hash(d)` — его SHA-256.
- `cache_dir(d)` возвращает `data/regen/R2/cache/<sha256[:16]>/` с `design.yaml`. При повторном обращении канон сверяется; если он другой — ошибка (коллизия префикса или смена схемы). Корень кэша переопределяется через `R2_CACHE` (на сервере — локальный NVMe).
- `complexity(d)` — число ключей со значением ≠ `default` по п. 2.4; правило то же, что `r2_loyo.slozhnost`.
  - Не считаются `fixed`, `h`, `seed` и `ens.seeds`.
  - `features` считаются по одной группе сверх G0.
  - `leg.enet` идёт в счёт один раз и только если задано одиночное значение; подсетка — это выбор внутри складки.
- `write_series(d, имя, df, h, seed, module=...)` пишет файл `<имя>_h{h}_s{seed|det}_val.parquet`.
  - Нужны колонки date, ticker, year, step, rv_pred; rv_pred должен быть > 0 и конечным.
  - val с годами > 2021 записать нельзя.
  - Запись атомарная (tmp + `os.replace`).
  - Индекс у каждого процесса свой (`_index/<модуль>_<хост>_<pid>.csv`), это и есть «один писатель на воркер». `merge_index()` собирает их в `INDEX_R2.csv`.
- `read_series` и `have_series` — чтение и проверка наличия ряда.
- **`load_series(design_sha, part='val', ryad=None, h=None, seed=None, modul=None, steps=None)`** — точка входа для модулей-потребителей: E берёт через неё ряды B и M; так же могут O и D.
  - Возвращает все ряды дизайна по его хэшу одним длинным DataFrame: колонки ряда плюс `ryad`, `h`, `seed`, `part`, `modul`, `hash`.
  - `design_sha` — полный SHA-256, префикс ≥ 16 символов (= имя каталога кэша) или сам дизайн.
  - Фильтры принимают значение или список. `modul` — кто записал ряд: значение `module=` у `write_series`.
  - Источник — индекс кэша. SHA-256 файла сверяется с индексом: если файл переписан мимо `write_series`, будет ошибка. Ряды без строки индекса тоже берутся, с `modul = '?'`.
  - `read_index()` возвращает индекс без записи `INDEX_R2.csv`.

**Данные d0 и эмбарго**
- `load_d0(tickers=None, plant=False)` загружает данные через `r2_nogi.load(d0=True)` (код L): rv_d — это RV дня d. Строки > 2021 отбрасываются. Добавляются колонки `_tend{h}` — дата конца окна цели. `check_d0(A)` проверяет, что info_set = d.
- `train_mask(A, h, vy)` — годы < vy, и окно цели [d+1, d+h] кончается до 1 января vy (эмбарго h). `val_mask(A, h, vy)` — val-год vy.
- `inner_masks(A, h, vy)` нужна для ранней остановки по QLIKE итога (п. 4). Обучение — < vy−1 с эмбарго; внутренняя складка — vy−1 без окон, заходящих в vy; итоговое переобучение — `train_mask(vy)`.
- `step_years(k)` возвращает (vy = 2015 + k, vy − 1). `assert_dev_step(k)` пропускает только шаги 1–6.

**Пул**
- Данные кладутся в общий словарь вызовом `share(A=A, ...)` один раз в родителе, до пула.
- `run_pool(fn, tasks, workers)`:
  - fork; воркеры видят `SHARED` без pickle;
  - BLAS = 1 поток на воркер;
  - задачи с `cost` идут дорогими вперёд;
  - ошибка задачи возвращается как `{ok: False, error}`, остальные задачи продолжаются.
- Число воркеров `n_workers()`: `R2_WORKERS` → аргумент → vCPU − 8.

**Проверки**
- `shift_check(V, A, h, имя, design, K=None)` — п. 2.6 по правилу агента O (PR #167). Это импорт, а не копия: корреляции считает `r2_loyo.sdvig`, решение принимает `r2_loyo.sdvig_ok`.
  - Корреляции: Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2, −1, 0, +1}, на val 2017–2021; по тикерам, затем медиана.
  - **Ряд проходит, если min K ≤ argmax ≤ 0**, где K — последние известные дни входов ряда. `+1` блокируется всегда.
  - K задаётся явно или берётся из `r2_loyo.K_dizaina(design)` по группам признаков (метаданные F, `r2_O_loyo.yaml` `sdvig.izvestnyy_den_grupp`; для G1 K = {−1, 0}). Без дизайна K = {0}.
  - Блок записывается в `R2/O/blok_sdvig.csv` через `r2_loyo.blok_zapis`: формат O, с полем `posledniy_izvestnyy_den`.
- `leak_check(build, A, имя, cuts)` — детектор как в Д26 и ВЫР-1, срезы только ≤ 2021. `build(A)` возвращает `{ряд: DataFrame}`.
  - Срез в середине года: на строках ≤ среза Δ = 0 (модели шагов vy ≤ год среза), после среза Δ > 0, подложенная утечка «вход d := d+1» видна до среза.
  - 31.12: проверка эмбарго. Ряд должен содержать прогнозы модели шага vy = год + 1 на строках ≤ среза (`part = all`).
  - `perturb_frame` и `plant_frame` доступны и по отдельности.

**Фазы**
- `run_phase(фаза, h, дизайны, fn, seeds, steps)`:
  - сначала ворота: B — SHA плана; C/D/E — файл `R2/O/gate_{пред}_h{h}.yaml` и SHA его `inputs`;
  - затем валидация, хэш, каталог кэша;
  - задачи (дизайн × seed) уходят в пул; `fn(design, seed, steps)` возвращает `{ряд: DataFrame}`;
  - каждый ряд записывается и проходит `shift_check`; в конце ставится отметка `DONE_h{h}_s{seed}`, и готовые задачи при повторе пропускаются.
- Сводка пишется в `R2/S/phase_<фаза>_h{h}_<время>.csv`.

**Дым (Мак)** — `python3 scripts/analysis/r2_core.py smoke --workers 3`: SBER, LKOH, AFLT × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}, нога HAR-J d0 с эмбарго. Проверяются:
- самопроверки схемы (17);
- d0 и эмбарго;
- `run_phase` → кэш → `shift_check` (argmax = 0); `load_series` по хэшу возвращает те же ряды, фильтр по модулю работает;
- правило O: ряд на d−1 при K = {0} блокируется, при K(G0 + G1) = {−1, 0} проходит; ряд = RV_{d+1} блокируется (+1);
- положительный контроль: та же нога на днях d−1 блокируется (argmax = −1 на всех h);
- детектор: Δ до среза = 0, подлог пойман; без эмбарго граница года «протекает»;
- пул: fork, результат совпадает с последовательным счётом, задача с ошибкой изолирована.

Итог — `data/regen/R2/S/smoke/smoke_core.csv`. На 26.10, 16:05 — ВСЕ OK, примерно за 2 с.

Разбор дизайна: `python3 scripts/analysis/r2_core.py show <design.yaml>` печатает полный дизайн, хэш, каталог кэша, сложность и сетки.
