# Handoff агента S (R2) — 26.10.2026, 16:57

Передача делается по решению владельца 16:45. Новый агент продолжает работу на тех же файлах.

## Что сделано (слито)

- **PR #157** — серверный инструментарий: `scripts/server.sh`, `env/*`, `scripts/server_repro_check.py`, `docs/server_r2.md`. Сервер не подключали: по решению 16:25 весь счёт идёт на Маке.
- **PR #162** — ядро `scripts/analysis/r2_core.py`:
  - схема и канонический хэш, кэш `R2/cache/<hash16>/`;
  - пул fork;
  - d0 и эмбарго, `shift_check`, `leak_check`, `run_phase`.
- **PR #164** — критерий фазы A `scripts/analysis/r2_repro_A.py`. Сейчас не нужен: серверов нет.
- **PR #171** — `shift_check` по правилу O (импорт `r2_loyo`), `load_series(design_sha)` для E.
- **PR #184**:
  - `scripts/analysis/r2_dataset_cache.py` — бинарный LightGBM, QuantileDMatrix до fork, ключ с SHA `features_r2_v3`, RAM-диск;
  - режим R2 в `r2_core`: отказ записи ≥ 2022 и `otkaz_2022.csv`, флаг только для A;
  - `models/designs/r2/schema_v2.yaml`: k16, k18, k20, `boost.grid_point`; хэши v1 не меняются, проверено на 552 дизайнах (`r2_core.py check-o`);
  - входы h=1 в хэше (`inputs`, `with_inputs`, `needs_h1_inputs`);
  - DONE и `.owner` — по модулю;
  - `sdvig_zapis` для каждого ряда;
  - `scripts/analysis/r2_driver.py` и `models/designs/r2/r2_S_driver.yaml`: генерация всех yaml B–E в `models/designs/r2/driver_v1/`, конвейеры h = 5, 22, 1, эконометрика сразу, ворота через функции O, отчёты `R2/report/gate_X.md`, стоп перед открытием;
  - правило «фактор неприменим» (`apply_gate_B`, `factor_unsupported`, пометка в gate_C).

## В PR этого handoff (ветка `r2-S-driver-E`)

- Вызовы E (PR #187) и G (PR #188) в `r2_S_driver.yaml` переведены на пакетный `run --designs … --h … --seeds … --steps … --workers …`; фаза передаётся как `--faza` у E и `--phase` у G.
- `ens.ec` на h ≠ 1 добавлен в `r2_core.needs_h1_inputs`: ряд h=1 теперь входит в хэш.
- `r2_driver.contract_check()` проверяет, что опции шаблонов объявлены в argparse модуля. Результат входит в дым драйвера.
- Дымы 16:56 — ВСЕ OK:
  - `r2_driver.py smoke` (около 78 с);
  - `r2_core.py smoke`.

## Открытые поручения

1. Подключать пакетные `run` от L (`r2_nogi.py`), L3 (`r2_calib.py`), B (`r2_roli.py`, `r2_resid_grid.py`) и R (`r2_refit.py`) сразу после слияния. Порядок для каждого:
   - сверить шаблон в `r2_S_driver.yaml` (раздел `modules`) с argparse модуля;
   - `python3 scripts/analysis/r2_driver.py smoke` — всё OK, в строке контракта у модуля «OK»;
   - открыть PR.

   На 16:56 контракт: P, M, E, G — OK; L, L3 — нет `run`; R, B, BG — файлов нет.
2. Правило «фактор неприменим» уже в коде (PR #184). Модулю достаточно объявить `neprimenimye_faktory` в своём yaml (буквы A–O, ключи схемы или группы признаков); у P работает раздел `applicable`.
3. Пока не слиты все модули, боевой запуск драйвера — `python3 scripts/analysis/r2_driver.py run` (без `--once` он опрашивает ворота и модули). Начинать только по команде дирижёра.

## Фоновые процессы

Нет. Все дымы шли в переднем плане и завершены. Логи последних дымов:
- `data/regen/R2/S/driver_smoke/smoke_driver.csv`;
- `data/regen/R2/S/smoke/smoke_core.csv`;
- `data/regen/R2/S/smoke/smoke_dataset_cache.csv`.

## Запреты

- Сеть и сервер не трогать: `ssh`, `scp`, `rsync`, `scripts/server.sh setup/run/pull/stop` не запускать. Допустимы только `git push`, `git fetch` и `gh pr`.
- Синоним `bg` в `server.sh` не заводить: он обошёл бы запрет `run` в `deny`.
- `.claude/settings*.json` и `.env` не трогать; без `sed -i`, heredoc для файлов, `rm -rf`.
- Общий код других агентов не править.
