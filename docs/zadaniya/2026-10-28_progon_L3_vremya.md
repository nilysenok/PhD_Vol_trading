# Задание kodir — Л-3 «время 5-минутных баров» (прогон этапа 1)

## Источники
- Контракт: `docs/otchety/2026-10-28_progon_etap1.md` §1.1, ось `t_vyhod`:
  - 15:00 и 17:00 — close бара, который кончается в это время;
  - выход по τ — день x, выход по сигналу — день x+1;
  - `open2` — open первого бара дня x+1.
- Код: `fastbt/etap1_data.py` (`build_days`, строки 41+), `fastbt/etap1.py` (`_flat`, строки 22–34), `fastbt/bars5m.py` (источник времени баров), `fastbt/exec_e_types.py` (`Day`).
- Поле `Day.t` добавляет Л-2 в `exec_e_types.py`: `t: np.ndarray | None = None`, начало бара в минутах от полуночи. До слияния Л-2 — merge его ветки в свою, без правок его файлов.
- База ≥ `45a93e6`. Ветка `progon-L3-vremya`.

## Цель
1. `fastbt/exec_e_vremya.py` (новый, чистые функции):
   - `bar_do(day, hhmm: int) -> int | None` — индекс последнего бара, у которого `t + 5 ≤ hhmm` (в минутах); нет — `None`;
   - `vyhod_plan(t_vyhod, prichina) -> (sdvig_dnya, hhmm | None)`:
     - `close` → `(0, None)` — вызывающий не меняет поведения;
     - `1500` / `1700` — по τ `(0, 900 | 1020)`, по сигналу `(1, 900 | 1020)`;
     - `open2` → `(1, None)`, с флагом open.
2. `build_days` заполняет `Day.t` из времени баров источника. День `pusto` → `t = None`.
3. `etap1._flat`: `t` входит в части ключа, если не `None` (Д5).

## Владение
`fastbt/exec_e_vremya.py`, `fastbt/etap1_data.py`, `fastbt/etap1.py` (только `_flat`), `tests/test_exec_e_vremya.py`.

## Тесты и мутации
Известный ответ:
- день из 108 баров с 09:50 → `bar_do(…, 1500) = 61`, `bar_do(…, 1700) = 85`;
- короткий день до 14:00 → `None`;
- пропущенный бар 14:55 сдвигает индекс.

Мутации:
- `bar_do` берёт бар, который кончается после hhmm;
- сигнал в 15:00 дня x;
- `t` не в `_flat` (подмена `t` при тех же ценах → тот же ключ: тест Д5, образец `tests/audit_fastbt/test_23_cache_keys.py`).

Регрессия:
```
pytest tests/test_bars5m.py tests/test_period_v1.py tests/test_etap1_run_yakor.py tests/audit_fastbt/test_23_cache_keys.py -q
```
Если тест фиксирует старый digest — не менять его, а доложить (класс B, дирижёр).

## Гейты
- G1: `pytest tests/test_exec_e_vremya.py -q` → 0 failed.
- G2: мутации пойманы.
- G3: регрессия без новых красных.
- G4: только владение.

## Запрещено
- данные ≥ 2022 (`build_days` и так отказывает ≥ LIMIT — не ослаблять), `test_locked/`, `.env`, метрики 2020–2021;
- `exec_e_run.py`, `etap1_eval.py`, `sweep_drivers.py`;
- `pytest tests` целиком;
- `skipif` / `xfail`;
- `ln`, `finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения.

## Отчёт
До 10 строк, «закрыто N из M», расход; разбор — в PR.

## Лимит
- до 2 процессов;
- ~25 мин (p90 45);
- 500k;
- коммит раз в 30 минут.

Модель — haiku. Правила: `polnota.md`, `testy.md`, `mutations.md`, `dannye.md`.
