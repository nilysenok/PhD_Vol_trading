# Задание kodir — Л-2 «новые оси исполнения» (прогон этапа 1)

## Источники
- Контракт: `docs/otchety/2026-10-28_progon_etap1.md` §1.1 (оси `ed_lim`, `zhizn`, `gap`, `t_vyhod`, их смысл), §1.8, §2 (владение), §3 IG-2.
- Код (читать полностью): `fastbt/exec_e_run.py`, `fastbt/exec_e.py`, `fastbt/exec_e_types.py`, `fastbt/exec_e_rules.py`; места вызова — `fastbt/etap1_run.py`.
- Соседний лист Л-3 (`fastbt/exec_e_vremya.py`) даёт:
  - `bar_do(day, hhmm) -> int | None`;
  - `vyhod_plan(t_vyhod, prichina) -> (sdvig_dnya, hhmm | None)`.

  Пиши вызовы против этих сигнатур. Тесты с реальными временами — после слияния Л-3 в твою ветку (merge его ветки, без правок его файлов).
- Ж-083: `exec_e_run.py` принадлежит листу скорости (`skorost-etap1`). **Не трогать.**
- База `origin/kanon-v22-pereschet` ≥ `45a93e6`. Ветка `progon-L2-ispolnenie`.

## Цель
Подкласс `Run2(Run)` в новом `fastbt/exec_e2.py`.

Поля `Params` (все со значением по умолчанию = нынешнее поведение побайтно):
- `zhizn: int | None = None`;
- `gap_propusk: bool = False`;
- `t_vyhod: str = "close"`;
- `bar_plus: int = 0`;
- `stop_tik: int = 0`.

`Day.t: np.ndarray | None = None` — начало бара в минутах от полуночи; поле есть, заполняет Л-3.

`exec_e.run(…, ed=None)`: `ed[d]` — единица лимита дня сигнала (§1.1). `Run2` выбирается, только если задан `ed` или хоть одно новое поле не по умолчанию; иначе — прежний `Run`. `_check_params` проверяет:
- `t_vyhod ∈ {close, 1500, 1700, open2}`;
- `zhizn ∈ {None, 12, 36, 60}`;
- `bar_plus`, `stop_tik` ≥ 0.

Правила `Run2`:
- **ed:** `L = c_d·(1 − sd·k·ed_d)`; стоп, тейк и размер — по σ̂.
- **zhizn:** исполнение только на барах b < N дня d+1.
- **gap_propusk:** open бара 0 уже за L (лонг `o₀ ≤ L − tick`) → заявка снята, `blocked = sd`.
- **t_vyhod ≠ close:**
  - выход, решённый на close x, переносится по `vyhod_plan`;
  - в день исполнения бары до `bar_do` сканируются как обычно, стоп первым;
  - на баре `bar_do` — выход по `c[b]`, на `open2` — по `o[0]` дня x+1, причина прежняя;
  - комиссия — в строке начала интервала;
  - нет такого бара (короткий день) — выход по close Н6 этого дня, флаг в `Trade.reason` не меняется;
  - счётчик `Result.x_t` — сколько выходов исполнено не по close.
- **bar_plus = 1:** найденное на баре b исполнение (лимит, стоп, тейк) идёт по open бара b+1, если он хуже уровня; иначе — по уровню. Последний бар дня — по close Н6.
- **stop_tik = 1:** цена стопа хуже на 1 тик.

## Владение
`fastbt/exec_e2.py` (новый), `fastbt/exec_e_types.py`, `fastbt/exec_e.py`, `fastbt/exec_e_rules.py`, `tests/test_exec_e2.py`.

## Тесты и мутации (синтетика с известным ответом, без данных)
- Умолчания: `exec_e.run` с новыми полями по умолчанию и `ed=None` — побайтно прежний `Result` на 30 синтетических лентах. Плюс `pytest tests/test_exec_e*.py tests/audit_fastbt/test_14_stop_take_bar.py tests/audit_fastbt/test_15_gap.py -q` — без новых красных.
- `ed = σ̂` → побайтно как без `ed`.
- Мутации:
  - лимит исполнен на баре ≥ N при `zhizn = N`;
  - гэп исполнен при `propusk`;
  - выход по сигналу в 15:00 дня x (а не x+1);
  - `bar_plus` берёт open того же бара;
  - `stop_tik` в лучшую сторону;
  - касание исполняет лимит (`mutations.md` № 2);
  - стоп после тейка в одном баре (№ 3).

## Гейты
- G1: `python3 -m pytest tests/test_exec_e2.py -q` → 0 failed.
- G2: мутации пойманы (вывод в PR).
- G3: регрессия выше — 0 новых красных против базы.
- G4: `git diff --stat` — только владение.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, метрики 2020–2021, счёт;
- `fastbt/exec_e_run.py`, `fastbt/etap1_eval.py`, `fastbt/sweep_drivers.py`;
- `pytest tests` целиком, `fastbt/tests/test_core.py`;
- `skipif` / `xfail`;
- `ln`, `scripts/finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения.

## Отчёт
До 10 строк: «закрыто N из M», отступления, расход. Разбор — в тело PR.

## Лимит
- до 3 процессов;
- ~60 мин (p90 100);
- если не укладываешься — режь на два PR: вход (`ed`, `zhizn`, `gap`, `bar_plus`) и выход (`t_vyhod`, `stop_tik`);
- порог передачи — 500k;
- коммит не реже раза в 30 минут.

Правила: `polnota.md`, `testy.md`, `mutations.md`, `dannye.md`, `pravki.md`.
