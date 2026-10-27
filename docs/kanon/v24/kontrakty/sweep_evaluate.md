# Контракт оценщика перебора (Ж-034, Ж-035), строка v1.1

Контракт фиксирует уже сделанное. Рамка — модуль владельца `fastbt/sweep_drivers.py` (2d732e1); оценщик — WIP `sweep-engine` (`sweep.py`, `sweep_sk.py`, `sweep_stat.py`).

## Функции
- Оценщик: `evaluate(obj: dict, params: dict, period: str, commission: float) -> row: dict`.
- Драйверам (`oat_sweep`, `factorial_sweep`, `morris_screen`, `optuna_search`) передаётся замыкание `lambda params, period: evaluate(obj, params, period, commission)`, то есть `Evaluate = (params, period) -> dict`.
- `obj`: `{kind: Sk, sid: S1…S6, tickers: 17}` (`SkEvaluator`); `{kind: A_sborka, …}` и `{kind: C|E|B|D, sigma: …}` — адаптеры этапов Э1-г, Э2, Э3.
- `params`: сигнал — ключи `sigprov.SIG_GRIDS[sid]`, входят в ключ кэша; остальное — PnL (RM, исполнение, сборка), в ключ не входит.
- `commission`: % на сторону, 0,04 (отбор) или 0,035 (пересчёт). Списывается при входе и при выходе; позиция на последнем баре платит выход.

## Ключи row
| ключ | смысл |
|---|---|
| `v` | `"1.1"` |
| `sharpe` | годовой Шарп избыточной над ставкой ЦБ, mean/sd(ddof=1)·√252 — нужен драйверам |
| `by_year` | {год: `sharpe`} — нужен драйверам; `_flat` разворачивает в `sharpe_<год>` |
| `sr_d`, `T`, `skew`, `kurt` | дневной SR, число дней, асимметрия, эксцесс (не избыточный, нормаль = 3) — для `dsr(sr_d, trials_sr_d, T, skew, kurt)`; SR проб — в той же дневной шкале |
| `msr`, `sortino`, `mdd` | `metriki_e.msr`, `sortino_e`; `mdd` — доля от пика, ≥ 0 |
| `oborot_y`, `dni_v_rynke`, `sdelok_y` | Σ\|Δпоз\| в год; доля дней с экспозицией > 0; сторон в год на пару |
| `period`, `okno_s`, `okno_po`, `chestnaya`, `comm_pct` | окно данных; признак `net` / `kontrol` / `validation`; комиссия |
| `excess_d` | список дневной избыточной доходности — столбец матрицы T×N для `pbo_cscv`; `_flat` его не переносит в таблицу |

Цель — `sweep_drivers.objective_stable`: среднее `by_year` − 0,5·σ, где σ по годам считается с ddof = 0, как в модуле. Плато считает `plateau_score`, PBO — `pbo_cscv` (8 блоков); оба на стороне драйверов.

## Периоды
| имя | годы | роль |
|---|---|---|
| `tune2018` | 2018 | настройка, входит в цель |
| `ctrl2019` | 2019 | контроль, в цель не входит |
| `train` | 2018–2019 | отчёт |
| `validation` | 2020–2021 | один взгляд на этап, в цель не входит никогда; повторный — с записью изменения, N × 2 |

В коде имена `setup`/`control` — псевдонимы. Ряд дневной до 18:50 (Н6); сигнал по close t, доход t → t+1; прогрев читает историю до периода, PnL — только дни периода; даты ≥ 2022 — `PermissionError`.

## Ключ кэша
`sha1(json([obj, sorted(sig_params), series_digest, code_id]))`; `series_digest` — sha1 склейки дайджестов рядов тикеров, `code_id` — `cache.code_hash()`. Кэшируются только сигналы; мемо PnL — на прогон, по ключу `(params, comm, годы)`.
