# Контракт оценщика перебора (Ж-034, Ж-035)

Фиксирует код ветки `sweep-engine` (7275ce2: `fastbt/sweep.py`, `sweep_stat.py`, `sweep_spec.py`) и рамку модуля владельца `fastbt/sweep_drivers.py` (2d732e1).

## Функция
- `Sweep.evaluate(params: dict, period: str = "train", comm_pct: float | None = None) -> dict`.
- `obj` (`{kind: Sk, sid: S1…S6, tickers}`), база точки и комиссии задаются в `Spec`. `params` накладываются на `Spec.base`.
- `comm_pct` — % на сторону; `None` = первая в `Spec.comm_pct` (комиссия отбора 0,04). Комиссия списывается при входе и при выходе, позиция на последнем баре платит выход.
- Драйверам `sweep_drivers` передаётся замыкание по комиссии: `lambda params, period: sweep.evaluate(params, period, 0.04)`.
- Параметры сигнала — ключи `sigprov.SIG_GRIDS[sid]`, они входят в ключ кэша. Остальные параметры — PnL.

## Ключи строки (отдаёт код сейчас)
| ключ | смысл |
|---|---|
| параметры точки, `comm_pct`, `period` | `base ⊕ params`; % на сторону; имя периода |
| `sharpe` | годовой Шарп избыточной над ставкой ЦБ, mean/sd(ddof=1)·√252 |
| `by_year` | {год: Шарп}; `_flat` разворачивает в `sharpe_<год>` |
| `sr_d`, `T`, `skew`, `kurt` | дневной SR, число дней, асимметрия, эксцесс сырой (нормаль = 3) — для `sweep_drivers.dsr` |
| `msr`, `sortino`, `mdd` | `metriki_e.msr`, `sortino_e`; `mdd` — доля от пика, ≥ 0 |
| `oborot_y`, `dni_v_rynke`, `sdelok_y` | Σ\|Δпоз\| в год; доля дней с экспозицией > 0; сторон в год на пару |
| `N`, `n_mult` | число уникальных точек ledger × n_mult (n_mult = 2 при повторном открытии validation) |

### v1.1, не реализовано
`excess_d` (дневной ряд — вход `pbo_cscv`; без него PBO считать не из чего), `v`, `okno_s`, `okno_po`, `chestnaya`.

## DSR и цель
- Ledger проб ведётся только при комиссии отбора 0,04 и вне validation; SR пробы = `sr_d`. N — число уникальных точек (`pkey`). Повтор точки пробой не является; 0,035 N не увеличивает (толкование 27.10, `tolkovaniya.md`). `n_trials` модуля владельца — отдельной колонкой.
- Цель — `sweep_drivers.objective_stable`: среднее `by_year` − 0,5·σ, σ по годам с ddof=0, как в модуле. ❓ Вопрос владельцу 27.10; по умолчанию ddof=0.
- Цель OAT и итераций считается на `setup` (tune 2018). `control` (ctrl 2019) — отдельной колонкой, в цель не входит. Сейчас `sweep_run` (#302) гоняет OAT на `train`, и 2019 попадает в цель — это расхождение с контрактом, править в коде.

## Периоды (`sweep_stat.Periods`)
`setup` 2018 · `control` 2019 · `train` 2018–2019 · `validation` 2020–2021.

Validation открывается только с флагом на диске; повторное открытие — только с `reopen_note`. Даты ≥ 2022 — `PermissionError`. Ряд дневной до 18:50 (Н6). Сигнал по close t, доход t → t+1.

## Ключ кэша (Д5)
`SignalCache.key = sha1(json([obj, sorted(sig_params), series_digest, code_id]))`:
- `series_digest` — sha1 склейки дайджестов рядов тикеров;
- `code_id` — `cache.code_hash()`.

Кэшируются только сигналы; мемо PnL — на прогон, ключ `(pkey, comm, годы)`.
