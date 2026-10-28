# Задание kodir — Л-5 «оценщик v2» (прогон этапа 1)

## Источники
- Контракт: `docs/otchety/2026-10-28_progon_etap1.md` — §1 целиком (строка v2.0, ряды по тикерам, тождество §1.4, `canon_v2`), §1.8, §3 IG-1…IG-3.
- Код (читать полностью): `fastbt/etap1_eval.py`, `fastbt/etap1_run.py`, `fastbt/etap1.py`, `fastbt/s9.py:602-657` (`port_rows`, `_finish`), `fastbt/rezhim_e1.py` (`razmer_ves`).
- Соседи (merge их веток, без правок их файлов):
  - Л-1 `ledger_v2` (`canon_v2`, `kvartaly`, `tselevaya`);
  - Л-2 `exec_e.run(…, ed)` и поля `Params`;
  - Л-4 `filtr_e1_rynok.razresheno`.
- Ж-083: `etap1_eval.py` принадлежит листу скорости — **не трогать**; наследуй.
- База ≥ `45a93e6`. Ветка `progon-L5-ocenshchik`.

## Цель
`fastbt/etap1_eval_v2.py`:
```python
NOVYE = dict(ed_lim="sigma", zhizn="den", lestnica="net", gap="vhod", t_vyhod="close", k_x="net", storona="obe", filtr="net")
class EvaluatorV2(Evaluator):
    def evaluate_v2(self, pt: dict, period: str, vozm: dict | None = None) -> tuple[dict, np.ndarray, np.ndarray]
        # (строка §1.2 без полей excess/potok — их ставит Pisatel, ex_port float64 [T], ex_tik float64 [17, T])
```

Правила:
- Все новые оси = умолчания → путь через `Evaluator` побайтно: та же `df["excess"]`, те же поля.
- `k_x` → `Params.a = k_x` при `b = None`.
- `ed_lim`: `atr14` = ATR14_d / c_d, `diap` = (h_d − l_d) / c_d — по `sk._td[t]` дня d; `pct` = 0,01.
- `storona`:
  - `long` — `sig = max(sig, 0)`;
  - `short_shag1` — шорт из позиций сигнальной точки со сдвигом пороговой оси (§1.1), лонг — из своей.
- `lestnica = x`: два `run_e1` с `w/2` и `k`, `k + x`, строки g, sides, expo, noch — суммой.
- `filtr`: `razresheno` → `gate_trades`; `None` → строка `ne_schitana`.
- `zaem`: `by_quarter_zaem` для 5 % и 10 % = ряд минус ставка × |вес шорта| × cal / 365 по тикеро-дням.
- `vozm` — ключи:
  - `sigma_mult` (все σ × m);
  - `bar_plus`, `stop_tik` (в `Params`);
  - `komissiya`;
  - `sig_fn` (замена позиций: плацебо, к2);
  - `sigma_lim_mediana` (σ лимита = медиана σ̂ календарного года).
- `run_e1(…, ed=None)` пробрасывает `ed` в `exec_e.run`.

## Владение
`fastbt/etap1_eval_v2.py` (новый), `fastbt/etap1_run.py`, `tests/test_etap1_eval_v2.py`.

## Тесты и мутации
- IG-2 `-k umolchaniya`: 50 случайных точек zh047 × 6 стратегий (seed 20261028) на `train4`. `excess_d` — sha256 равны, остальные поля v1.0 — `==`. Нужны данные ≤ 2019: `FASTBT_MAIN=<копия>`, `FASTBT_BARS1M=…/data/regen/D25/bars1m`.
- IG-3 `-k tozhdestvo`: nanmean тикеров = портфель (≤ 1e-12 float64).
- IG-1 `-k yakor`: режим якоря через `EvaluatorV2` = `tests/test_yakor_v1.py` (18 или 20 точек).
- Синтетика:
  - лестница с `x` → ∞ (вторая половина не исполняется) = половина веса;
  - `storona = long` → нет шортов.
- Мутации:
  - `k_x` при `k_s ≠ net`;
  - вес лестницы не w/2;
  - `long` пропускает шорт;
  - заём на лонгах;
  - ex тикера без `− u·rf·cal`;
  - `filtr` `None` посчитан как `net`.

## Гейты
- G1: `pytest tests/test_etap1_eval_v2.py -q` → 0 failed.
- G2: мутации пойманы.
- G3: IG-1, IG-2, IG-3 зелёные.
- G4: `pytest tests/test_etap1_run_yakor.py tests/test_dividendy_yakor.py tests/test_period_v1.py -q` — без новых красных.
- G5: только владение.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, **метрики 2020–2021**;
- значения метрик в выводе тестов (`--tb=no`, сравнение — sha и `==`);
- `etap1_eval.py`, `exec_e_run.py`, `sweep_drivers.py`;
- `pytest tests` целиком;
- `skipif` / `xfail`;
- `ln`, `finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения.

## Отчёт
До 10 строк, «закрыто N из M», расход; замер с/точку на 20 точках (только время) — в PR.

## Лимит
- до 3 процессов;
- ~60 мин (p90 100);
- 500k;
- коммит раз в 30 минут.

Правила: `polnota.md`, `testy.md`, `mutations.md`, `dannye.md`, `pravki.md`, `proverka.md` п. 3.
