# Задание kodir — Л-4 «ось filtr» (прогон этапа 1)

## Источники
- Контракт: `docs/otchety/2026-10-28_progon_etap1.md` §1.1 (ось `filtr`, перевод `sobytiya`) и §2.1 (определения девяти значений — точные формулы).
- Определения-первоисточники: `docs/otchety/2026-10-28_kontrakt_fdr_iter16_20.md` §2.1 (z_d, σ̂_{d−1}, IMOEX), §2.2 (Т16), §2.3 (Т17), §2.5 (Т19), §2.6 (Т20), §2Б.3 (И-21).
- Код: `fastbt/sobytiya_e1.py`, `fastbt/rezhim_e1.py` (`imoex_daily`), `fastbt/e1_sektor.py` (`allow`, центр итерации 4), `fastbt/sigma_real.py` (rv5, rv22, rv66), `fastbt/etap1_data.py` (`gate_trades`).
- База ≥ `45a93e6`. Ветка `progon-L4-filtr`.

## Цель
- `fastbt/filtr_e1.py` — фильтры по собственным рядам тикера: `post_shok`, `szhatie`, `rv_perevorot`, `obyom`.
- `fastbt/filtr_e1_rynok.py` — фильтры по рынку: `rel_sila_imoex`, `sektor_trend`, а также `cbr` и `div` через существующие `inp.sobytiya`.

Единая функция:
```python
ZNACHENIYA = ("net", "cbr", "div", "post_shok", "szhatie", "rv_perevorot", "obyom", "rel_sila_imoex", "sektor_trend")
def razresheno(imya: str, t: str, inp, sd: np.ndarray, dates: np.ndarray) -> np.ndarray | None
```
- Возвращает bool «вход разрешён» по дням тикера `t`. `sd` — сторона сигнала дня (+1, −1, 0).
- `None` — значение не считается (например, `div` без сверенных дат 2016) → оценщик пишет строку `ne_schitana`.
- Только данные ≤ close d.
- Нет истории → вход запрещён.

## Владение
`fastbt/filtr_e1.py`, `fastbt/filtr_e1_rynok.py`, `tests/test_filtr_e1.py`.

## Тесты и мутации
- Синтетика с известным ответом на каждый фильтр, обе стороны.
- Для каждого из 7 новых фильтров:
  - мутация «данные d+1» (сдвиг ряда на −1);
  - мутация «сторона перепутана».
- Д1 — для `z`, `Rg`, `V`, `U`, `rel`: корреляция с исходной дневной величиной при сдвигах −2…+1, максимум на 0. Это первая строка вывода теста.
- `net` → все `True`. `cbr` → побайтно `~inp.sobytiya["cbr"]`.

## Гейты
- G1: `pytest tests/test_filtr_e1.py -q` → 0 failed.
- G2: 14 мутаций пойманы.
- G3: Д1 — максимум на сдвиге 0 у всех рядов.
- G4: только владение.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, метрики 2020–2021, счёт стратегий;
- `etap1_eval.py`, `exec_e_run.py`, `sweep_drivers.py`, `sobytiya_e1.py`, `e1_sektor.py` (только читать);
- `pytest tests` целиком;
- `skipif` / `xfail`;
- `ln`, `finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения.

## Отчёт
До 10 строк, «закрыто N из M», отступления (каждое отличие формулы от §2.1 — отдельной строкой), расход.

## Лимит
- до 2 процессов;
- ~45 мин (p90 75);
- 500k;
- коммит раз в 30 минут.

Правила: `polnota.md`, `testy.md`, `mutations.md`, `dannye.md` (Д1, Д3).
