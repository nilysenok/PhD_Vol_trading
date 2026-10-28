# Задание kodir — Л-7 «регион и плато» (прогон этапа 1)

## Источники
- Контракт `docs/otchety/2026-10-28_progon_etap1.md`:
  - отступления ↺ «регион в проекции» и ↺ «ансамбль рядов»;
  - §1.2 (`by_quarter`, `mediana_kv`, `dolya_pol_kv`), §1.5 (`Chitatel`);
  - §3 IG-7, §5 вопрос 1.
- Ж-080: PBO — той же процедурой, что отбор. Модуль `fastbt/pbo_pravilo.py` (`pbo_cscv_pravilo(M, coords, select_fn, nblk)`) ведёт лист `pbo-pravilo-kod`, он тебе **не принадлежит**: ты подаёшь `select_fn`.
- Код (образцы): `fastbt/etap1_poisk_vorota.py` (`dopustima`, `pick_best`, `sigma126`, TIE = 0,05), `fastbt/sweep_drivers.py:74-92` (`plateau_score`; только читать).
- База ≥ `45a93e6`. Ветка `progon-L7-region`.

## Цель
`fastbt/progon_region.py`, чистые функции над матрицей и строками (данных не читает):
```python
K_OSI, RADIUS, DOLYA_MIN, MIN_CHLENOV = 4, 1, 0.8, 10
def rang_osey(rows, tsel, osi) -> list           # полный индекс Соболя (Jansen) целевой по оси; на переданных строках
def regiony(rows, osi_top, radius) -> dict       # центр (индексы по osi_top) -> индексы строк в |Δиндекса| ≤ radius по каждой из osi_top
def ryad_regiona(M, chleny) -> np.ndarray        # среднее столбцов (равные доли капитала)
def vybor_region(M, rows, daty, kvartaly_idx=None, radius=RADIUS, tsel="mediana") -> dict | None
    # на строках kvartaly_idx (None — все 16): ранжирование осей -> регионы -> допустимые (членов >= MIN_CHLENOV,
    # доля кв. с SR ряда региона > 0 >= DOLYA_MIN, центр допустим по активности, как dopustima) -> лучший по
    # tsel ("mediana" | "dolya"), ничья <= 0,05 — меньшая sigma126 ряда; -> {centr, chleny, ryad, score, osi}
def select_fn_region(rows, daty, coords) -> callable   # обёртка под сигнатуру pbo_pravilo (❓ ниже)
```

## ❓ для `pbo-pravilo-kod`
Правило региона выбирает **ансамбль**, а не столбец.

Просьба к PBO: пусть `select_fn` сможет вернуть ряд, и его ранг на test-половине считается среди всех N столбцов.

Умолчание до ответа: `select_fn` возвращает индекс столбца — член региона с лучшей целевой на train-половине. Это пишется в отчёт отступлением.

Блоки CSCV на В1 — 8 блоков по 2 календарных квартала, если `pbo_pravilo` принимает метки блоков. Иначе — равные блоки, как у него.

## Владение
`fastbt/progon_region.py`, `tests/test_progon_region.py`.

## Тесты и мутации
Синтетика с известным ответом:
- куб 4 оси × 5 значений плюс 2 шумовые оси;
- плато вокруг известного центра;
- шумовой пик в одной точке → регион выбирает плато, а не пик;
- доля + кварталов 0,75 → регион не допустим.

Мутации:
- argmax точки вместо региона (обязана сменить выбор на синтетике);
- оси ранжируются по всем 16 кварталам, хотя передана половина;
- `DOLYA_MIN` сравнивается по `≥` с nan как положительным;
- ансамбль — медиана вместо среднего.

## Гейты
- G1: `pytest tests/test_progon_region.py -q` → 0 failed.
- G2: мутации пойманы.
- G3: только владение.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, метрики 2020–2021;
- `fastbt/pbo_pravilo.py`, `etap1_poisk_vorota.py`, `sweep_drivers.py`;
- `pytest tests` целиком;
- `skipif` / `xfail`;
- `ln`, `finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения.

## Отчёт
До 10 строк, «закрыто N из M», отступления, расход.

## Лимит
- до 2 процессов;
- ~50 мин (p90 80);
- 500k;
- коммит раз в 30 минут.

Правила: `polnota.md`, `otbor_statistika.md` п. 3–5, `mutations.md`.
