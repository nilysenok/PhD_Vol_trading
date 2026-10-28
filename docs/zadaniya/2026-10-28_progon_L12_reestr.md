# Задание kodir — Л-12 «реестр отбора из ledger v2» (прогон этапа 1)

## Источники
- Контракт `docs/otchety/2026-10-28_progon_etap1.md` §1.2 (поля), §1.5 (`Chitatel`), §6 (реестр: файл, колонки, строка-указатель).
- Образец: `fastbt/reestr_otbora_v1.py` (`dopisat` — отказ при пропуске, только дополнение) — читать и вызывать, не править.
- §4 п. 4: все кандидаты, все метрики и флаги в одной таблице, только дополнение, ничего не удаляется.
- База ≥ `45a93e6`. Ветка `progon-L12-reestr`.

## Цель
`fastbt/reestr_progon.py`:
```python
VYHOD = "docs/otbor/reestr_otbora_progon_v2.csv.gz"
def kolonki() -> list            # strategiya, istochnik, klyuch_sha, klyuch (канон. JSON), kod, kod_slit, sloy, status, prichina,
                                 # sharpe_2016…2019, sdelok_2016…2019, kv_2016Q1…kv_2019Q4 (16), mediana_kv, dolya_pol_kv,
                                 # zaem5_mediana_kv, zaem10_mediana_kv, msr_2016…2019, vyrozhdena, kray, region_id, chlen_regiona,
                                 # kandidat, wf_pravilo, komissiya, fdr
def dopisat_iz_ledger(root, sid, otbor: dict | None = None) -> int
    # все строки Chitatel (status ok, ne_schitana, oshibka; все istochnik); известные (sid, istochnik, klyuch_sha) не дублируются;
    # новый член gzip дописывается в конец; число записанных + уже известных == числу строк ledger, иначе ValueError
def ukazatel(sid) -> dict        # строка в docs/otbor/reestr_otbora_A_v1.csv через reestr_otbora_v1.dopisat: istochnik=progon_v2,
                                 # prichina = "sha256 <gz>; строк <n>"
```
`otbor` — разметка региона и WF (Л-7, Л-8): дописывается новыми строками с `status = otbor`, старые не меняются.

## Владение
`fastbt/reestr_progon.py`, `tests/test_reestr_progon.py`, `docs/otbor/reestr_otbora_progon_v2.csv.gz` (создаёт только код по команде дирижёра).

## Тесты и мутации
Синтетический ledger через `ledger_v2.Pisatel` в `tmp_path`:
- все строки записаны;
- повтор ничего не добавляет;
- gzip читается целиком после двух дописываний.

Мутации:
- пропуск строки `ne_schitana`;
- пропуск строки контроля;
- перезапись файла вместо дописывания;
- 16 кварталов → 15 колонок.

## Гейты
- G1: `pytest tests/test_reestr_progon.py -q` → 0 failed.
- G2: мутации пойманы.
- G3: только владение.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, метрики 2020–2021;
- `reestr_otbora_v1.py`, `reestr_otbora_A_v1.csv` вручную (только через `dopisat`);
- `pytest tests` целиком;
- `skipif` / `xfail`;
- `ln`, `finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения.

## Отчёт
До 10 строк, «закрыто N из M», расход.

## Лимит
- 1 процесс;
- ~25 мин (p90 45);
- 500k;
- коммит раз в 30 минут.

Модель — haiku. Правила: `polnota.md`, `otbor_statistika.md` п. 4, `mutations.md`.
