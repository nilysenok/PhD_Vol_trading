# Общая матрица стратегий (Ж-051 п. 2): где лежит и как восстановить

`ledger_it1.pkl`, `ledger_it2.pkl` (6–48 МБ каждый) — pickle кэша `Ledger` (`etap1_poisk_vorota.Ledger.cache`: `{(ключ точки, период): (параметры, строка v1.0)}`). **В git не включены** (размер); лежат в рабочей копии `.claude/worktrees/agent-a33ab257766d548bf/docs/tables/zh051/_ledger/S{k}/` и на диске, пока копию не очистят. Контрольные суммы и N — в `SHA256SUMS.txt` (sha256 файла проверять до использования: `shasum -a 256 -c`-формат не используется, строки вида `хэш  путь  N=…`).

- `ledger_it1` — итерация 1: Моррис (тот же seed) + строки `docs/tables/zh047/S{k}_faktor.csv` + пробы плато; N = честные уникальные точки (на 20 меньше, у S5 на 40, чем N в `S{k}_itog_v2.csv` до поправки: двойной счёт проб, совпавших с точкой Морриса).
- `ledger_it2` — то же + 6 конфигураций ансамбля (ключи `("ansambl", набор, режим)`); итерации 3–10 загружают его и дописывают (`etap1_ansambl.dobavit_v_ledger`), PBO и N — по общей матрице.

Восстановление, если файла нет или хэш не сошёлся (нужны данные 2017–2019 в рабочей копии и `FASTBT_MAIN=<своя копия>`):

```
from fastbt import etap1, etap1_poisk as P
from fastbt.etap1_ansambl_eval import AnsamblEvaluator
from fastbt.etap1_ansambl_run import restore
from fastbt.etap1_eval import YAKOR
S = P.load_setka("S1")
evr = AnsamblEvaluator("S1", etap1.build_inputs("S1"), etap1.base_point("S1"), 0.0004)
led, center, ranked, itog_v2, n_itog_v2 = restore("S1", lambda p, per: evr.evaluate({**YAKOR, **p}, per), S)
assert led.n_unique == N из SHA256SUMS.txt (ledger_it1)
```
`restore()` останавливается (SystemExit), если Моррис + факторный не дали `N_do_dosheta`. Время: от 2 до 10 минут на стратегию (S1 ≈ 10).
