# fastbt/signals — семейства сигналов

Контракт — `interface.py` (докстрока модуля). Коротко:

- Модуль семейства: `fastbt/signals/s1.py` … `s6.py` (S1/S2 могут жить в одном файле, тогда
  `s2.py` = `from fastbt.signals.s1 import *`). В конце модуля — `register(Family(...))`.
- `generate(td, ind, grid) -> SignalSet`: K конфигураций × T баров, `sig` int8, `exit_l/exit_s`
  bool, `strength` f64, `extra` (S5: P,S1,S2,R1,R2; S6: vwap,dev), `split` как в старом коде.
- `grid` — сетка daily из c22a4b3: S1–S4 `SW.SIGNAL_GRIDS`, S5 `S5R.S5_SIG_DAILY`,
  S6 `S6R.S6D_SIG` (V41 для печатей). Порядок — `expand_grid` (ключи по алфавиту).
- `ind` — `fastbt.indicators.compute_base` (побитово = SW.compute_base; проверено SBER/IRAO/MTLR).
  Нужен свой индикатор семейства (SMA/STD окна, Donchian, Supertrend, VWAP, pivot) —
  считать внутри семейства, теми же формулами (pandas rolling там, где старый код pandas).
- `legacy(L, td, params) -> (sig, exit_l, exit_s|None, strength)`: вызов старых функций
  c22a4b3 на одной конфигурации (`L = fastbt.legacy.load_legacy()`: `L.sw`, `L.v4`, `L.v41`,
  `L.s5r`, `L.s6r`).
- Тест семейства: `fastbt/tests/test_signals_<sid>.py` → `check_family(fam, td, ind)`
  по SBER, IRAO, MTLR; `bad` должен быть пуст.
- Кэш делает ядро (`cache_key`): хэш исходника модуля семейства входит в ключ.
- Данные: `fastbt.data.load_ticker(ticker)` (только чтение основной копии).

RM-сетки, бэктест, метрики, отбор, голосование, B/C/D — ядро (не семейства).
