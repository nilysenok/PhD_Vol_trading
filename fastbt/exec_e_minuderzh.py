"""V7-4 (Ж-153 §1.1): минимальное удержание контртренда — Params.min_uderzhanie = N дней (в yaml #909: `min_uderzhanie: {dney: 2}`).

Выход по сигналу и по τ раньше N-го дня удержания переносится на N-й: пока j − ej < N (ej — шаг ВХОДА, не сигнала: вход по close d,
лимитом d+1, в T того же дня), условия signal/vremya не проверяются; на N-й день проверяются как обычно (сигнал смотрится заново).
Стоп и тейк действуют с первого дня — их не трогает (скан баров в exec_e_run не знает о константе). Правило исполнения, не ось: в N не входит.
Константа ≤ 1 или не задана — путь v6 побайтно. Контртренду S1, S2, S5, S6 — EvaluatorV7(min_uderzhanie=N) кладёт её в Params и в ключ кэша
(_fkey); тренду (S3/S4 и прочим семействам) — нет, `dney` её обнуляет. Быстрый слой: поле вне FAST_POLYA — точка уходит эталону (_slow_only).
Не покрыто: разворот в T (exec_e_t.RunT._scan, причина "razvorot") — выход по сигналу в T до N-го дня остаётся (лист V7-6)."""
from __future__ import annotations

KLYUCH = "min_uderzhanie"
KONTRTREND = ("S1", "S2", "S5", "S6")             # Ж-153 §1.1


def kontrtrend(sid: str) -> bool:
    return sid in KONTRTREND


def dney(sid: str, n: int) -> int:
    """Константа куба N для семейства sid (EvaluatorV7): контртренд и N ≥ 2 — N, иначе 0 (путь v6 побайтно)."""
    return int(n) if int(n) > 1 and kontrtrend(sid) else 0


def rano(s, j: int) -> bool:
    """Шаг j (день удержания j − ej от дня входа) раньше N-го: выход по signal/vremya запрещён. s — Run с позицией (s.ej, s.p)."""
    n = s.p.min_uderzhanie
    return n > 1 and j - s.ej < n
