"""V7-4 (Ж-153 §1.1): минимальное удержание контртренда — Params.min_uderzhanie = N дней (в yaml #909: `min_uderzhanie: {dney: 2}`).

Выход по сигналу и по τ раньше N-го дня удержания ПЕРЕНОСИТСЯ на N-й: пока j − ej < N (ej — шаг ВХОДА, не сигнала: вход по close d,
лимитом d+1, в T того же дня), выход не исполняется, но ранний сигнал/τ запоминается (`reshenie`, s.mu = (сделка, причина));
на N-й день выход обязателен — по той же цене, что у обычного выхода дня N (close дня; t_vyhod — как у Run2), даже если сигнал вернулся.
Стоп и тейк действуют с первого дня и снимают отложенный выход (память привязана к объекту сделки, в новую позицию не течёт).
Разворот в T (exec_e_t.RunT._scan, причина "razvorot"): до N-го дня не исполняется, запоминается (`zapomnit_razvorot`, s.mu_raz = сделка);
на N-й день закрытие по cena_T обязательно, даже если сигнал вернулся за позицию; новая позиция — по правилам входа дня N (_decide:
есть сигнал и нет блока — вход в T по cena_T). ❓ умолчание: вернувшийся сигнал за позицию = закрыть и открыть заново (комиссии платятся).
Константа ≤ 1 или не задана — путь v6 побайтно. Контртренду S1, S2, S5, S6 — EvaluatorV7(min_uderzhanie=N) кладёт её в Params и в ключ кэша
(_fkey); тренду (S3/S4 и прочим семействам) — нет, `dney` её обнуляет. Быстрый слой: поле вне FAST_POLYA — точка уходит эталону (_slow_only)."""
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


def reshenie(s, i: int, j: int, vrem: bool) -> str | None:
    """Причина выхода дня j по signal/vremya (vrem — τ истёк) с переносом: до N-го дня — None, но запомнено; с N-го — текущая
    причина или отложенная. Память — только для текущей сделки (s.tr)."""
    now = "vremya" if vrem else ("signal" if s._sig_exit(i) else None)
    if rano(s, j):
        if now and not (s.mu and s.mu[0] is s.tr):
            s.mu = (s.tr, now)
        return None
    return now or (s.mu[1] if s.mu and s.mu[0] is s.tr else None)


def zapomnit_razvorot(s) -> None:
    """Разворот в T раньше N-го дня: отложен на N-й (RunT._scan)."""
    s.mu_raz = s.tr


def razvorot_otlozhen(s) -> bool:
    return s.mu_raz is s.tr and s.tr is not None
