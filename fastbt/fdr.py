"""Занятость капитала на 18:50 по тикеру для учёта ФДР (контракт #376, §1.3 А).

Режим t0d — основной для новых сеток (Ж-111 п. 12): занятость noch для ночи + доля баров дня внутри (zanyatost).
Строка t — интервал [close t, close следующего торгового дня). noch[t] — доля слота тикера,
занятая в этой ночи. Чистые функции, никуда не подключены (подключение — К-Ф2).
"""
import numpy as np

REZHIMY = ("t0", "t1", "t0d")
OSNOVNOY = "t0d"          # Ж-111 п. 12: основной режим новых сеток (не якоря): ночь по занятости на 18:50, день — по барам
SESSIYA = 530 / 1440      # доля календарных суток в сессии 10:00–18:50 (530 мин); остальное интервала строки — «ночь»


def _check(rezhim):
    if rezhim not in REZHIMY:
        raise ValueError(f"rezhim должен быть из {REZHIMY}, получено {rezhim!r}")


def noch_iz_sdelok(trades, w_tr: dict, n: int, rezhim) -> np.ndarray:
    """noch0[t] = Σ w·1[entry_day ≤ t < exit_day]; T+1 добавляет w в строке exit_day, min(1, ·).

    Внутридневная сделка (entry_day = exit_day) не занимает ни одной ночи при T+0.
    exit_day None — открыта на конце окна: занята до строки n−1 включительно.
    Вес сделки — w_tr[tr.d_sig]; отсутствие веса — ошибка (KeyError), не молчаливая единица.
    """
    _check(rezhim)
    noch = np.zeros(n)
    extra = np.zeros(n)
    for tr in trades:
        if getattr(tr, "parent", None) is not None:     # Ж-092: частичный выход ½ — доля родителя, занятость не растёт
            continue
        w = float(w_tr[tr.d_sig])
        a = max(0, int(tr.entry_day))
        x = n if tr.exit_day is None else min(n, int(tr.exit_day))
        if a < x:
            noch[a:x] += w
        if rezhim == "t1" and tr.exit_day is not None and 0 <= tr.exit_day < n:
            extra[int(tr.exit_day)] += w
    if rezhim == "t1":
        return np.minimum(1.0, noch + extra)
    return noch


def noch_iz_pos(pos, first, rezhim) -> np.ndarray:
    """По позициям A (дробная pos): noch0 = |pos|; T+1: min(1, |pos| + zakryto).

    zakryto[t] = |pos[t−1]| при смене знака, иначе max(0, |pos[t−1]| − |pos[t]|); на первой строке пары — 0.
    """
    _check(rezhim)
    pos = np.asarray(pos, dtype=float)
    first = np.asarray(first, dtype=bool)
    ap = np.abs(pos)
    if rezhim in ("t0", "t0d"):
        return ap
    prev = np.r_[0.0, pos[:-1]]
    aprev = np.abs(prev)
    flip = np.sign(pos) != np.sign(prev)
    zak = np.where(flip, aprev, np.maximum(0.0, aprev - ap))
    zak[first] = 0.0
    return np.minimum(1.0, ap + zak)


def zanyatost(noch, expo, cal, rezhim) -> np.ndarray:
    """Занятость u для ставки строки (1 − u)·rf·cal. t0/t1 — noch (18:50), как раньше.

    t0d (Ж-111 п. 12): из cal календарных суток интервала [close t, close t+1) SESSIYA приходится на сессию дня t+1 —
    там занятость по 5-минутным барам (expo), остальное — ночь/выходные с занятостью на 18:50 (noch):
    u = (noch·(cal − SESSIYA) + expo·SESSIYA) / cal. Внутридневная сделка ночь не занимает, но день занимает пропорционально барам.
    """
    noch, expo, cal = (np.asarray(x, dtype=float) for x in (noch, expo, cal))
    if rezhim != "t0d":
        return noch
    for name, x in (("noch", noch), ("expo", expo)):
        if not np.all((x >= -1e-9) & (x <= 1.0 + 1e-9)):      # u > 1 дал бы отрицательную ставку на свободный капитал
            raise ValueError(f"t0d: {name} должна быть в [0; 1], получено min={np.nanmin(x)}, max={np.nanmax(x)}")
    s = np.minimum(SESSIYA, cal)
    return np.where(cal > 0, (noch * (cal - s) + expo * s) / np.where(cal > 0, cal, 1.0), noch)
