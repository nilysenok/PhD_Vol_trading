"""Z2, события и контроль (план Z2 раздел 2.1; только БАРЫ, цели RV нет). Событие N_{d,k} = 1[бар присутствует ∧ |r_{d,k}| > q_d], q_d = Q_{0,99}{|r| баров сессий СТРОГО < d} (порог e_d: `de_e1_state.states`);
метка m_{d,k} = ln(|r|/q_d) > 0. Дни с историей < 20 сессий: порога нет, событий нет. Время: бар k = (10k, 10k + 10] минут от 10:00; событие — в конце бара. Hc: внутри дня (N, m) переставлены по присутствующим слотам (default_rng(0), дни по порядку)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

QS, Q_MAIN, MIN_PAST, NS, DELTA, T_CLOSE = (0.98, 0.99, 0.995), 0.99, 20, 53, 10.0, 530.0


def past_end(p: int) -> int:
    """Бары, учтённые в пороге дня на позиции p склада: СТРОГО до p (мутация: p + 1 — день d входит в порог)."""
    return p


def thresholds(absr: np.ndarray, qs=QS) -> np.ndarray:
    """(n_дней, len(qs)); NaN у дней с историей < MIN_PAST сессий."""
    out = np.full((len(absr), len(qs)), np.nan)
    for j in range(MIN_PAST, len(absr)):
        past = absr[: past_end(j)]
        past = past[np.isfinite(past)]
        if len(past):
            out[j] = np.quantile(past, qs)
    return out


@lru_cache(maxsize=1)
def real_thresholds() -> tuple:
    from models_v2.s15 import de0_data as D
    st = D.load_store()
    return st["days"], np.asarray(st["r"], float), thresholds(np.abs(st["r"]))


def events(r: np.ndarray, q: np.ndarray) -> tuple:
    """r (n, 53) (NaN = бара нет), q (n,) -> (N bool, m float): N — событие, m = ln(|r|/q) на событиях, иначе 0."""
    a = np.abs(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        N = np.isfinite(a) & (a > q[:, None])
        m = np.where(N, np.log(np.where(N, a, 1.0) / np.where(np.isfinite(q), q, 1.0)[:, None]), 0.0)
    return N, m


def permute_events(N: np.ndarray, m: np.ndarray, pres: np.ndarray, seed: int = 0) -> tuple:
    """Hc: внутри каждого дня (по порядку) набор (N, m) переставлен по присутствующим слотам одной перестановкой π_d."""
    rng, N2, m2 = np.random.default_rng(seed), np.zeros_like(N), np.zeros_like(m)
    for j in range(len(N)):
        ix = np.flatnonzero(pres[j])
        if len(ix):
            pi = rng.permutation(len(ix))
            N2[j, ix], m2[j, ix] = N[j, ix][pi], m[j, ix][pi]
    return N2, m2


def check_control(N, m, N2, m2, pres) -> None:
    """Инварианты Hc: N_d, множество меток, e_d сохранены; события только на присутствующих слотах; перестановка не тождественна (иначе Hc ≡ H1)."""
    if not (np.array_equal(N.sum(1), N2.sum(1)) and np.array_equal(np.sort(m, 1), np.sort(m2, 1)) and not (N2 & ~pres).any()):
        raise SystemExit("Hc: перестановка меняет N_d, набор меток или ставит событие на отсутствующий бар")
    if N.any() and np.array_equal(N, N2):
        raise SystemExit("Hc ≡ H1: перестановка тождественна")


def last_tau(N: np.ndarray) -> np.ndarray:
    """τ^last_d — минуты от последнего события дня до закрытия (T = 530; событие — конец бара k: 10(k + 1)); NaN — событий нет."""
    k = np.where(N.any(1), NS - 1 - np.argmax(N[:, ::-1], 1), -1)
    return np.where(k >= 0, T_CLOSE - DELTA * (k + 1), np.nan)


def delta_next(days) -> np.ndarray:
    """Δ дня j = (дней до следующей сессии склада) − 1; последний день — 0."""
    g = np.diff(days.values).astype("timedelta64[D]").astype(int) - 1
    return np.r_[g, 0].astype(float)
