"""Y-A, признаки дня d (PIT: только бары дня d склада M10 V7 `store_raw.npz`; бары t⁺ и позже не читаются). Карта слотов — паспорт V7 (`v7_0_seq`): слот = (старт бара − 10:00)/10 минут, 0…52 (53 бара; 52 — старт 18:40).
Часть закрытия — бары со стартом 17:40 … 18:30 (слоты 46…51; план раздел 8). s_d = ln(RV^last_d·n_d/n^last_d) − ln RV_d; n_d, n^last_d — числа присутствующих баров дня и части закрытия; RV — Σ r² присутствующих баров;
n^last_d = 0 или RV^last_d = 0 → s_d = 0 (такие дни считаются). Δ_d = (t⁺ − d) в календарных днях − 1. e_d — колонка `s` x1.parquet (= `de_e1_state.states`). Файлов не пишет."""
from __future__ import annotations

import numpy as np
import pandas as pd

T_OPEN, SLOT_MIN = 600, 10        # паспорт V7: T0 = 600 минут (10:00), шаг 10 минут
CLOSE_FROM, CLOSE_TO = "17:40", "18:30"


def slot_of(hhmm: str) -> int:
    """Номер слота склада V7 по времени старта бара (паспорт: (старт − 10:00)/10 минут)."""
    h, m = (int(x) for x in hhmm.split(":"))
    return (h * 60 + m - T_OPEN) // SLOT_MIN


CLOSE_SLOTS = np.arange(slot_of(CLOSE_FROM), slot_of(CLOSE_TO) + 1)


def feat_row(i: int) -> int:
    """Строка склада, из которой берутся бары признака дня на позиции i: СТРОГО день d (мутация: i + 1 — бары дня t⁺)."""
    return i


def e_state(x1: pd.DataFrame) -> np.ndarray:
    """e_d — экстремальный M10-бар дня d: колонка `s` файла x1.parquet (мутация: любой другой ряд)."""
    return x1["s"].to_numpy(int)


def delta_days(d, tp) -> np.ndarray:
    """Δ_d = (t⁺ − d) в календарных днях − 1: 0 — следующий календарный день недели, 2 — выходные (мутация: не по календарю t⁺)."""
    return ((pd.DatetimeIndex(tp) - pd.DatetimeIndex(d)).days.to_numpy() - 1).astype(float)


def day_features(r2: np.ndarray, present: np.ndarray) -> tuple:
    """r2, present — (53,) одного дня -> (s_d, n_d, n_last, zero_flag). zero_flag: нет части закрытия (n_last = 0 или RV_last = 0), s_d = 0."""
    n = int(present.sum())
    x = np.where(present, r2, 0.0)
    nl = int(present[CLOSE_SLOTS].sum())
    rv, rl = float(x.sum()), float(x[CLOSE_SLOTS].sum())
    if nl == 0 or rl <= 0.0 or rv <= 0.0:
        return 0.0, n, nl, True
    return float(np.log(rl * n / nl) - np.log(rv)), n, nl, False


def features(store: dict, idx, tp) -> dict:
    """store: days, r2 (D, 53), present (D, 53). -> s, n, nlast, zero (дни s_d = 0 по правилу), delta, rv_d (RV дня d по складу)."""
    days = pd.DatetimeIndex(store["days"])
    pos = days.get_indexer(pd.DatetimeIndex(idx))
    if (pos < 0).any():
        raise SystemExit("день d вне склада M10")
    out = {k: [] for k in ("s", "n", "nlast", "zero", "rv_d")}
    for i, p in enumerate(pos):
        row = feat_row(int(p))
        s, n, nl, z = day_features(store["r2"][row], store["present"][row])
        for k, v in zip(("s", "n", "nlast", "zero"), (s, n, nl, z)):
            out[k].append(v)
        out["rv_d"].append(float(np.where(store["present"][row], store["r2"][row], 0.0).sum()))
    res = {k: np.asarray(v) for k, v in out.items()}
    res["delta"] = delta_days(idx, tp)
    return res


def load_store() -> dict:
    """Склад M10 V7 (только чтение): days, r2 (канал r2, ×100² к r), present."""
    from models_v2.s15 import de0_data as D
    b = np.load(D.OUT7 / "store_raw.npz", allow_pickle=True)
    ch = [str(c) for c in b["channels"]]
    return dict(days=pd.DatetimeIndex(b["days"]), r2=np.asarray(b["raw"][:, :, ch.index("r2")], float), present=np.asarray(b["present"], bool))
