"""Шаг 4, 4.D: проверка сдвига σ и связка с главой 3 (корреляции σ, расхождения решений фильтра, QLIKE открыт/закрыт).
Чистые функции над матрицами [тикер, бар]; окно разработки 2018–2021."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fastbt.shag4 import NAN, POMETKA, YEARS


def shifted(M: np.ndarray, k: int) -> np.ndarray:
    """out[:, d] = M[:, d + k] (NaN, где нет)."""
    out = np.full_like(M, np.nan, dtype=float)
    if k > 0:
        out[:, :-k] = M[:, k:]
    elif k < 0:
        out[:, -k:] = M[:, :k]
    else:
        out[:] = M
    return out


def shift_profile(S: np.ndarray, RV: np.ndarray, valid: np.ndarray, ks=(-2, -1, 0, 1), diff=False) -> dict:
    """Медиана по тикерам корреляции ln σ² (или её приращения — для оконных) с ln RV_{d+k}, k = −2…+1 (§0.1)."""
    ls, lr = np.log(np.maximum(S, 1e-12) ** 2), np.log(np.maximum(RV, 1e-12))
    out = {}
    for k in ks:
        cs = []
        for ti in range(S.shape[0]):
            a, b = ls[ti], shifted(lr, k)[ti]
            if diff:
                a, b = np.diff(a, prepend=np.nan), np.diff(b, prepend=np.nan)
            m = valid[ti] & np.isfinite(a) & np.isfinite(b) & np.isfinite(S[ti]) & np.isfinite(RV[ti])
            if m.sum() > 2:
                cs.append(np.corrcoef(a[m], b[m])[0, 1])
        out[k] = float(np.median(cs)) if cs else NAN
    return out


def check_shift(prof: dict) -> None:
    """Максимум профиля — на k = 0 (последний известный день); иначе σ сдвинута, в расчёт не допускается."""
    k = max(prof, key=lambda x: prof[x])
    assert k == 0, f"сдвиг σ: максимум корреляции на k={k}: {prof}"


def cut_target(d, h: int, last):
    """Цель строки t смотрит на t+1…t+h торговых дней. Допустима, только если день t+h (по календарю дат d) ≤ last;
    иначе NaN — значение обнуляется сразу при чтении. d: [date, ticker, sigma], уже date ≤ last."""
    d = d[d["date"] <= last].copy()
    cal = np.sort(d["date"].unique())
    pos = np.searchsorted(cal, d["date"].values)
    d.loc[pos + h > len(cal) - 1, "sigma"] = np.nan
    return d


def dev_mask(years: np.ndarray) -> np.ndarray:
    return (years >= YEARS[0]) & (years <= YEARS[-1])


def corr_sigma(S: dict, valid: np.ndarray, tickers) -> list:
    """Корреляция σ-рядов попарно по тикерам (обе определены, окно разработки); + медиана по тикерам."""
    rows, names = [], list(S)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cs = []
            for ti, t in enumerate(tickers):
                m = valid[ti] & np.isfinite(S[a][ti]) & np.isfinite(S[b][ti])
                c = float(np.corrcoef(S[a][ti][m], S[b][ti][m])[0, 1]) if m.sum() > 2 else NAN
                cs.append(c)
                rows.append(dict(pometka=POMETKA, rezhim="корреляция σ", para=f"{a} – {b}", tiker=t, znachenie=c))
            rows.append(dict(pometka=POMETKA, rezhim="корреляция σ", para=f"{a} – {b}", tiker="медиана",
                             znachenie=float(np.nanmedian(cs))))
    return rows


def open_mask(G: np.ndarray) -> np.ndarray:
    """Фильтр открыт в тикер-день: средний затвор по трём категориям (тренд / контртренд / диапазон) > 0,5.
    Оговорка: категории равновесны; других определений «открыт» у C-затвора нет (затвор — по категории стратегии)."""
    return G.mean(axis=0) > 0.5


def share_diff(Ga: np.ndarray, Gb: np.ndarray, valid: np.ndarray) -> float:
    """Доля тикер-дней, где решения фильтра (вектор из 3 категорий) двух σ различаются."""
    d = np.any(Ga != Gb, axis=0)
    return float(d[valid].mean()) if valid.any() else NAN


def qlike_ch3():
    """QLIKE главы 3 — scripts/analysis/d26_lib.qlike (mean(y/f − ln(y/f) − 1), f ограничена снизу 1e-12)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
    from d26_lib import qlike
    return qlike


def qlike_split(y, f_harj, f_hyb, open_, valid, qlike=None) -> list:
    """QLIKE гибрида и HAR-J (б) в днях «фильтр открыт» / «закрыт»; выигрыш = QLIKE(HAR-J) − QLIKE(гибрид) > 0 —
    гибрид лучше. y — цель (дисперсия), f — прогноз дисперсии (σ²). Окно — valid (2018–2021)."""
    ql = qlike or qlike_ch3()
    fin = valid & np.isfinite(f_harj) & np.isfinite(f_hyb)
    ok = fin & np.isfinite(y) & (y > 1e-12)
    nan_y = fin & np.isnan(y)                         # цель с окном за 2021 (cut_target) — из QLIKE исключена
    rows = []
    for lab, m in (("открыт", ok & open_), ("закрыт", ok & ~open_), ("все", ok)):
        n = int(m.sum())
        n_nan = int((nan_y & (open_ if lab == "открыт" else ~open_ if lab == "закрыт" else True)).sum())
        qh, qy = (ql(y[m], f_harj[m]), ql(y[m], f_hyb[m])) if n else (NAN, NAN)
        rows.append(dict(pometka=POMETKA, rezhim="QLIKE", filtr=lab, n=n, n_nan_celi=n_nan, qlike_harj=qh, qlike_gibrid=qy,
                         vyigrysh=qh - qy, otn_vyigrysh=(qh - qy) / qh if n and qh else NAN))
    return rows
