"""Книга МСП (Ф4; спецификация `2026-10-28_msp_fazy_spec.md` §3.3, §11.2; Ж-099 п. 5, Ж-109, Ж-111 пп. 4, 14).

Вход на день d — скоры кандидатов (интерфейс Ф3 `reyting`) и ИСТОРИЯ рядов: функции берут строки строго до d
(`progon_urovni._istoriya`), день d в σ, β, SR, выключатель не входит. Порядок (§11.2: шаг 7а до нетто):
топ-K -> лимит сектора -> размер -> веса Sk -> нетто -> таргет и плечо -> хедж MIX -> выключатель ×0,5.
Слоты открытых ставок ведёт вызывающий (Ф3/Ф5): здесь каждый день набирается K заново из переданных кандидатов.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastbt.progon_urovni import _istoriya, _prosadka
from fastbt.tf_os_v5 import KOMISSIYA      # Д7: та же ставка, что в движке (0,04 % на сторону)

MAX_SEKTOR, NETTO, LAM_SR, OTKL_SR, OTKL_MNOZH = 3, 0.30, 0.5, -0.5, 0.25
OKNO_TSEL, OKNO_SK, OKNO_SR, OKNO_BETA, OKNO_VYKL = 21, 63, 252, 63, 10
VYKL_VOLA, VYKL_PROSADKA, VYKL_MNOZH, MARZHA_NADBAVKA = 2.0, 0.05, 0.5, 0.02


def _ann(x) -> float:
    """std(ddof=1)·√252 без nan; nan, если строк < 2."""
    x = x[~np.isnan(x)]
    return float(x.std(ddof=1) * np.sqrt(252)) if len(x) > 1 else np.nan


def vybor_stavok(score, sigma, sektor, k, rho, ks=1.0, m_score=None, max_sek=MAX_SEKTOR) -> np.ndarray:
    """Шаги 1–3: топ-K по |score|, не более max_sek на сектор; u = знак·ρ·m/(ks·σ_d). Ничья — порядок индексов."""
    score, sigma = np.asarray(score, float), np.asarray(sigma, float)
    m = np.ones_like(score) if m_score is None else np.asarray(m_score, float)
    u, zan = np.zeros(len(score)), {}
    ok = np.isfinite(score) & (score != 0) & np.isfinite(sigma) & (sigma > 0)
    for j in np.argsort(-np.where(ok, np.abs(score), -np.inf), kind="stable"):
        if not ok[j] or (u != 0).sum() >= k:
            break
        if zan.get(sektor[j], 0) >= max_sek:
            continue
        zan[sektor[j]] = zan.get(sektor[j], 0) + 1
        u[j] = np.sign(score[j]) * min(1.0, rho * m[j] / (ks * sigma[j]))   # §3.3 п. 7: не выше 100 % NAV
    return u


def sr_ryad(ex: np.ndarray, okno: int = OKNO_SR) -> np.ndarray:
    """[T, S]: SR_k на строках t−okno…t−1 (ddof=1, √252, ex уже над КС); nan, пока нет полного окна."""
    out = np.full(ex.shape, np.nan)
    for t in range(okno, len(ex)):
        w = ex[t - okno:t]
        sd = w.std(axis=0, ddof=1)
        out[t] = np.where(sd > 0, w.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), np.nan)
    return out


def otklyucheno(sr: np.ndarray, kvartal: np.ndarray, i: int, porog=OTKL_SR, podryad=2) -> np.ndarray:
    """Самоотключение на день i (bool [S]): SR<порога на `podryad` концах кварталов подряд -> с первого дня
    следующего квартала; снятие — в первый день, когда SR>0. Всё по sr[t≤i] (sr[t] — данные до t−1)."""
    S, off, ser = sr.shape[1], np.zeros(sr.shape[1], bool), np.zeros(sr.shape[1], int)
    for t in range(1, i + 1):
        if kvartal[t] != kvartal[t - 1]:                         # t — первый день квартала, t−1 — конец прошлого
            ser = np.where(sr[t - 1] < porog, ser + 1, 0)        # nan < порога = False
            off |= ser >= podryad
        off &= ~(sr[t] > 0)
    return off


def vesa_sk(ex_sk: np.ndarray, kvartal: np.ndarray, i: int, lam=LAM_SR, podryad=2) -> np.ndarray:
    """w_k ∝ (1/σ̂_k)(1+λ·clip(SR,−1,1)), Σ=1; затем отключённые ×0,25 (без перенормировки). Нет данных — поровну."""
    S = ex_sk.shape[1]
    sig = np.array([_ann(_istoriya(ex_sk[:, k], i, OKNO_SK)) for k in range(S)])
    sr = sr_ryad(ex_sk[:i + 1])                                  # строка i использует ex[i−252…i−1]
    slope = np.nan_to_num(np.clip(sr[i], -1, 1))
    ok = np.isfinite(sig) & (sig > 0)
    w = np.where(ok, (1 + lam * slope) / np.where(ok, sig, 1), 0.0)
    w = w / w.sum() if w.sum() > 0 else np.full(S, 1.0 / S)
    return w * np.where(otklyucheno(sr, kvartal[:i + 1], i, podryad=podryad), OTKL_MNOZH, 1.0)


def netto(u: np.ndarray, maks=NETTO) -> np.ndarray:
    """Шаг 8: |L − S| ≤ maks·(L + S); доминирующая сторона уменьшается пропорционально до равенства."""
    L, S = u[u > 0].sum(), -u[u < 0].sum()
    if abs(L - S) <= maks * (L + S) + 1e-15:
        return u.copy()
    big, small = (L, S) if L > S else (S, L)
    f = small * (1 + maks) / (1 - maks) / big
    return np.where((u > 0) == (L > S), u * f, u)


def mnozhitel_vola(ex_port: np.ndarray, i: int, valovaya: float, tsel=0.10, plecho=2.0) -> float:
    """Шаги 9–10: m = tsel/σ̂_p по строкам i−21…i−1 (ex_port — excess книги при m=1); <21 строки — 1; m·валовая ≤ плечо."""
    w = _istoriya(ex_port, i, OKNO_TSEL)
    sig = _ann(w)
    m = tsel / sig if len(w) >= OKNO_TSEL and sig > 0 else 1.0
    return min(m, plecho / valovaya) if valovaya > 0 else m


def marzha_v_den(valovaya: float, ks_god: float) -> float:
    """Плата за плечо: на часть валовой сверх 100 % — КС + 2 % годовых, в долях NAV за день."""
    return max(valovaya - 1.0, 0.0) * (ks_god + MARZHA_NADBAVKA) / 252


def beta(ex_kniga: np.ndarray, imoex: np.ndarray, i: int, okno=OKNO_BETA) -> float:
    """β книги к open→open IMOEX по строкам i−63…i−1; меньше окна — 0 (не хеджируем вслепую)."""
    y, x = _istoriya(ex_kniga, i, okno), _istoriya(imoex, i, okno)
    ok = ~(np.isnan(x) | np.isnan(y))
    if len(y) < okno or ok.sum() < okno or x[ok].var(ddof=1) <= 0:
        return 0.0
    return float(np.cov(y[ok], x[ok], ddof=1)[0, 1] / x[ok].var(ddof=1))


def hedzh_mix(b: float) -> float:
    """Доля NAV в MIX: продажа β·NAV (знак минус), у отрицательной беты — покупка."""
    return -b


def vyklyuchatel(ex_kniga: np.ndarray, i: int, tsel=0.10) -> float:
    """×0,5, если вола за 10 дней (до i−1) > 2·tsel или просадка накопленного excess > 5 % (до i−1)."""
    vola = _ann(_istoriya(ex_kniga, i, OKNO_VYKL))
    plohaya = (len(_istoriya(ex_kniga, i, OKNO_VYKL)) >= OKNO_VYKL and vola > VYKL_VOLA * tsel) or \
              _prosadka(np.cumsum(np.nan_to_num(ex_kniga)), i) > VYKL_PROSADKA
    return VYKL_MNOZH if plohaya else 1.0


def komissiya_mnozh(hedzh: float, hedzh_prev: float, stavka=KOMISSIYA) -> float:
    """Комиссия (доля NAV) за оборот, не оплаченный в excess пар: хедж MIX, ставка·|Δ хеджа|.
    Ставки-акции дневные: excess пары (`tf_os_v5.dohod`) закрыт к последнему бару дня и платит ставку на |pos| входа и
    выхода, а excess·w масштабируется весом линейно — смена множителей, весов Sk, нетто, лимитов между днями лишь меняет
    размер завтрашней внутридневной сделки, её комиссия уже в excess·w. Облагать |Δw| акций — двойной учёт.
    Хедж MIX (open→open IMOEX, `hedzh·imoex`) переносится через ночь и комиссии в excess не имеет."""
    return float(stavka * abs(hedzh - hedzh_prev))


@dataclass
class Itog:
    u: np.ndarray        # веса ставок в долях NAV (после всего, кроме хеджа)
    hedzh: float         # доля NAV в MIX (знак: минус — короткая)
    w_sk: np.ndarray
    m: float
    vykl: float
    komissiya: float = 0.0   # доля NAV за оборот множителей и хеджа в день i; вычитает вызывающий из доходности дня


def kniga_dnya(i, score, sigma, sektor, sk, ist: dict, k=20, rho=0.01, plecho=2.0, tsel=0.10) -> Itog:
    """ist: ex_sk [T,S], ex_port [T] (m=1), ex_kniga [T] (после шагов 8–10), imoex [T], kvartal [T] — строки ≥ i не читаются.
    ist["pred"] — состояние предыдущего вызова (пишет сама): вчерашний хедж для комиссии его оборота."""
    u = vybor_stavok(score, sigma, sektor, k, rho)
    w = vesa_sk(ist["ex_sk"], ist["kvartal"], i)
    u = netto(u * (len(w) * w)[np.asarray(sk)])
    m = mnozhitel_vola(ist["ex_port"], i, np.abs(u).sum(), tsel, plecho)
    f = vyklyuchatel(ist["ex_kniga"], i, tsel)
    b = beta(ist["ex_kniga"], ist["imoex"], i)
    wf, h = u * m * f, hedzh_mix(b) * f
    kom = komissiya_mnozh(h, ist.get("pred", {"h": 0.0})["h"])
    ist["pred"] = {"h": h}
    return Itog(wf, h, w, m, f, kom)
