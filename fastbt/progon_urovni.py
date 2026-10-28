"""Л-11 (Ж-078): уровни 2–3 прогона этапа 1 — A′ из рядов excess по парам (тикер, Sk), 54 × 24 = 1 296 конфигураций.

Вход — `komponenty`: {sid: float64 [тикеров, T]}, nan — у пары нет строки на дату (контракт §1.4).
Уровень 2: веса стратегий `ves` (ravnyj | obr_korr_W: a_k ∝ 1/(1+Σ_{j≠k} max ρ_kj)), дубли ρ > порога — одна компонента
с одним весом, лимит λ на совпадение сигналов (нужны `pozicii`), состав. Вес пары = вес её стратегии, пары нормируются
по присутствующим на дату. Все веса и ρ — по данным до d−1, пересмотр раз в месяц (PIT, Д2).
Уровень 3: `tsel_vol` (масштаб = min(1, цель/σ̂), без плеча), `tormoz_mdd` (при просадке накопленного excess до d−1
глубже порога масштаб ×0,5). Умолчания: ряд = среднее по присутствующим парам (тождество §1.4 -> IG-4: A′ = A).
Основная конфигурация — умолчания; остальные — чувствительность, без ранжирования и выбора (Ж-078).
"""
from __future__ import annotations
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

YAML = Path(__file__).parent / "designs" / "progon" / "urovni_2_3.yaml"
OUT = Path(__file__).resolve().parent.parent / "docs" / "tables" / "progon_e1" / "urovni_S.csv"


def setka(path=YAML) -> dict:
    return yaml.safe_load(Path(path).read_text("utf-8"))


def konfiguracii(sk: dict | None = None) -> list:
    """54 × 24 конфигурации в порядке сетки; основная (умолчания) помечена `osnovnaya`."""
    sk = sk or setka()
    keys = ["ves", "lam", "rho", "sostav", "tsel_vol", "okno_vol", "tormoz_mdd"]
    vals = [sk["uroven2"][k] for k in keys[:4]] + [sk["uroven3"][k] for k in keys[4:]]
    out = []
    for v in itertools.product(*vals):
        c = dict(zip(keys, v))
        c["osnovnaya"] = all(c[k] == sk["umolchaniya"][k] for k in keys)
        out.append(c)
    return out


def _chislo(x):
    return None if x == "net" else float(x)


def _starty(daty) -> list:
    """Индексы первого торгового дня каждого месяца: пересмотр весов (PIT)."""
    m = pd.DatetimeIndex(daty).to_period("M").asi8
    return [0] + [i for i in range(1, len(m)) if m[i] != m[i - 1]]


def _istoriya(x, i: int, w: int):
    """Окно строго до дня i (то есть до d−1)."""
    return x[..., max(0, i - w):i]


def _klastery(R: np.ndarray, porog) -> list:
    """Связные компоненты графа ρ > порога (дубли — одна компонента); порог None — каждая стратегия отдельно."""
    n = R.shape[0]
    par = list(range(n))

    def f(a):
        while par[a] != a:
            a = par[a]
        return a
    if porog is not None:
        for a, b in itertools.combinations(range(n), 2):
            if R[a, b] > porog:
                par[f(a)] = f(b)
    g = {}
    for a in range(n):
        g.setdefault(f(a), []).append(a)
    return list(g.values())


def _vektor(h: np.ndarray, ves: str, porog, act: list, mini: int) -> np.ndarray:
    """Веса стратегий [K] по истории h [K, дней]; Σ = 1 по act."""
    a = np.zeros(h.shape[0])
    if h.shape[1] < mini:
        a[act] = 1.0 / len(act)
        return a
    df = pd.DataFrame(h[act].T)
    R = df.corr(min_periods=mini).fillna(0.0).to_numpy()
    cl = _klastery(R, porog)
    Rc = pd.DataFrame({j: df[c].mean(axis=1) for j, c in enumerate(cl)}).corr(min_periods=mini).fillna(0.0).to_numpy()
    for j, c in enumerate(cl):
        ac = 1.0 if ves == "ravnyj" else 1.0 / (1.0 + np.maximum(np.delete(Rc[j], j), 0.0).sum())
        a[[act[k] for k in c]] = ac / len(c)
    return a / a.sum()


def _vesa_strategij(S: np.ndarray, daty, ves: str, porog, act: list, mini: int = 20) -> np.ndarray:
    """a [K, T]: веса стратегий на каждую дату, пересмотр раз в месяц по окну W до d−1."""
    W = 0 if ves == "ravnyj" else int(ves.split("_")[-1])
    st = _starty(daty) + [S.shape[1]]
    a = np.zeros(S.shape)
    for i, e in zip(st[:-1], st[1:]):
        h = _istoriya(S, i, W if W else 252)
        a[:, i:e] = _vektor(h, ves, porog, act, mini)[:, None]
    return a


def _limit(X, a, poz, lam):
    """Множитель [тикеров, T] ≤ 1: |Σ_k a_k·pos_k| ≤ λ·max a_k на тикеро-день; позиции известны на close d−1."""
    P = np.stack([np.nan_to_num(poz[s]) for s in poz])
    tot = np.abs((a[:, None, :] * P).sum(0))
    cap = lam * a.max(0)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > cap, cap / tot, 1.0)


def _prosadka(cum: np.ndarray, i: int) -> float:
    """Просадка накопленного excess по дням до d−1 (i — день d)."""
    return float(cum[:i].max() - cum[i - 1]) if i > 0 else 0.0


def _uroven3(A2: np.ndarray, tsel, okno: int, tormoz, mnozh: float, mini: int = 20) -> np.ndarray:
    out = np.full(len(A2), np.nan)
    cum = np.cumsum(np.nan_to_num(A2))                  # просадка A′ до торможения: путь без обратной связи
    for i in range(len(A2)):
        w = _istoriya(A2, i, okno)
        w = w[~np.isnan(w)]
        s = 1.0
        if tsel is not None and len(w) >= max(mini, 2) and w.std(ddof=1) > 0:
            s = min(1.0, tsel / 100.0 / (w.std(ddof=1) * np.sqrt(252)))
        if tormoz is not None and _prosadka(cum, i) > tormoz / 100.0:
            s *= mnozh
        out[i] = A2[i] * s
    return out


def a_shtrih(komponenty: dict, konf: dict, daty, pozicii: dict | None = None, proshli=None, sk: dict | None = None):
    """excess A′ [T] из рядов пар. `pozicii` ({sid: [тикеров, T]}, позиция, держимая в день d) — для λ;
    `proshli` — стратегии, прошедшие ворота (состав `proshedshie`)."""
    sk = sk or setka()
    mini, mnozh = sk["parametry"]["min_istoriya"], sk["parametry"]["tormoz_mnozhitel"]
    sids, proshli = list(komponenty), set(proshli or ())
    X = np.stack([np.asarray(komponenty[s], float) for s in sids])
    M = ~np.isnan(X)
    if konf["sostav"] == "proshedshie":
        if not proshli:
            raise ValueError("состав proshedshie: нужен список прошедших ворота")
        act = [k for k, s in enumerate(sids) if s in proshli]
    else:
        act = list(range(len(sids)))
    if not act:
        raise ValueError("пустой состав")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.nanmean(X, axis=1)
    a = _vesa_strategij(S, daty, konf["ves"], _chislo(konf["rho"]), act, mini)
    f = np.ones(X.shape[1:])
    if _chislo(konf["lam"]) is not None:
        if pozicii is None:
            raise ValueError("лимит λ: нужны позиции")
        f = _limit(X, a, {s: pozicii[s] for s in sids}, _chislo(konf["lam"]))
    num = (a * (np.nan_to_num(X) * f[None]).sum(1)).sum(0)
    den = (a * M.sum(1)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        A2 = np.where(den > 0, num / den, np.nan)
    return _uroven3(A2, _chislo(konf["tsel_vol"]), int(konf["okno_vol"]), _chislo(konf["tormoz_mdd"]), mnozh, mini)


def metriki(x: np.ndarray) -> dict:
    """Шарп (ddof=1, √252; Д7) и MDD накопленного excess; nan пропускаются."""
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    cum = np.cumsum(x)
    return dict(sharpe=float(x.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
                mdd=float((np.maximum.accumulate(cum) - cum).max()) if len(x) else float("nan"), n_dnej=len(x))


def tablica_1296(komponenty: dict, daty, pozicii=None, proshli=None, out=OUT) -> pd.DataFrame:
    """Таблица всех конфигураций в порядке сетки: без сортировки, ранга и выбора; основная помечена `osnovnaya`."""
    sk = setka()
    rows = []
    for c in konfiguracii(sk):
        m = metriki(a_shtrih(komponenty, c, daty, pozicii, proshli, sk))
        rows.append({**c, "status": "osnovnaya" if c["osnovnaya"] else "chuvstvitelnost", **m})
    df = pd.DataFrame(rows)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def pary_iz_dvizhka(eng, comm: float):
    """Ряды excess пар из строк движка s9 (float64): ex = pos·lr − стороны·comm − |pos|·rf·cal (контракт §1.4).
    Возвращает ({sid: [тикеров, T]}, {sid: позиции}, даты). Только чтение движка (IG-4: на данных IG-0)."""
    T, tick = len(eng.dates), {t: i for i, t in enumerate(eng.tickers)}
    d, q, pos = eng.r["date"], eng.r["pair"], eng.A_row
    dp = np.diff(pos, prepend=0.0)
    dp[eng.first] = pos[eng.first]
    sides = np.abs(dp) + np.where(np.r_[eng.first[1:], True], np.abs(pos), 0.0)
    ex = pos * eng.lr_row - sides * comm - np.abs(pos) * (eng.rf * eng.cal)[d]
    kom = {s: np.full((len(tick), T), np.nan) for _, s in eng.pairs}
    poz = {s: np.full((len(tick), T), np.nan) for s in kom}
    for k, (t, s) in enumerate(eng.pairs):
        m = q == k
        kom[s][tick[t], d[m]], poz[s][tick[t], d[m]] = ex[m], pos[m]
    return kom, poz, np.asarray(eng.dates.values, "datetime64[D]")
