"""W3 D-π (Ж-250 п. 3–4): диагностика слоя доверия X1 по СОХРАНЁННЫМ рядам. Обучения моделей нет; только онлайн-веса над M и C.

Объявлено ДО счёта (11.11.2026, архитектор):
- окно M: x1 архив (harj_deep/E1) + продолжение (fwd/models/new), t⁺ 2018-01-04…2026-11-10, n = 2198;
- периоды: год t⁺ 2018…2025, 2026a (t⁺ ≤ 2026-02-27), 2026F; состояния — колонка s файла x1 (e_d: 0 = N, 1 = E);
- потолок «лучший постоянный π» — по клетке (период × состояние), сетка 0…1 шаг 0,05, ИСПОЛЬЗУЕТ ИСХОД; основной знаменатель разрыва;
- две основные PIT-схемы вердикта (по состоянию, как X1, те же эксперты M и C):
  P1 = DFTL126: π_d = argmin_g Σ_j λ^(возраст) · QLIKE(combine(M_j, C_j, π_g)) по закрытым парам состояния (t⁺_j ≤ d), λ = 0,5^(1/126);
  P2 = AH: AdaHedge (de Rooij и др., 2014) над 21 экспертом π_g с fixed share α = 0,01; π_d = Σ w_g π_g;
- чувствительность (вне вердикта): DFTL63, DFTL252, DFTLinf, HEDGE0 (проход X1 с α = 0);
- вердикт: гипотеза «слабое место — скорость доверия» подтверждена, если max(захват P1, захват P2) на окне M ≥ 1/3,
  захват = (Q_X1 − Q_P) / (Q_X1 − Q_потолок).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp

ROOT = Path(__file__).resolve().parents[2]
SRC = {"x1_a": "data/regen/harj_deep/E1/x1.parquet", "x1_n": "data/fwd/models/new/x1.parquet",
       "k23_a": "data/fwd/models/k23.parquet", "k23_n": "data/fwd/models/new/k23.parquet",
       "k1_a": "data/fwd/models/k1.parquet", "k1_n": "data/fwd/models/new/k1.parquet"}
ARCH_END = pd.Timestamp("2026-02-27")
GRID = np.round(np.arange(0, 21) * 0.05, 2)
EPS = 1e-12
MAIN_SCHEMES = ("DFTL126", "AH")
SENS_SCHEMES = ("DFTL63", "DFTL252", "DFTLinf", "HEDGE0")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def qlike(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def combine(M, C, pi):
    """Как `v4_ntp_state.combine`: F = M^(1−π)·C^π, концы побитно."""
    M, C, pi = (np.asarray(a, float) for a in (M, C, pi))
    mid = np.exp((1.0 - pi) * np.log(M) + pi * np.log(C))
    return np.where(pi >= 1.0, C, np.where(pi <= 0.0, M, mid))


def period(tp: pd.Series) -> np.ndarray:
    y = tp.dt.year.astype(str).to_numpy(object)
    y[(tp.dt.year == 2026).to_numpy() & (tp <= ARCH_END).to_numpy()] = "2026a"
    y[(tp.dt.year == 2026).to_numpy() & (tp > ARCH_END).to_numpy()] = "2026F"
    return y


def _rd(rel, ren=None):
    df = pd.read_parquet(ROOT / rel)
    if ren:
        df = df.rename(columns=ren)
    df["d"], df["t_plus"] = pd.to_datetime(df["d"]), pd.to_datetime(df["t_plus"])
    return df


def load() -> pd.DataFrame:
    """Окно M: одна строка на пару (d, t⁺). Стоп при дубле d/t⁺, расхождении RV или M ≠ √(K2·K3)."""
    x = pd.concat([_rd(SRC["x1_a"]), _rd(SRC["x1_n"])], ignore_index=True).sort_values("d").reset_index(drop=True)
    if x["d"].duplicated().any() or x["t_plus"].duplicated().any():
        raise SystemExit("x1: дубль d или t⁺")
    ka, kn = _rd(SRC["k23_a"], {"date": "d"}), _rd(SRC["k23_n"], {"date": "d"})
    k = pd.concat([ka[ka["t_plus"] <= ARCH_END], kn], ignore_index=True)
    ha, hn = _rd(SRC["k1_a"], {"target_date": "t_plus"}), _rd(SRC["k1_n"], {"target_date": "t_plus"})
    h = pd.concat([ha[ha["t_plus"] <= ARCH_END], hn], ignore_index=True)
    if k["d"].duplicated().any() or h["d"].duplicated().any():
        raise SystemExit("K: дубль d")
    x = x.merge(k[["d", "t_plus", "rv", "sigma2_k2", "sigma2_k3"]].rename(columns={"t_plus": "tp_k", "rv": "rv_k", "sigma2_k2": "K2", "sigma2_k3": "K3"}), on="d", how="left")
    x = x.merge(h[["d", "t_plus", "rv", "sigma2"]].rename(columns={"t_plus": "tp_h", "rv": "rv_h", "sigma2": "K1"}), on="d", how="left")
    for c in ("K1", "K2", "K3"):
        if x[c].isna().any():
            raise SystemExit("нет %s на %d датах" % (c, x[c].isna().sum()))
    if not ((x["tp_k"] == x["t_plus"]).all() and (x["tp_h"] == x["t_plus"]).all()):
        raise SystemExit("t⁺ расходится")
    for c in ("rv_k", "rv_h"):
        if not np.allclose(x[c], x["rv"], rtol=1e-12, atol=0):
            raise SystemExit("RV расходится (%s)" % c)
    if not np.allclose(np.sqrt(x["K2"] * x["K3"]), x["M"], rtol=1e-10, atol=0):
        raise SystemExit("M ≠ √(K2·K3)")
    x["per"] = period(x["t_plus"])
    return x.drop(columns=["tp_k", "tp_h", "rv_k", "rv_h"])


# ---------- онлайн-проходы: прогноз дня i — только пары с t⁺_j ≤ d_i ----------

def closed_date(tp, d):
    """Дата доступности пары: t⁺ (мутация «закрыто + 1» — подменить на d пары)."""
    return tp


def _arrivals(d, tp):
    """-> список: для дня i — индексы пар, закрытых к d_i впервые (по avail)."""
    avail = np.asarray(closed_date(np.asarray(tp), np.asarray(d)))
    order, out, ptr, n = np.argsort(avail, kind="stable"), [], 0, len(d)
    for i in range(n):
        new = []
        while ptr < n and avail[order[ptr]] <= d[i]:
            new.append(int(order[ptr]))
            ptr += 1
        out.append(new)
    return out


def fs_pass(qc, qm, s, d, tp, alpha_day):
    """Проход X1 (`de_e1_state._pass`) с заданным α по дням: alpha_day (n, 2). -> (π по дням для своего состояния, η по дням, π обоих состояний (n, 2))."""
    import bisect
    n, arr, lg, pool, eta = len(qc), _arrivals(d, tp), [0.0, 0.0], [], np.nan
    pis, etas = np.empty((n, 2)), np.full(n, np.nan)
    for i in range(n):
        for j in arr[i]:
            bisect.insort(pool, qc[j])
            bisect.insort(pool, qm[j])
        if pool:
            eta = 1.0 / max(0.5 * (pool[len(pool) // 2] + pool[(len(pool) - 1) // 2]), EPS)
        for j in arr[i]:
            k, a = int(s[j]), alpha_day[i][int(s[j])]
            lg[k] -= (eta if np.isfinite(eta) else 0.0) * (qc[j] - qm[j])
            if a > 0:
                lg[k] = float(logit((1.0 - a) * expit(lg[k]) + a / 2.0))
        pis[i], etas[i] = expit(lg), eta
    return pis[np.arange(n), np.asarray(s, int)], etas, pis


def grid_losses(rv, M, C) -> np.ndarray:
    """L[i, g] = QLIKE(rv_i, M_i^(1−π_g) C_i^π_g)."""
    return np.column_stack([qlike(rv, combine(M, C, np.full(len(rv), g))) for g in GRID])


def dftl(L, s, d, tp, hl):
    """Дисконтированный «следуй за лидером» по сетке π, раздельно по состояниям; нет закрытых пар состояния — π = 0,5."""
    n, arr = len(s), _arrivals(d, tp)
    lam = 1.0 if not np.isfinite(hl) else 0.5 ** (1.0 / hl)
    cum, cnt, pi = np.zeros((2, len(GRID))), np.zeros(2), np.empty(n)
    for i in range(n):
        for j in arr[i]:
            k = int(s[j])
            cum[k] = lam * cum[k] + L[j]
            cnt[k] += 1
        k = int(s[i])
        pi[i] = 0.5 if cnt[k] == 0 else GRID[int(np.argmin(cum[k]))]
    return pi


def adahedge_fs(L, s, d, tp, alpha=0.01):
    """AdaHedge над экспертами π_g (η = ln K / Δ, Δ — накопленный разрыв смешиваемости), fixed share α к равным весам; раздельно по состояниям.
    Прогноз π = Σ w_g π_g (QLIKE выпукла по ln F, ln F линеен по π — средний π не хуже среднего потерь)."""
    n, K, arr = len(s), len(GRID), _arrivals(d, tp)
    lw, gap, pi = np.full((2, K), -np.log(K)), np.zeros(2), np.empty(n)
    for i in range(n):
        for j in arr[i]:
            k, l = int(s[j]), L[j]
            w = np.exp(lw[k] - logsumexp(lw[k]))
            eta = np.log(K) / gap[k] if gap[k] > 0 else 1e8
            h = float(w @ l)
            m = float(-(logsumexp(lw[k] - eta * l) - logsumexp(lw[k])) / eta)
            gap[k] += max(h - m, 0.0)
            lw[k] = lw[k] - eta * l
            lw[k] -= logsumexp(lw[k])
            w = (1.0 - alpha) * np.exp(lw[k]) + alpha / K
            lw[k] = np.log(w)
        w = np.exp(lw[int(s[i])] - logsumexp(lw[int(s[i])]))
        pi[i] = float(w @ GRID)
    return pi


def scheme(name, x, L):
    s, d, tp = x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy()
    if name.startswith("DFTL"):
        hl = np.inf if name.endswith("inf") else float(name[4:])
        return dftl(L, s, d, tp, hl)
    if name == "AH":
        return adahedge_fs(L, s, d, tp, 0.01)
    if name == "HEDGE0":
        qc, qm = qlike(x["rv"], x["C"]), qlike(x["rv"], x["M"])
        return fs_pass(qc, qm, s, d, tp, np.zeros((len(s), 2)))[0]
    raise KeyError(name)


def fs_fixed_point(D, eta, a):
    """Неподвижная точка fixed share при постоянной разнице потерь D: π = (1−α)·σ(logit π − η·D) + α/2."""
    p = 0.5
    for _ in range(20000):
        p = (1.0 - a) * expit(logit(min(max(p, 1e-15), 1 - 1e-15)) - eta * D) + a / 2.0
    return p


def fs_after(D, eta, a, steps=252, p0=0.5):
    """π после steps шагов fixed share на постоянной D из p0."""
    p = p0
    for _ in range(steps):
        p = (1.0 - a) * expit(logit(min(max(p, 1e-15), 1 - 1e-15)) - eta * D) + a / 2.0
    return p
