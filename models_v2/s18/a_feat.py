"""S-A-IMPL-1 (Ж-254; `docs/otchety/harj_struct/S_A_SPEC.md`, разд. 1–2): признаки памяти S[k,t] для вариантов S-A0/A1/A2/A2p. float64, torch-тензоры (градиент для a_model).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Файл читает бары SBER только через `models_v2.s17.fwd_src.Src` (обрезка по cut, 18:40); модели и ряды K1–K3 не трогает.
Публичное API:
- `NS = 53`, `KS = ("s","m","l")`, `LAGS = (5, 22, 66)` (носители L_k), `RECT = (1, 5, 22)` (длина прямоугольника HAR), `TAU = arange(53)/52` (τ слота: (минута начала − 600)/520), `PERM` = default_rng(0).permutation(53) (A2p).
- `spline_basis(tau) -> (n, 4)`: натуральный кубический сплайн, узлы 0; 0,25; 0,5; 0,75; 1, без константы (N2…N5), центрирован по сетке TAU (константа кода); `slot_tau(perm=False)` — TAU или TAU[PERM] (τ_{π(j)}).
- `Feat` — дневной ряд и бары: `dates`, `seg`, `pos` (позиция строки в сегменте), `r2 (T, 53)` (r² бара, NaN — бара нет), `pres`, `n` (баров в дне), `rv`, `rv_w`, `rv_m`, `j` (из `daily.build_daily`), длинная форма `lrow/lslot/lr2`.
  `Feat.from_r2(r2, dates)` — синтетика (один сегмент, J = 0). `build(bars, index_days) -> Feat`; `build_sber(cut=LAST_SESSION, src=None) -> Feat` (бары float64 через `Src.bars`, дни IMOEX через `Src.index_days`).
- `load_k1() -> DataFrame`: K1 из `data/fwd/models/{k1,new/k1}.parquet` (2452 строки, d, target_date, rv, sigma2, s2, b0, b_d, b_w, b_m, b_J, cross_seg).
- `har_theta()` -> dict тензоров: `omega (3,)`, `psi (3, 2)`, `c (3, 4)`; нули = HAR точно (a = HAR, b ≡ 1).
- `slot_exp(c, tau) -> (3, 53)`: exp(h_k(τ_j)); `rv_b(F, E) -> (T, 3)` (torch) и `rv_b_exact(F, E) -> (T, 3)` (numpy, тот же путь суммирования, что в `daily.build_daily`; при E = 1 побитно равно RV).
- `lag_idx(F, rows) -> [(ix, ok)] ×3`: индексы дней t − l и маска l < min(L_k, pos + 1) (усечение по сегменту, разд. 2.1); `lag_weights(ok, k, omega_k, psi_k) -> (n, L_k)`: a = (1 − ω)R + ωG, оба нормированы по допустимым лагам; a ≥ 0, Σa = 1.
- `ln_S(F, theta, lag, tau=TAU) -> (n, 3)`: ln S[k, t] = ln Σ_l a[k,l]·RV_b[k](t − l) на строках `rows`, по которым построен `lag`.
Запуск: `python3 -W ignore -m models_v2.s18.a_feat [--check | --check-k1 | --check-pit]` (только чтение; код проверок — `a_feat_chk.py`)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from models_v2 import daily as D
from models_v2.s17.fwd_src import LAST_SESSION, Src

NS, KS, LAGS, RECT = 53, ("s", "m", "l"), (5, 22, 66), (1, 5, 22)
TAU = np.arange(NS) / 52.0
PERM = np.random.default_rng(0).permutation(NS)
KNOTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SCALE = 1e4


def spline_basis(tau) -> np.ndarray:
    x, xi, K = np.asarray(tau, float), KNOTS, len(KNOTS)

    def d(k):
        return (np.maximum(x - xi[k], 0.0) ** 3 - np.maximum(x - xi[K - 1], 0.0) ** 3) / (xi[K - 1] - xi[k])
    return np.column_stack([x] + [d(k) - d(K - 2) for k in range(K - 2)]) - _RAW_MEAN


_RAW_MEAN = 0.0
_RAW_MEAN = spline_basis(TAU).mean(axis=0)


def slot_tau(perm: bool = False) -> np.ndarray:
    return TAU[PERM] if perm else TAU


class Feat:
    def __init__(self, dates, seg, r2, lrow, lslot, lr2, rv, rv_w, rv_m, j):
        self.dates, self.seg, self.r2, self.lrow, self.lslot, self.lr2 = dates, np.asarray(seg), r2, lrow, lslot, lr2
        self.rv, self.rv_w, self.rv_m, self.j = (np.asarray(a, float) for a in (rv, rv_w, rv_m, j))
        self.pres = np.isfinite(r2)
        self.n = self.pres.sum(axis=1)
        self.pos = pd.Series(self.seg).groupby(self.seg).cumcount().to_numpy()
        self._t = None

    @classmethod
    def from_r2(cls, r2: np.ndarray, dates) -> "Feat":
        """Синтетика: r2 (T, 53) в долях (RV = 1e4·Σ r²), один сегмент, J = 0; rv_w, rv_m — как `daily.add_windows`."""
        r2, dates = np.asarray(r2, float), pd.DatetimeIndex(dates)
        rv = np.nansum(r2, axis=1) * SCALE
        d = D.add_windows(pd.DataFrame({"rv": rv}, index=dates))
        row, slot = np.nonzero(np.isfinite(r2))
        return cls(dates, np.zeros(len(dates), int), r2, row, slot, r2[row, slot], rv, d["rv_w"], d["rv_m"], np.zeros(len(dates)))

    def tens(self):
        """(r2 с нулями вместо NaN, pres) как float64-тензоры."""
        if self._t is None:
            self._t = (torch.as_tensor(np.where(self.pres, self.r2, 0.0)), torch.as_tensor(self.pres.astype(float)), torch.as_tensor(self.n.astype(float)))
        return self._t


def build(bars: pd.DataFrame, index_days) -> Feat:
    d, _ = D.build_daily(bars, index_days=index_days)
    b = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = D.intraday_returns(b)
    mins = (b["begin"].dt.hour * 60 + b["begin"].dt.minute).to_numpy() - 600
    if (mins % 10).any() or mins.min() < 0 or mins.max() > 520:
        raise ValueError("бар вне сетки 10 минут основной сессии")
    row = d.index.get_indexer(r["date"])
    keep = row >= 0
    row, slot, r2 = row[keep], (mins // 10)[keep], (r["r"].to_numpy(float) ** 2)[keep]
    full = np.full((len(d), NS), np.nan)
    full[row, slot] = r2
    return Feat(d.index, d["seg"].to_numpy(), full, row, slot, r2, d["rv"], d["rv_w"], d["rv_m"], d["j"])


def build_sber(cut=LAST_SESSION, src: Src | None = None) -> Feat:
    src = src or Src()
    return build(src.bars("SBER", cut), src.index_days(cut))


def load_k1() -> pd.DataFrame:
    """K1 (архив + новые 180 строк, только чтение): d, target_date, rv, sigma2, s2, b0, b_d, b_w, b_m, b_J, cross_seg … (2452 строки)."""
    from models_v2.s17.fwd_src import ROOT
    k = pd.concat([pd.read_parquet(ROOT / p) for p in ("data/fwd/models/k1.parquet", "data/fwd/models/new/k1.parquet")], ignore_index=True)
    if k["d"].duplicated().any():
        raise SystemExit("K1: дубль d")
    return k.sort_values("d").reset_index(drop=True)


def har_theta() -> dict:
    z = torch.float64
    return dict(omega=torch.zeros(3, dtype=z), psi=torch.zeros(3, 2, dtype=z), c=torch.zeros(3, 4, dtype=z))


def slot_exp(c, tau=TAU):
    """E[k, j] = exp(h_k(τ_j)), h_k = Σ_m c[k, m]·N_m(τ). Нормировка по присутствующим барам — в `rv_b`."""
    B = torch.as_tensor(spline_basis(tau))
    return torch.exp(B @ torch.as_tensor(c).T).T


def rv_b(F: Feat, E) -> torch.Tensor:
    """RV_b[k](s) = 1e4·Σ_{j ∈ present} b_k(s, j) r²_{s,j}, b = E_j / mean_{present} E (mean b = 1 в каждой сессии)."""
    r2, pres, n = F.tens()
    E = torch.as_tensor(E)
    return SCALE * (r2 @ E.T) / ((pres @ E.T) / n[:, None])


def rv_b_exact(F: Feat, E) -> np.ndarray:
    """То же в numpy; суммирование — `pandas.groupby.sum` по тем же барам в том же порядке, что в `daily.build_daily` (при E = 1 побитно RV)."""
    E = np.asarray(E, float)
    mean_p = (F.pres.astype(float) @ E.T) / F.n[:, None]
    out = np.empty((len(F.dates), 3))
    for k in range(3):
        b = E[k][F.lslot] / mean_p[F.lrow, k]
        out[:, k] = pd.Series(b * F.lr2).groupby(F.lrow).sum().reindex(np.arange(len(F.dates))).to_numpy() * SCALE
    return out


def lag_idx(F: Feat, rows) -> list:
    rows, out = np.asarray(rows), []
    for L in LAGS:
        lg = np.arange(L)
        ok = lg[None, :] <= F.pos[rows][:, None]
        out.append((np.where(ok, rows[:, None] - lg[None, :], 0), ok))
    return out


def lag_weights(ok, k: int, omega, psi) -> torch.Tensor:
    L = LAGS[k]
    m = torch.as_tensor(ok, dtype=torch.float64)
    x = torch.arange(L, dtype=torch.float64) / (L - 1)
    R = m * (torch.arange(L) < RECT[k]).to(torch.float64)
    G = m * torch.exp(psi[0] * x + psi[1] * x * x)
    return (1.0 - omega) * R / R.sum(1, keepdim=True) + omega * G / G.sum(1, keepdim=True)


def ln_S(F: Feat, theta: dict, lag: list, tau=TAU) -> torch.Tensor:
    rb = rv_b(F, slot_exp(theta["c"], tau))
    cols = []
    for k, (ix, ok) in enumerate(lag):
        a = lag_weights(ok, k, theta["omega"][k], theta["psi"][k])
        cols.append(torch.log((a * rb[torch.as_tensor(ix), k]).sum(1)))
    return torch.stack(cols, dim=1)


if __name__ == "__main__":
    from models_v2.s18 import a_feat_chk
    sys.exit(a_feat_chk.main(sys.argv[1:]))
