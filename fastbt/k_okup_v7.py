"""Куб v7, ось k_okup (Ж-153): маска входа «PnL̂ ≥ k × издержки сделки», k ∈ {выкл., 2, 3}. Вне движка.

Оценка на день входа d — по теневым сделкам точки, ЗАКРЫТЫМ не позже d−1 (M-d1: закрытая в d не входит), в окне 252 рабочих дня.
Корзина удержания при входе неизвестна, поэтому ожидание берётся по распределению удержаний точки в окне:
    w_b = (число сделок окна с корзиной b, все терцили) / (число сделок окна);
    PnL̂ = Σ_b w'_b · mean(gross | терциль, b),  издержки̂ = Σ_b w'_b · mean(kom + short | терциль, b),
где сумма — по корзинам, у которых в ячейке (терциль, b) есть сделки, а w' — веса w_b, перенормированные на эти корзины.
n = число сделок терциля в окне; n < MIN_SDELOK (20) → вход разрешён, malo=True (умолчание плана). k=None («выкл.») — всё разрешено.
baza="netto" — мутация (валовой − издержки ≥ k·издержки), в расчёт не входит. Корзины и OKNO — из izderzhki_zh150 (Д7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.izderzhki_zh150 import KORZ, OKNO, korzina

MIN_SDELOK = 20
KS = (None, 2, 3)


def _ozhidanie(g: pd.DataFrame, d, tercil) -> tuple:
    """(PnL̂, издержки̂, n) на день входа d; nan, nan, n, если оценить нечем."""
    d = np.datetime64(pd.Timestamp(d), "D")
    lo = np.busday_offset(d, -OKNO, roll="forward")
    vy = g.vyhod.values.astype("datetime64[D]")
    w = g[(vy <= d - np.timedelta64(1, "D")) & (vy > lo)]
    t = w[w.tercil == tercil]
    if len(t) == 0:
        return np.nan, np.nan, 0
    vesa = w.korz.value_counts()
    yach = t.assign(izd=t.kom + t["short"].fillna(0)).groupby("korz")[["gross", "izd"]].mean()
    ww = np.array([vesa.get(b, 0) for b in KORZ if b in yach.index], float)
    ym = yach.reindex([b for b in KORZ if b in yach.index])
    return float((ww * ym.gross).sum() / ww.sum()), float((ww * ym.izd).sum() / ww.sum()), len(t)


def maska_vhoda(zhurnal: pd.DataFrame, vhody: pd.DataFrame, k, baza: str = "valovoj") -> pd.DataFrame:
    """Маска по строкам vhody (vhod, tercil): столбцы razresheno, malo, pnl_hat, izd_hat, n.
    zhurnal — теневые сделки точки: vyhod, tercil, uderzhanie, gross, kom, short (проценты, как в izderzhki_zh150)."""
    g = zhurnal.assign(vyhod=pd.to_datetime(zhurnal.vyhod), korz=zhurnal.uderzhanie.map(korzina)) if len(zhurnal) else zhurnal.assign(korz=[])
    rows = []
    for r in vhody.itertuples():
        p, izd, n = _ozhidanie(g, r.vhod, r.tercil) if len(g) else (np.nan, np.nan, 0)
        malo = n < MIN_SDELOK
        if k is None or malo:
            ok = True
        else:
            ok = bool((p if baza == "valovoj" else p - izd) >= k * izd)
        rows.append(dict(razresheno=ok, malo=malo and k is not None, pnl_hat=p, izd_hat=izd, n=n))
    return pd.DataFrame(rows, index=vhody.index, columns=["razresheno", "malo", "pnl_hat", "izd_hat", "n"])
