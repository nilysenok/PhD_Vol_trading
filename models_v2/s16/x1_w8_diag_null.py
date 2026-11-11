"""W8-A: перенос `w8/null.py` БЕЗ изменения логики (`W8_SPEC.md` разд. 1.5, Д-2: ПОСЛЕ просмотра E_verdict). Файлов не пишет.
Нули для W8D8: (а) перестановка меток групп между днями gap (seed 0, 500 раз); (б) 19 циклических сдвигов ряда ln RV^IMOEX на s = 252 + 100·(j − 1) с PIT-терцилями заново; (в) c* групп по годам.
Публичное API: `verdict(x, grp) -> 0/1` (П1–П5 по `blocks_cde`), `raw_imoex(x)`, `run(x, feats, n_perm=500) -> (прошли перестановки, n_perm, [прошли сдвиги ×19], {год: {группа: (n, c*)}})`."""
from __future__ import annotations

import numpy as np

from models_v2.s16 import x1_w8_diag_cde as B
from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s16 import x1_w8_diag_feat as F


def verdict(x, grp):
    _, _, re_ = B.blocks_cde(x, {"f": grp})
    return re_[0]["passed"]


def raw_imoex(x):
    im = F.imoex_day()
    irv = im["rv"].reindex(x["d"]).to_numpy()
    return np.log(np.where(irv > 0, irv, np.nan))


def run(x, feats, n_perm=500):
    g = x["gap"].to_numpy()
    base = np.asarray(feats["W8D8_imoex_rv"], object)
    rng = np.random.default_rng(0)
    idx = np.where(g == 1)[0]
    perm_pass = 0
    for _ in range(n_perm):
        gg = base.copy()
        gg[idx] = rng.permutation(base[idx])
        perm_pass += verdict(x, gg)
    raw = raw_imoex(x)
    shift_pass = []
    for j in range(1, 20):
        s = 252 + 100 * (j - 1)
        shift_pass.append(verdict(x, K.terc_past(np.roll(raw, s))))
    yrs = {}
    u = x["u"].to_numpy()
    for y in K.YEARS:
        m = (x["yr"] == y).to_numpy() & (g == 1)
        yrs[y] = {G: (int((m & (base == G)).sum()), float(np.log(np.mean(np.exp(u[m & (base == G)])))) if (m & (base == G)).any() else np.nan)
                  for G in ("T1", "T2", "T3")}
    return perm_pass, n_perm, shift_pass, yrs
