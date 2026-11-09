"""V6-5, G41: помесячная переоценка робастных обучателей. Та же схема, что у движка листа 1a (`v6_1_wf.walk_forward`: первый торговый день месяца, закрытые пары t⁺ ≤ d0, расширяющееся окно),
но обучателю передаётся ещё t⁺ закрытых пар (по ним строятся среды, PIT). Движок листа 1a не менялся: используются его `closed_rows`, `refit_positions`, `residual`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_1_wf as WF


def walk_forward_env(tab: pd.DataFrame, S: pd.DataFrame, fit, first: str = WF.FIRST_REFIT) -> dict:
    """fit(X, e, tp) -> модель с методом g(X). -> {g, fits (d0, n_train, model, tp_max)}; до первой переоценки g = 0."""
    X, e, tp = S.to_numpy(float), WF.residual(tab), tab["t_plus"].to_numpy()
    g = np.zeros(len(tab))
    pos = WF.refit_positions(tab.index, first)
    fits = []
    for a, b in zip(pos, pos[1:] + [len(tab)]):
        tr = WF.closed_rows(tp, tab.index[a])
        m = fit(X[tr], e[tr], tp[tr])
        g[a:b] = m.g(X[a:b])
        fits.append(dict(d0=tab.index[a], n_train=len(tr), model=m))
    return dict(g=pd.Series(g, index=tab.index, name="g"), fits=fits)


def conv_table(run: dict, cid: str) -> pd.DataFrame:
    """Сходимость по датам переоценки: итерации L-BFGS-B, 2-норма градиента цели в решении, признак успеха, число сред, λ, значение цели."""
    rows = [dict(config=cid, d0=str(f["d0"].date()), n_train=int(f["n_train"]), n_env=f["model"].info["n_env"], lam=float(f["model"].lam), nit=f["model"].info["nit"],
                 grad_norm=f["model"].info["grad_norm"], success=f["model"].info["success"], L=f["model"].info["L"]) for f in run["fits"]]
    return pd.DataFrame(rows)
