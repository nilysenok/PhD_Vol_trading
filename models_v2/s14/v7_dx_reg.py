"""V7-DX, G39 (в): регуляризация.
1) Частоты выбранных λ_g и λ_LB по переоценкам и seeds (конфигурации 1, 4, 6): из повтора walk-forward, побитно равного сохранённым прогнозам (`K.same_as_saved`); для 4 и 6 — сверка с `lam_selected` в CHECKPOINT ключ `2`
   (для 1 в CHECKPOINT не записано — только повтор). Для 5 и 7 — из CHECKPOINT как есть.
2) Распределение |G| по годам (сохранённый прогноз = среднее 5 seeds; и по seeds из повтора): квантили, доля |G| > 0,9 (граница tanh при c = 1).
3) В ВЫБОРКЕ: снижение QLIKE к H на обучающих парах ансамбля при λ_g ∈ {0; 0,01; 0,1; 1} для конфигураций 1 и 4 на переоценках 2019-01, 2020-01, 2021-01 (те же батчи, что в повторе; λ_LB = 0).
   Это обучение на уже закрытых парах с параметрами вне решётки (0; 0,01) — только в выборке; прогнозов на новых датах нет. Решётка 0,1 и 1 воспроизводит перехваченный выбор (проверка `lattice_repro`)."""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1b_cfg as CFG, v7_1m_fit as FT, v7_1m_forms as FM, v7_2_cfg as C2, v7_dx_core as K

LAMS = (0.0, 0.01, 0.1, 1.0)
SWEEP_DATES = ("2019-01", "2020-01", "2021-01")
SWEEP_CFGS = ("1", "4")
EDGE = 0.9


def lam_table(rts: dict, cp2: dict) -> pd.DataFrame:
    """Частоты (λ_g, λ_LB) по годам переоценки и в целом; колонка `cp_match` — совпадение с CHECKPOINT (4, 6)."""
    rows = []
    for cfg, rt in rts.items():
        allc = collections.Counter()
        for y, g in list(rt.groupby("year")) + [("all", rt)]:
            c = collections.Counter((a, b) for ls, lb in zip(g["lam_g"], g["lam_lb"]) for a, b in zip(ls, lb))
            if y == "all":
                allc = c
            rows += [dict(cfg=cfg, year=str(y), lam_g=a, lam_lb=b, n=n, share=n / sum(c.values())) for (a, b), n in sorted(c.items())]
        stored = cp2.get("cfg" + cfg, {}).get("lam_selected")
        mine = {"%g/%g" % k: v for k, v in allc.items()}
        for r in rows:
            if r["cfg"] == cfg:
                r["cp_match"] = (mine == stored) if stored is not None else None
    for cfg in ("5", "7"):
        for k, n in cp2["cfg" + cfg]["lam_selected"].items():
            a, b = (float(x) for x in k.split("/"))
            rows.append(dict(cfg=cfg, year="all", lam_g=a, lam_lb=b, n=n, share=n / sum(cp2["cfg" + cfg]["lam_selected"].values()), cp_match=None))
    return pd.DataFrame(rows)


def g_table(reps: dict) -> pd.DataFrame:
    rows = []
    for cfg, rep in reps.items():
        sv = pd.read_parquet(K.SAVED[cfg])
        gm, ty = sv[K.GCOL[cfg]].to_numpy(float), pd.DatetimeIndex(sv["t_plus"]).year.to_numpy()
        ix = rep["index"]
        pos = ix.get_indexer(pd.DatetimeIndex(sv["d"]))
        gs = np.abs(rep["G"][:, pos])
        for y in sorted(set(ty)):
            m = ty == y
            a = np.abs(gm[m])
            rows.append(dict(cfg=cfg, year=int(y), n=int(m.sum()), mean_abs_g=float(a.mean()), q50=float(np.quantile(a, .5)), q90=float(np.quantile(a, .9)), q99=float(np.quantile(a, .99)),
                             share_gt09_avg=float((a > EDGE).mean()), share_gt09_seed=float((gs[:, m] > EDGE).mean()), max_seed_abs_g=float(gs[:, m].max())))
    return pd.DataFrame(rows)


def _find(rep: dict, ym: str) -> int:
    """Номер перехвата (среди переоценок с моделью) для первой переоценки месяца ym ('2019-01')."""
    fitted = [f for f in rep["fits"][0] if f["fitted"]]
    return next(i for i, f in enumerate(fitted) if f["d0"].strftime("%Y-%m") == ym)


def _fit(cfg: str, spec: dict, s: int, cap: dict, grid: tuple) -> dict:
    if cfg == "1":
        with CFG.lin5_patch():
            return FT.fit_config(spec, cap["tr"], cap["va"], (s,), grid)
    return FT.fit_config(spec, cap["tr"], cap["va"], (s,), grid)


def _lnf(res: list, caps: list) -> np.ndarray:
    """ln F ансамбля seeds на обучающих парах (tr ∪ va) — как в `K.capture`."""
    return np.mean(np.stack([FT.ensemble_lnF(r["models"], FT.cat(c["tr"], c["va"])).numpy() for r, c in zip(res, caps)]), axis=0)


def sweep_one(cfg: str, rep: dict, ym: str) -> list:
    """Снижение QLIKE к H в выборке при λ_g из LAMS (λ_LB = 0) на переоценке месяца ym; `lattice_repro` — решётка плана воспроизводит перехваченный ln F побитно."""
    caps = [c[_find(rep, ym)] for c in rep["caps"]]
    spec = CFG.SPEC if cfg == "1" else C2.SPECS[cfg]
    H, rv, rows = caps[0]["H"], caps[0]["rv"], []
    with FM.deterministic():
        for lam in LAMS:
            res = [_fit(cfg, spec, s, caps[s], ((lam, 0.0),)) for s in FT.SEEDS]
            qh, qf, red = K.q_red(rv, H, np.exp(_lnf(res, caps)))
            rows.append(dict(cfg=cfg, date=ym, lam_g=lam, n_pairs=len(H), q_H=qh, q_F=qf, red_in_pct=red, val_q_mean=float(np.mean([r["val"][0]["q"] for r in res])),
                             mean_epochs=float(np.mean([r["epochs"][0] for r in res]))))
        lat = [_fit(cfg, spec, s, caps[s], CFG.GRID if cfg == "1" else C2.GRIDS[cfg]) for s in FT.SEEDS]
        ok = bool(np.array_equal(_lnf(lat, caps), np.mean(np.stack([c["lnf"] for c in caps]), axis=0)))
    return [dict(r, lattice_repro=ok) for r in rows]


def sweep_table(reps: dict) -> pd.DataFrame:
    return pd.DataFrame([r for cfg in SWEEP_CFGS for ym in SWEEP_DATES for r in sweep_one(cfg, reps[cfg], ym)])


def tables(reps: dict, rts: dict, cp2: dict) -> dict:
    return dict(LAM=lam_table(rts, cp2), G=g_table(reps), LAMSWEEP=sweep_table(reps))
