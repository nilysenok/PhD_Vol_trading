"""P1a (R10, У-3): мир замороженных K2 (XGB `x2_best_j`) и K3 (LGBM `C0_inh_F4`) для ПРОИЗВОЛЬНОЙ бумаги.
Замороженный код (`s2/s4/s5/s6`) не правится: привязка к SBER (`prov.BARS`) обходится тем, что мир собирается здесь из переданных баров — теми же вызовами, что `run_fix.prepare`
и `s5/b2/world.load` (F, сетка `har.eligible`, дизайн Б-2 `m_mkx`, признаки F1–F6 `b2.feat.build`, ε), а годовой выбор HP (И2) — тем же правилом, что `run_fix.choose`
(сетка GRID132, фолды LOYO y ∈ [2015, Y−1], QLIKE(б), ничья — `select.order_key`), но без записи в чужие каталоги.
Рыночный блок Б-2 считается по тому же пулу 17 основных бумаг ([SBER] + PAPERS), что у SBER, для любой бумаги (пул не зависит от бумаги)."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from models_v2 import har
from models_v2.s2 import features as ft
from models_v2.s4 import design, fit, mkt, prov4, select
from models_v2.s4.fit import HP
from models_v2.s5 import common, pit
from models_v2.s5.b2 import feat as b2feat
from models_v2.s6 import paths6

POOL = ["SBER"] + list(prov4.PAPERS)
ACC = common.DEEP / "b2" / "accepted.json"
FIRST_PAIRS = 126           # У-3: первый год Y, для которого у бумаги >= 126 пар до 01.01.Y
YEARS = paths6.YEARS


LN_BV = 4                   # столбец ln BV в F.X (`ft.NAMES`)


def _bv_to_nan(col: np.ndarray) -> np.ndarray:
    """У-18 (GATES_PANEL.md): BV = 0 (вырожденный день) даёт ln BV = −inf; нефинитное значение — пропуск (XGB/LGBM умеют пропуски), конечные — без изменений."""
    return np.where(np.isfinite(col), col, np.nan)


def pool_papers(read) -> list:
    """read(тикер) -> бары; результат — как `mkt.load_papers`: [paper(SBER)] + paper(PAPERS)."""
    return [mkt.paper(read(t)) for t in POOL]


def build_world(bars: pd.DataFrame, imoex: pd.DataFrame, pool: list) -> dict:
    """Мир бумаги: F, grid, dz0 (Б-2 m_mkx), lvl (level_base для И2), feats, acc, eps. HP (`i2`) задаётся отдельно."""
    F = ft.build(bars)
    F.X[:, LN_BV] = _bv_to_nan(F.X[:, LN_BV])
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))
    M, _ = mkt.block(None, F.X[:, [0, 1, 2]], F.seg, F.dates, pool, imoex)
    acc = {g: tuple(c) for g, c in json.loads(ACC.read_text()).items()}
    return {"F": F, "grid": np.asarray(grid, int), "dz0": design.make(F, "m", design.columns(M, "imoex")), "lvl": design.make(F, "level", None),
            "feats": b2feat.build(F, bars, imoex), "acc": acc, "eps": pit.eps_floor(F.rv), "i2": {}}


def pairs_before(w: dict, Y: int) -> int:
    """Число обучающих пар (цель < 01.01.Y) дизайна K2/K3 (те же `fit.pair_index`, что в боевом пути)."""
    dz = w["dz0"]
    return int(len(fit.pair_index(w["F"].seg, dz.X, dz.y, (0, 1, 2, 3), select.cut_row(w["F"], Y))))


def first_year(w: dict, need: int = FIRST_PAIRS):
    """У-3: первый Y из 2017…2026 с >= need парами до 01.01.Y; None, если такого нет."""
    return next((Y for Y in YEARS if pairs_before(w, Y) >= need), None)


def loyo_scores(w: dict, years, procs: int = 2) -> pd.DataFrame:
    """Строки (Y, label, score, n_folds): средний QLIKE(б) фолдов LOYO y ∈ [2015, Y−1] при цели < 01.01.Y (`select.run_loyo`, сетка `select.GRID132`)."""
    cands = {Y: {hp: ("lvl", hp, False) for hp in select.GRID132} for Y in years}
    return select.run_loyo(w["F"], {"lvl": w["lvl"]}, cands, years, procs)[0]


def choose_i2(w: dict, years, procs: int = 2) -> dict:
    """И2 бумаги по правилу `run_fix.choose` (level_base × GRID132): {Y: HP}; год без конечной оценки пропускается (в словаре нет)."""
    sc = loyo_scores(w, years, procs)
    out = {}
    for Y in years:
        s = sc[sc.Y == Y].set_index("label")["score"].to_dict()
        p = select.pick(s, select.order_key)
        if p is not None:
            out[Y] = p
    return out


def i2_dump(i2: dict) -> str:
    return json.dumps({str(Y): asdict(h) for Y, h in sorted(i2.items())})


def i2_load(text: str) -> dict:
    return {int(Y): HP(**v) for Y, v in json.loads(text).items()}
