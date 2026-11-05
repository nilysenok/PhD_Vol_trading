"""Поток LGBM волны B (Ж-206): гиперпараметры LightGBM и сетка LB0. Сетка объявлена до счёта (GATES_B_lgbm.md), подвыборка — seed 42."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

NL, MD, L2, LR, FF, BAG = (7, 15, 31), (30, 100, 200), (2.0, 10.0, 30.0), (0.02, 0.05), (0.5, 0.8), (0.8, 1.0)
SEED_SUB, N_SUB, MAX_TREES = 42, 15, 300
PEN = 0.25   # штраф за худший прошлый год: mean + PEN·(max − mean)


@dataclass(frozen=True)
class Hp:
    num_leaves: int
    min_data: int
    l2: float
    lr: float
    ff: float
    bag: float
    max_depth: int = -1
    max_trees: int = MAX_TREES
    seed: int = 42

    def params(self, seed: int | None = None) -> dict:
        seed = self.seed if seed is None else seed
        return dict(num_leaves=self.num_leaves, max_depth=self.max_depth, min_data_in_leaf=self.min_data, lambda_l1=0.0, lambda_l2=self.l2,
                    learning_rate=self.lr, feature_fraction=self.ff, bagging_fraction=self.bag, bagging_freq=1 if self.bag < 1 else 0,
                    min_gain_to_split=0.0, seed=int(seed), num_threads=1, deterministic=True, force_row_wise=True, verbose=-1)

    def label(self) -> str:
        s = "" if self.seed == 42 else "_s%d" % self.seed
        return "nl%d_md%d_l2%g_lr%g_ff%g_bag%g_dp%d%s" % (self.num_leaves, self.min_data, self.l2, self.lr, self.ff, self.bag, self.max_depth, s)


def from_i2(hp) -> Hp:
    """Наследованные HP (I2 из сетки 132 XGBoost) в поля LightGBM — как `s5/d/lgbfit.params`: глубина d, 2^d листьев, bagging 0,8 / 1."""
    d = int(hp.depth)
    return Hp(2 ** d, max(1, int(hp.min_child)), float(hp.l2), float(hp.lr), float(hp.colsample), float(hp.subsample), d, int(hp.max_trees))


def full_grid() -> tuple:
    return tuple(Hp(*p) for p in itertools.product(NL, MD, L2, LR, FF, BAG))   # 3·3·3·2·2·2 = 216


def lb0_points() -> tuple:
    """15 точек: случайная подвыборка полной сетки (seed 42, без возвращения), порядок — по номеру в сетке."""
    g = full_grid()
    ix = np.sort(np.random.default_rng(SEED_SUB).choice(len(g), N_SUB, replace=False))
    return tuple(g[i] for i in ix)


def order_key(hp: Hp) -> tuple:
    """Ничья (|Δ| ≤ 1e-12): проще — меньше листьев, больше min_data, больше L2, меньше lr, меньше ff, bagging раньше."""
    return (hp.num_leaves, -hp.min_data, -hp.l2, hp.lr, hp.ff, hp.bag)


def trial_id(kind: str, hp: dict | Hp | str, feats=(), drop: str = "", extra: dict | None = None) -> str:
    """Хэш конфигурации: вид (цель/потеря/веса), HP (или ссылка), добавленные столбцы, drop, прочее."""
    h = asdict(hp) if isinstance(hp, Hp) else hp
    cfg = {"kind": kind, "hp": h, "feats": list(feats), "drop": drop, "extra": extra or {}, "seed": 42}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
