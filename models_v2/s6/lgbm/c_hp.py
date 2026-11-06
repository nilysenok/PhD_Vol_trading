"""Волна C, поток LGBM (Ж-206): HP с lambda_l1 и min_gain_to_split (подкласс `grid.Hp`, B не меняется) и оси одно-осевых развёрток.
Шаги объявлены в GATES_C_lgbm.md до счёта: ось -> (нижняя точка, верхняя точка) как функции HP года; применяются к HP каждого года выбора."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from models_v2.s6.lgbm.grid import Hp


@dataclass(frozen=True)
class HpC(Hp):
    l1: float = 0.0
    gain: float = 0.0

    def params(self, seed: int | None = None) -> dict:
        p = super().params(seed)
        p.update(lambda_l1=float(self.l1), min_gain_to_split=float(self.gain))
        return p

    def label(self) -> str:
        return super().label() + ("_l1%g" % self.l1 if self.l1 else "") + ("_g%g" % self.gain if self.gain else "")


def lift(h: Hp) -> HpC:
    return h if isinstance(h, HpC) else HpC(**asdict(h))


def lift_y(hp_y: dict) -> dict:
    return {Y: lift(h) for Y, h in hp_y.items()}


def _set(h: Hp, **kw) -> HpC:
    return replace(lift(h), **kw)


# ось -> (поле, f_lo(h), f_hi(h)); относительные шаги: листья/min_data/L2 — ÷2 и ×2 (L2: ÷3 и ×3), lr — ÷2 и ×2,
# ff — ∓0,2 в [0,3; 1,0]; num_leaves hi у глубинно-ограниченной базы (max_depth > 0) поднимает и глубину на 1 (иначе листья не добавятся);
# max_depth при базе d > 0: {max(2, d−1); d+1} при прежнем num_leaves, при базе −1: {3; 6};
# абсолютные: bagging {0,6; 1,0}, lambda_l1 {1; 5}, min_gain {1; 10}.
AXES = {
    "num_leaves": lambda h: (_set(h, num_leaves=max(3, round(h.num_leaves / 2))),
                             _set(h, num_leaves=h.num_leaves * 2, max_depth=h.max_depth + 1 if h.max_depth > 0 else -1)),
    "max_depth": lambda h: (_set(h, max_depth=max(2, h.max_depth - 1)), _set(h, max_depth=h.max_depth + 1)) if h.max_depth > 0 else (_set(h, max_depth=3), _set(h, max_depth=6)),
    "min_data": lambda h: (_set(h, min_data=max(5, round(h.min_data / 2))), _set(h, min_data=h.min_data * 2)),
    "lr": lambda h: (_set(h, lr=round(h.lr / 2, 6)), _set(h, lr=round(h.lr * 2, 6))),
    "ff": lambda h: (_set(h, ff=round(max(0.3, h.ff - 0.2), 2)), _set(h, ff=round(min(1.0, h.ff + 0.2), 2))),
    "bag": lambda h: (_set(h, bag=0.6), _set(h, bag=1.0)),
    "l1": lambda h: (_set(h, l1=1.0), _set(h, l1=5.0)),
    "l2": lambda h: (_set(h, l2=round(h.l2 / 3, 6)), _set(h, l2=round(h.l2 * 3, 6))),
    "gain": lambda h: (_set(h, gain=1.0), _set(h, gain=10.0)),
}
AXIS_NAMES = ("num_leaves", "max_depth", "min_data", "lr", "ff", "bag", "l1", "l2", "gain")
FIELD = {"num_leaves": "num_leaves", "max_depth": "max_depth", "min_data": "min_data", "lr": "lr", "ff": "ff", "bag": "bag", "l1": "l1", "l2": "l2", "gain": "gain"}


def move(hp_y: dict, axis: str, side: int) -> dict:
    """side 0 — нижняя точка оси, 1 — верхняя; для каждого года выбора."""
    return {Y: AXES[axis](h)[side] for Y, h in hp_y.items()}


def changed_axes(hp_y: dict, base_y: dict) -> list:
    """Оси, по которым конфигурация отличается от базы хотя бы в одном году."""
    return [a for a in AXIS_NAMES if any(getattr(lift(hp_y[Y]), FIELD[a]) != getattr(lift(base_y[Y]), FIELD[a]) for Y in hp_y)]
