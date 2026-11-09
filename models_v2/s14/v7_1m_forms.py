"""V7-1a-M (Ж-232, Ж-233), формы прогноза, torch float64. G = ln(F/H), F = H·exp(G) > 0 при любом конечном G.
Residual MoE: G = Σ_{k=1..K} π_k·g_k(z), π = softmax на K+1 выходов (выход 0 — «ничего», g₀ ≡ 0); эксперт g_k = C·tanh((a_k + b_kᵀz)/C), C = 1, d_z = 4.
Маршрутизатор: проекция входа в 4 (матрица без смещения) → линейный softmax на K+1 выходов. Вход: A — Z (9), B — (Z, z) (13), C — z (4).
Якорь: G = w·ℓ + Σ π_k g_k, w — обучаемый скаляр, ℓ = ln(M/H). Плоский корректор: G = C·tanh(MLP 13→5→1), последний слой с нулём в начале.
Нулевая инициализация экспертов и выходного слоя: до обучения G ≡ 0 и F = H побитно. Модуль данных не читает."""
from __future__ import annotations

import contextlib

import torch
from torch import nn

DZ, DR, C = 4, 4, 1.0
DZ_GATE = 9                                    # признаков режима Z
IN_DIM = {"A": DZ_GATE, "B": DZ_GATE + DZ, "C": DZ}
INIT_STD = 0.3                                 # маршрутизатор — малый случайный старт (seed); эксперты — нули


@contextlib.contextmanager
def deterministic():
    """Один поток, детерминированные алгоритмы (план п. 7); прежнее состояние процесса ВОССТАНАВЛИВАЕТСЯ на выходе (класс № 43/52: общее состояние не утекает в соседние тесты)."""
    prev = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0])
        torch.set_num_threads(prev[1])


def router_input(kind: str, Zg: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Состав входа маршрутизатора по Ж-233 п. 3. МУТАНТЫ «z в A», «Z в C» подменяют эту функцию в тесте."""
    if kind == "A":
        return Zg
    if kind == "B":
        return torch.cat([Zg, z], dim=1)
    if kind == "C":
        return z
    raise ValueError("маршрутизатор %r не из A/B/C" % kind)


def _init_(module: nn.Module, seed: int, zero: tuple = ()) -> None:
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name in zero:
                p.zero_()
            else:
                p.copy_(INIT_STD * torch.randn(p.shape, generator=g, dtype=torch.float64))


class ResMoE(nn.Module):
    """K экспертов + маршрутизатор A/B/C; anchor=True добавляет w·ℓ. forward(Zg, z, ell) -> (G, π)."""

    def __init__(self, router: str, K: int = 2, anchor: bool = False, seed: int = 0):
        super().__init__()
        self.kind, self.K, self.anchor = router, K, anchor
        self.proj = nn.Linear(IN_DIM[router], DR, bias=False)
        self.out = nn.Linear(DR, K + 1)
        self.a = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K, DZ))
        if anchor:
            self.w = nn.Parameter(torch.zeros(()))
        self.double()
        _init_(self, seed, zero=("a", "b", "w"))

    def pi(self, Zg, z):
        return torch.softmax(self.out(self.proj(router_input(self.kind, Zg, z))), dim=1)

    def experts(self, z):
        return C * torch.tanh((self.a[None, :] + z @ self.b.T) / C)

    def forward(self, Zg, z, ell):
        p = self.pi(Zg, z)
        G = (p[:, 1:] * self.experts(z)).sum(dim=1)
        if self.anchor:
            G = self.w * ell + G
        return G, p

    def theta_params(self) -> list:
        return list(self.parameters())

    def extra_penalty(self):
        return 0.0


class Flat(nn.Module):
    """Плоский корректор g(z, Z) = C·tanh(MLP 13→5→1) (конфигурации 1–3 берут только z: in_dim = 4)."""

    def __init__(self, in_dim: int = DZ_GATE + DZ, seed: int = 0, hidden: int = 5):
        super().__init__()
        self.use_gate = in_dim == DZ_GATE + DZ
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, 1)
        self.double()
        _init_(self, seed, zero=("l2.weight", "l2.bias"))

    def forward(self, Zg, z, ell):
        x = torch.cat([Zg, z], dim=1) if self.use_gate else z
        return C * torch.tanh(self.l2(torch.tanh(self.l1(x)))).squeeze(1), None

    def theta_params(self) -> list:
        return list(self.parameters())

    def extra_penalty(self):
        return 0.0


def forecast(H: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """F = H·exp(G); при G ≡ 0 F = H побитно (exp(0) = 1)."""
    return H * torch.exp(G)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def counts(K: int = 2) -> dict:
    """Объявленные числа параметров (план п. 4–5): маршрутизатор A/B/C, эксперты, плоский, якорь."""
    r = {k: n_params(ResMoE(k, K)) - K * (DZ + 1) for k in "ABC"}
    return dict(router=r, experts=K * (DZ + 1), flat=n_params(Flat()), moe_b=n_params(ResMoE("B", K)), anchor_b=n_params(ResMoE("B", K, anchor=True)))
