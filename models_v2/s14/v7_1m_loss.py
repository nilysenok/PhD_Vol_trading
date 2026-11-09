"""V7-1a-M, функция потерь (план п. 4): L = mean(ln F + RV/F) + λ_g·mean(G²) + λ₂‖θ‖² + λ_LB·Σ_k(mean_t π_k − 1/(K+1))² + λ_s·mean‖π_t − π_{t−1}‖² (+ λ_e‖φ − φ₀‖² в совместном обучении).
Первое слагаемое — QLIKE с точностью до константы (ln F + RV/F = QLIKE + ln RV + 1). Строки батча идут по датам: λ_s сравнивает соседние строки.
θ — все параметры головы (маршрутизатор, эксперты, якорь w, плоский корректор), включая смещения; энкодер штрафуется только λ_e."""
from __future__ import annotations

import torch
from torch import nn

REG = dict(lam2=1e-3, lam_s=1e-2, lam_e=1e-2)      # ОБЪЯВЛЕНО одной константой (Ж-232, п. 2 задания): L2, сглаживание весов, стягивание энкодера к предобучению


def data_term(H, rv, G) -> torch.Tensor:
    lnF = torch.log(H) + G
    return torch.mean(lnF + rv / (H * torch.exp(G)))


def qlike_np(rv, F):
    """QLIKE определения проекта (rv/F − ln(rv/F) − 1), torch: для ранней остановки."""
    r = rv / F
    return torch.mean(r - torch.log(r) - 1.0)


def loss(model: nn.Module, b: dict, lam_g: float, lam_lb: float, reg: dict = REG) -> torch.Tensor:
    G, pi = model(b["Zg"], b["z"], b["ell"])
    L = data_term(b["H"], b["rv"], G) + lam_g * torch.mean(G ** 2)
    L = L + reg["lam2"] * sum(torch.sum(p ** 2) for p in model.theta_params())
    if pi is not None:
        L = L + lam_lb * torch.sum((pi.mean(dim=0) - 1.0 / pi.shape[1]) ** 2)
        if pi.shape[0] > 1:
            L = L + reg["lam_s"] * torch.mean(torch.sum((pi[1:] - pi[:-1]) ** 2, dim=1))
    return L + model.extra_penalty()


class Joint(nn.Module):
    """Stage B: encoder(X) -> z (n, 4), затем голова. φ₀ — копия весов предобучения; штраф lam_e·‖φ − φ₀‖². Энкодер и его предобучение — лист 1a-E (`v7_1e_*`): здесь любой nn.Module с выходом (n, 4)."""

    def __init__(self, head: nn.Module, encoder: nn.Module, lam_e: float = REG["lam_e"]):
        super().__init__()
        self.head, self.encoder, self.lam_e = head, encoder, lam_e
        self.phi0 = [p.detach().clone() for p in encoder.parameters()]

    def forward(self, Zg, x, ell):
        return self.head(Zg, self.encoder(x), ell)

    def theta_params(self) -> list:
        return self.head.theta_params()

    def extra_penalty(self):
        return self.lam_e * sum(torch.sum((p - p0) ** 2) for p, p0 in zip(self.encoder.parameters(), self.phi0))
