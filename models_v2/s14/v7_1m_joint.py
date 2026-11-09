"""V7-1a-M: адаптер совместного обучения (Stage B) к энкодерам листа 1a-E (импорт `v7_1e_net`, `v7_1e_rep`; файлы 1a-E не правятся). `PCAEncoder` повторяет `v7_1e_rep.encode` + `transform_pca` дифференцируемо в torch float64:
z = [среднее по барам окна ‖ последнее] скрытое состояние (8+8) → стандартизация и главные компоненты (4), оценённые по z дат < 01.01.Y (`fit_pca_past`, PIT — у 1a-E) и ЗАМОРОЖЕННЫЕ (буферы); обучаются только веса энкодера φ, штраф λ_e‖φ − φ₀‖² — в `LS.Joint`.
Вход адаптера — тензор (N, K+1, L): каналы 0…K−1 = `to_seq` (слоты без бара обнулены), канал K = маска слотов (`pack_seq`). Данных не читает."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from models_v2.s14 import v7_1e_net as NET


def pack_seq(X, M) -> np.ndarray:
    """X (N, W, 53, 11), M (N, W, 53) -> (N, 12, W·53) float64: последовательность и маска слотов."""
    x = NET.to_seq(X, M).double().numpy()
    return np.concatenate([x, np.asarray(M).reshape(len(M), 1, -1).astype(np.float64)], axis=1)


class PCAEncoder(nn.Module):
    def __init__(self, base: nn.Module, pca: dict):
        super().__init__()
        self.base = base
        for k in ("mu", "sd", "comps", "ssd"):
            self.register_buffer(k, torch.as_tensor(pca[k], dtype=torch.float64))

    def forward(self, xm: torch.Tensor) -> torch.Tensor:
        x, m = xm[:, :-1], xm[:, -1] > 0.5
        h = self.base.hidden(x)                                                   # (N, 8, L)
        mf = m.to(h.dtype)
        mean = (h * mf[:, None, :]).sum(2) / mf.sum(1, keepdim=True).clamp(min=1.0)
        last_i = m.shape[1] - 1 - torch.flip(m, [1]).to(torch.int64).argmax(1)
        last = h[torch.arange(len(h)), :, last_i] * m.any(1, keepdim=True).to(h.dtype)
        raw = torch.cat([mean, last], 1)
        return ((raw - self.mu) / self.sd) @ self.comps.T / self.ssd
