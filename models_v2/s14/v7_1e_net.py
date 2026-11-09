"""V7-1a-E: M10-энкодеры TCN и CNN (torch). Только архитектуры; данные SBER не читаются.
Вход — последовательность окна W дней: W×53 слотов подряд, каналы = 11 каналов листа 0 (`v7_0_seq.CH`); слот без бара — нули (маска из листа 0).
Все свёртки причинные: выход на позиции j зависит от входов ≤ j (левое дополнение нулями).
TCN: 1×1 (11→8) + 4 остаточных блока (ядро 3, расширения 1, 2, 4, 8) + голова 8→1.  CNN: ядро 5, 11→8→8 + голова 8→1.
Голова предсказывает ln(r²_{j+1}+ε) — метку предобучения (Stage A)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v2.s14.v7_0_seq import K, SLOTS

HID = 8
DIL = (1, 2, 4, 8)
PARAMS = {"tcn": 905, "cnn": 785}     # объявлено; тест сверяет с фактическим числом, предел плана — 1000
MAX_PARAMS = 1000


def pad_left(x: torch.Tensor, p: int) -> torch.Tensor:
    """Причинное дополнение: p нулей слева по времени."""
    return F.pad(x, (p, 0))


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv1d(K, HID, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(HID, HID, 3, dilation=d) for d in DIL)
        self.head = nn.Linear(HID, 1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for c, d in zip(self.blocks, DIL):
            h = h + torch.relu(c(pad_left(h, 2 * d)))
        return h


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(K, HID, 5)
        self.c2 = nn.Conv1d(HID, HID, 5)
        self.head = nn.Linear(HID, 1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.c1(pad_left(x, 4)))
        return torch.relu(self.c2(pad_left(h, 4)))


def forward_pred(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x (B, K, L) -> прогноз метки на каждой позиции (B, L)."""
    return model.head(model.hidden(x).transpose(1, 2)).squeeze(-1)


def build(kind: str) -> nn.Module:
    if kind == "tcn":
        return TCN()
    if kind == "cnn":
        return CNN()
    raise ValueError("kind: tcn | cnn")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_seq(X, M) -> torch.Tensor:
    """X (N, W, SLOTS, K), M (N, W, SLOTS) bool -> (N, K, W·SLOTS); слоты без бара обнуляются."""
    x = torch.as_tensor(X, dtype=torch.float32) * torch.as_tensor(M).to(torch.float32)[..., None]
    n, w = x.shape[0], x.shape[1]
    assert x.shape[2:] == (SLOTS, K), "форма X вне (N, W, 53, 11)"
    return x.reshape(n, w * SLOTS, K).transpose(1, 2).contiguous()
