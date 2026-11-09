"""V7-1b-B: конфигурации 1 (TCN-w5) и 2 (CNN-w5) — корректор g(z) = c·tanh(a + bᵀz), c = 1, 5 параметров (a, b₁…b₄), старт в нуле (G ≡ 0 ⇒ F = H побитно).
Потери плана: QLIKE + λ_g·mean(G²) + L2; решётка λ_g ∈ {0,1; 1}; λ_LB и λ_s НЕ применяются (маршрутизатора нет, π = None), в решётке λ_LB = 0.
Помесячный walk-forward (`v7_1m_wf.walk_forward`) по закрытым парам t⁺ ≤ d. Для seed s ∈ 0–4: при переоценке в году Y z ВСЕХ обучающих пар (в т. ч. прошлых лет и прогрева) и прогнозов блока — от энкодера года Y того же seed (`z_by_year`, версия 2, лист FX: версия 1 брала для каждой строки z энкодера
её года — координаты разных энкодеров не согласованы, `FX_DIAG.csv`), seed корректора = seed энкодера (старт корректора детерминирован, различие между seeds — только z). Прогноз = среднее ln F по 5 seeds.
Выбор энкодера для конфигурации 3: меньшая средняя QLIKE на датах целей 2018–2021 (`CHOICE_WIN`, только они). Данные читаются при вызове функций."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from models_v2 import metrics
from models_v2.s14 import v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_wf as WF, v7_1b_sta as STA
from models_v2.s13 import v6_1_wf as W6

KINDS, CFG_ID = STA.KINDS, {"tcn": "1", "cnn": "2"}
GRID = ((0.1, 0.0), (1.0, 0.0))                       # λ_g × λ_LB (λ_LB = 0: не применяется)
SPEC = dict(kind="lin5")
CHOICE_WIN = (2018, 2021)                              # годы t⁺, по которым выбирается энкодер для 3
WINDOW_SIZE = 5


class Lin5(nn.Module):
    """g(z) = c·tanh((a + bᵀz)/c), c = 1: 5 параметров, старт в нуле. forward -> (G, None)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(FM.DZ))
        self.double()

    def forward(self, Zg, z, ell):
        return FM.C * torch.tanh((self.a + z @ self.b) / FM.C), None

    def theta_params(self) -> list:
        return list(self.parameters())

    def extra_penalty(self):
        return 0.0


@contextlib.contextmanager
def lin5_patch():
    """Подключает вид 'lin5' к `FT.make` на время вызова; прежняя функция ВОССТАНАВЛИВАЕТСЯ (класс № 43/52)."""
    orig = FT.make
    FT.make = lambda spec, seed: Lin5() if spec.get("kind") == "lin5" else orig(spec, seed)
    try:
        yield
    finally:
        FT.make = orig


def enc_year(y: int) -> int:
    """Год энкодера для дат года y. МУТАНТ «z не того года» (энкодер года y + 1) подменяет эту функцию в тесте."""
    return y


def z_of_year(kind: str, seed: int, index: pd.DatetimeIndex, year: int, load=STA.load, window: int = STA.WINDOW) -> np.ndarray:
    """(n, 4): z ОДНОГО энкодера — года `year` (seed `seed`) — на всех датах index до 31.12.year включительно (в том числе на парах прошлых лет: одна система координат); позже — NaN.
    Параметры энкодера и главные компоненты известны на 01.01.year (бары и z дат < year), вход даты — бары ≤ даты. window = 1 — конфигурация 3. МУТАНТ «z пары от энкодера её года» подменяет эту функцию."""
    a = load(STA.directory() / STA.name(kind, enc_year(year), seed, window))
    days = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
    pos = index.get_indexer(days)
    use = (pos >= 0) & (days < pd.Timestamp(year + 1, 1, 1))
    z = np.full((len(index), FM.DZ), np.nan)
    z[pos[use]] = a["z"][use]
    return z


def z_by_year(kind: str, seed: int, index: pd.DatetimeIndex, load=STA.load, years=STA.YEARS, window: int = STA.WINDOW) -> dict:
    """{год Y: (n, 4) — z энкодера года Y}: переоценка в году Y берёт матрицу года Y целиком."""
    return {y: z_of_year(kind, seed, index, y, load, window) for y in years}


@contextlib.contextmanager
def year_z_patch(zmap: dict):
    """На время вызова `WF._refit` получает z = zmap[год d0] (энкодер года переоценки для ВСЕХ обучающих строк и прогнозов блока); года без энкодера — модели нет (G = 0). Прежняя функция ВОССТАНАВЛИВАЕТСЯ."""
    orig = WF._refit

    def wrapped(tab_cols, Zg, z, spec, d0, lo, b, seeds, grid):
        zy = zmap.get(d0.year)
        return orig(tab_cols, Zg, zy, spec, d0, lo, b, seeds, grid) if zy is not None else (None, 0)
    WF._refit = wrapped
    try:
        yield
    finally:
        WF._refit = orig


def walk(tab: pd.DataFrame, Zg: np.ndarray, z, spec: dict, **kw) -> dict:
    """`WF.walk_forward`; z — словарь {год: (n, 4)} (боевой путь, `year_z_patch`) или один массив (синтетика тестов)."""
    if not isinstance(z, dict):
        return WF.walk_forward(tab, Zg, z, spec, **kw)
    with year_z_patch(z):
        return WF.walk_forward(tab, Zg, np.zeros((len(tab), FM.DZ)), spec, **kw)


def with_warm(tab: pd.DataFrame, warm: pd.DataFrame | None) -> pd.DataFrame:
    """Таблица с прогревом H (2015-07-07 … до первой даты tab; K2, K3 = NaN, корректору 1–3 не нужны) перед `tab`; warm = None — без прогрева."""
    return tab if warm is None else pd.concat([warm, tab])


def wf_inputs(tab: pd.DataFrame) -> tuple:
    H, rv = tab["H"].to_numpy(float), tab["rv_next"].to_numpy(float)
    return H, rv, tab["t_plus"].to_numpy("datetime64[ns]"), np.log(np.sqrt(tab["K2"].to_numpy(float) * tab["K3"].to_numpy(float)) / H)


def run_kind(kind: str, tab: pd.DataFrame, load=STA.load, seeds=FT.SEEDS, years=STA.YEARS, window: int = STA.WINDOW, warm: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    """-> таблица (d, t_plus, rv, H, g, F) с первой оцениваемой даты; g = среднее по seeds, F = H·exp(g). Переоценка в году Y — z энкодера года Y (`z_by_year`); warm — прогрев H (пары без K2/K3)."""
    full = with_warm(tab, warm)
    Zg, Gs = np.zeros((len(full), 1)), []
    with lin5_patch():
        for s in seeds:
            Gs.append(walk(full, Zg, z_by_year(kind, s, full.index, load, years, window), SPEC, seeds=(s,), grid=GRID, **kw)["G"].to_numpy(float))
    g = np.mean(np.stack(Gs), axis=0)
    out = pd.DataFrame({"d": full.index, "t_plus": full["t_plus"].to_numpy(), "rv": full["rv_next"].to_numpy(float), "H": full["H"].to_numpy(float), "g": g, "F": full["H"].to_numpy(float) * np.exp(g)})
    return out[out["F"].notna()].reset_index(drop=True)


def refit_block(kind: str, tab: pd.DataFrame, a: int, load=STA.load, seeds=FT.SEEDS, years=STA.YEARS, window: int = STA.WINDOW, warm: pd.DataFrame | None = None) -> np.ndarray:
    """Одна переоценка (блок месяца с позиции a в `tab`): g по среднему 5 seeds — тот же вызов `_refit`, что внутри walk_forward."""
    full = with_warm(tab, warm)
    a += len(full) - len(tab)
    pos = W6.refit_positions(full.index, WF.FIRST_REFIT)
    b = (pos + [len(full)])[pos.index(a) + 1]
    Gs = []
    with lin5_patch():
        for s in seeds:
            zy = z_of_year(kind, s, full.index, full.index[a].year, load, window)
            res, _ = WF._refit(wf_inputs(full), np.zeros((len(full), 1)), zy, SPEC, full.index[a], a, b, (s,), GRID)
            Gs.append(res["G"] if res else np.zeros(b - a))
    return np.mean(np.stack(Gs), axis=0)


def qlike_choice(df: pd.DataFrame) -> tuple:
    """Средняя QLIKE (определение `models_v2.metrics.qlike`) на датах целей CHOICE_WIN. -> (qlike, n)."""
    y = pd.DatetimeIndex(df["t_plus"]).year.to_numpy()
    m = (y >= CHOICE_WIN[0]) & (y <= CHOICE_WIN[1]) & np.isfinite(df["F"].to_numpy(float))
    return float(metrics.qlike(df["rv"].to_numpy(float)[m], df["F"].to_numpy(float)[m]).mean()), int(m.sum())


def choose(dfs: dict) -> dict:
    """dfs: {kind: таблица}. Лучший — меньшая QLIKE на CHOICE_WIN при одинаковом наборе дат; ничья — TCN."""
    ok = None
    for df in dfs.values():
        y = pd.DatetimeIndex(df["t_plus"]).year.to_numpy()
        s = set(df["d"][(y >= CHOICE_WIN[0]) & (y <= CHOICE_WIN[1]) & np.isfinite(df["F"])])
        ok = s if ok is None else ok & s
    sub = {k: df[df["d"].isin(ok)] for k, df in dfs.items()}
    q = {k: qlike_choice(sub[k]) for k in sub}
    best = min(KINDS, key=lambda k: (q[k][0], KINDS.index(k)))
    return dict(choice=best, qlike={k: q[k][0] for k in q}, n_dates=q[best][1], window="t⁺ %d–%d" % CHOICE_WIN)
