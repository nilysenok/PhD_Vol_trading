"""Накопленная разность ежедневных QLIKE-потерь Б-2 − HAR-J(б): таблица и PNG."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def table(df: pd.DataFrame, sign: int = 1) -> pd.DataFrame:
    """sign=-1 — мутация (перепутанный знак)."""
    dl = sign * (df["qlike_b2"] - df["qlike_harj"])
    return pd.DataFrame({"target_date": df["target_date"], "dloss_b2_minus_harj": dl, "cum_b2_minus_harj": dl.cumsum()})


def plot(t: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t["target_date"], t["cum_b2_minus_harj"], lw=1.2, color="#1f4e79")
    ax.axhline(0, color="k", lw=0.6)
    for y in range(2018, 2027):
        ax.axvline(pd.Timestamp("%d-01-01" % y), color="grey", lw=0.4, ls=":")
    ax.set_title("SBER h=1: накопленная Σ(QLIKE Б-2 − QLIKE HAR-J(б)); вниз — Б-2 выигрывает")
    ax.set_ylabel("накопленная разность потерь")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
