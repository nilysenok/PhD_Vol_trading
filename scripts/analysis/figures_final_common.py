"""Общий загрузчик, стиль и статистика для figures_final*.py (рисунки глав 3-4).

Все числа берутся из рядов/результатов; здесь только оформление и формулы.
Бутстрап - схема moex_volatility/scripts/fix_all_bootstrap_ce.py: seed 42, iid
randint(0, n, n), одни индексы для обеих серий, 10 000 реплик, p = доля дельт <= 0.
"""
import re
import warnings
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
VOL = ROOT / "moex_volatility"
BPY, N_BOOT, SEED = 252, 10_000, 42
BLUE1, BLUE2, BLUE3, GREY, RED, GREEN = "#2E75B6", "#4472C4", "#A5C8E1", "#808080", "#C0504D", "#70AD47"
SN = ["S1", "S2", "S3", "S4", "S5", "S6"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    "axes.unicode_minus": False,
})


class Ctx:
    """Пути и накопитель сверки рисунок = таблица."""

    def __init__(self, series, out, wf=None, t2019=None, keyrate=None, features=None):
        self.series = Path(series)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.wf = Path(wf) if wf else VOL / "results/final/data/predictions_walkforward"
        self.t2019 = Path(t2019) if t2019 else VOL / "data/predictions/test_2019"
        self.keyrate = Path(keyrate) if keyrate else ROOT / "moex_discovery/data/external/macro/raw/key_rate_cbr.parquet"
        self.features = Path(features) if features else VOL / "data/features/features_all.parquet"
        self.checks, self.status = [], []

    def chk(self, fig, item, value, ref, src, tol=0.01):
        self.checks.append(dict(fig=fig, item=item, figure_value=float(value), ref=float(ref),
                                diff=float(value - ref), flag=bool(abs(value - ref) > tol), src=src))

    def skip(self, fig, why):
        self.status.append((fig, "не построен", why))
        print(f"  {fig}: не строится - {why}")

    def ok(self, fig, note=""):
        self.status.append((fig, "построен", note))


def dec(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


def comma_axes(*axes, axis="both"):
    f = FuncFormatter(lambda v, p: f"{v:.10g}".replace(".", ","))
    for ax in axes:
        if axis in ("both", "x"):
            ax.xaxis.set_major_formatter(f)
        if axis in ("both", "y"):
            ax.yaxis.set_major_formatter(f)


def save(ctx, fig, name, df):
    fig.savefig(ctx.out / f"{name}.png")
    plt.close(fig)
    df.to_csv(ctx.out / f"{name}.csv", index=False)
    print(f"  сохранён {name}")


# ---------- загрузка ----------
def load_msp(ctx, ap, comm="inst"):
    return pd.read_csv(ctx.series / f"MSP_{ap}_{comm}.csv", parse_dates=["date"]).set_index("date")


def load_strat(ctx, i, ap, comm="inst"):
    return pd.read_csv(ctx.series / f"S{i}_{ap}_{comm}.csv", parse_dates=["date"]).set_index("date")


def comm_rates(ctx):
    p = ctx.series.parent / "perstrat_results.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p).drop_duplicates("comm_label")
    return dict(zip(d.comm_label, d.comm))


def key_rate_daily(ctx):
    """Ключевая ставка, % годовых, ступени forward-fill по календарным дням."""
    k = pd.read_parquet(ctx.keyrate)
    k["date"] = pd.to_datetime(k["date"])
    s = k.sort_values("date").set_index("date")["value"]
    return s.resample("D").last().ffill()


def imoex_ret(ctx):
    f = pd.read_parquet(ctx.features, columns=["date", "close_IMOEX"]).drop_duplicates("date")
    f["date"] = pd.to_datetime(f["date"])
    c = f.sort_values("date").set_index("date")["close_IMOEX"]
    return c.pct_change()


def econ_costs():
    """Затраты внедрения (млн руб.) из scripts/stage8_economic.py: COSTS_YEAR1/2 в тыс. руб."""
    t = (VOL / "scripts/stage8_economic.py").read_text()

    def g(n):
        return float(re.search(rf"{n}\s*=\s*(\d+)", t).group(1)) / 1000.0
    return g("COSTS_YEAR1"), g("COSTS_YEAR2")


# ---------- статистика ----------
def sharpe(x):
    s = np.std(x, ddof=1)
    return np.mean(x) * sqrt(BPY) / s if s > 1e-12 else 0.0


def ce(r, gamma):
    w = np.maximum(1.0 + r, 1e-10)
    if gamma == 1:
        return np.exp(np.mean(np.log(w))) - 1
    return np.mean(w ** (1 - gamma)) ** (1 / (1 - gamma)) - 1


def dce(a, b, gamma):
    """Delta CE (a против b), % годовых."""
    return (ce(a, gamma) - ce(b, gamma)) * BPY * 100


def boot_deltas(f, a, b, n_boot=N_BOOT):
    np.random.seed(SEED)
    n = len(a)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        d[i] = f(a[idx]) - f(b[idx])
    return d


def mdd_pct(r):
    """Максимальная просадка по накопленной сумме доходностей, % (как fix_all_bootstrap_ce)."""
    cum = np.cumsum(r)
    return float(((np.maximum.accumulate(cum) - cum).max()) * 100)
