"""М0: общие пути и загрузчики (данные основной копии — только чтение)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
DF = MAIN / "moex_discovery/data/dataset_final/02_external"
RAW = MAIN / "moex_discovery/data/external"
MV = MAIN / "moex_volatility"
PREP = MV / "data/prepared"
ISS_IMOEX = MAIN / ".claude/worktrees/agent-a51b95b5ae61fb6bc/data/iss_check/index_d/IMOEX.parquet"
OUT = ROOT / "docs/otchety/m0"
REGEN = ROOT / "data/regen"
PARTS = ["train", "val", "test"] + [f"walkforward_{y}" for y in range(2020, 2027)]

INDICES = ["MOEXMM", "IMOEX", "MOEXTL", "MOEXEU", "CNYRUB_TOM", "RVI", "MOEXCH", "MOEXFN", "MOEXCN", "RGBI", "MOEXOG"]
YAHOO = {"brent": "BZ_F", "vix": "VIX", "sp500": "GSPC", "gold": "GC_F", "dxy": "DX_Y_NYB", "eem": "EEM",
         "wti": "CL_F", "tbill_3m": "IRX", "treasury_10y": "TNX", "treasury_5y": "FVX"}
MACRO_RAW = {"cpi_ru": "cpi_russia", "cpi_us": "cpi_usa", "unemp_us": "unemployment_usa", "m2_ru": "m2_russia",
             "pmi_ru": "pmi_russia", "key_rate_cbr": "key_rate_cbr", "fed_rate": "fed_rate", "si_fut": "si_fut_daily"}
IDX_SUF = ["lag1", "ret1", "ret5", "ret22", "ma5", "ma22", "vol22"]


def feature_cols():
    return json.load(open(PREP / "config.json"))["feature_cols"]


def load_prepared():
    """Матрица в формате walk_forward_full.py: train+val+test+walkforward_* (NaN уже = медианы train)."""
    return pd.concat([pd.read_parquet(PREP / f"{p}.parquet") for p in PARTS], ignore_index=True)


def load_features_all():
    """Матрица ДО заполнения медианами (moex_volatility/data/features/features_all.parquet)."""
    f = pd.read_parquet(MV / "data/features/features_all.parquet")
    f["date"] = pd.to_datetime(f["date"])
    return f


def calendar():
    """2 874 торговых дня MOEX (общий календарь dataset_final)."""
    y = pd.read_parquet(DF / "yahoo.parquet")
    return pd.DatetimeIndex(pd.to_datetime(y["date"])).sort_values()


def yahoo_raw(col):
    r = pd.read_parquet(RAW / f"yahoo/raw/{YAHOO[col]}.parquet")[["date", "close"]]
    r["date"] = pd.to_datetime(r["date"])
    return r.dropna().sort_values("date")


def macro_raw(col):
    r = pd.read_parquet(RAW / f"macro/raw/{MACRO_RAW[col]}.parquet")
    r["date"] = pd.to_datetime(r["date"])
    v = "close" if col == "si_fut" else "value"
    return r[["date", v]].rename(columns={v: "value"}).dropna().sort_values("date")


def asof(cal, obs, strict):
    """Значение ряда obs(date, value) на каждый день cal: последнее с датой < t (strict) или <= t."""
    left = pd.DataFrame({"date": cal})
    o = obs.rename(columns={obs.columns[1]: "v"}).sort_values("date")
    m = pd.merge_asof(left, o, on="date", allow_exact_matches=not strict)
    return pd.Series(m["v"].values, index=cal)


def index_bars(t):
    d = pd.read_parquet(DF / f"candles_10m/{t}.parquet")
    d["begin"] = pd.to_datetime(d["begin"])
    d["date"] = d["begin"].dt.normalize()
    return d.sort_values("begin")


def rv_by_day(bars, cut=None):
    """Формула create_external_final.py:190-221: RV = sum ln(c_i/c_{i-1})^2 внутри дня, close = последний бар;
    дни с < 10 барами пропускаются. cut — последний допустимый begin (time)."""
    b = bars if cut is None else bars[bars["begin"].dt.time <= cut]
    g = b.groupby("date")["close"]
    rv = g.apply(lambda c: float(np.sum(np.diff(np.log(c.values)) ** 2)))
    out = pd.DataFrame({"rv_daily": rv, "close": g.last(), "n_bars": g.size()})
    return out[out["n_bars"] >= 10]


def rv_moex_pit(cut=None, tickers=("CNYRUB_TOM", "RVI", "RGBI")):
    """rv_moex.parquet, где для tickers RV/close пересчитаны по барам с begin <= cut (по умолчанию 18:50).
    Строки source='filled' и дни, где после среза < 10 баров, не трогаем."""
    import datetime as _dt
    cut = cut or _dt.time(18, 50)
    r = pd.read_parquet(DF / "rv_moex.parquet")
    r["date"] = pd.to_datetime(r["date"])
    for t in tickers:
        c = rv_by_day(index_bars(t), cut)
        m = (r.ticker == t) & (r.source != "filled") & r.date.isin(c.index)
        d = r.loc[m, "date"]
        r.loc[m, "rv_daily"] = c.loc[d, "rv_daily"].values
        r.loc[m, "close"] = c.loc[d, "close"].values
        r.loc[m, "n_bars"] = c.loc[d, "n_bars"].values
        r.loc[m, "rv_annualized"] = np.sqrt(r.loc[m, "rv_daily"] * 252)
    return r


def qlike(y, f):
    f = np.clip(f, 1e-10, None)
    return float(np.mean(y / f - np.log(y / f) - 1))


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
