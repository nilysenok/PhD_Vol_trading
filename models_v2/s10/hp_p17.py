"""P17, G16 (Ж-220; У-19…У-21): обучающая вселенная поправки и K4/K5 — 17 утверждённых бумаг (P.MAIN17: M16 + SBER); 11 дополнительных (E11) — только перенос зафиксированной модели.
Три правила (крючки — мутации подменяют их в тестах, боевой путь вызывает только их):
  `cross_members` — поперечные признаки дня t (m_med, m_disp, m_share_up, c_rank, c_dev_med) считаются по бумагам из 17, у которых есть день t (R4); бумага вне 17 получает ранг и отклонение ОТНОСИТЕЛЬНО 17;
  `train_exclude` — бумаги вне 17 в обучение (пары, блок ранней остановки) не входят: `hp_engine.walk_forward(exclude=...)`, прогноз строится для всех строк панели;
  `sel_mask` — выбор параметров года Y (глубина, c, λ, κ, a, b, w) — только по строкам из 17 (У-20).
Существующие модули не менялись: проходы и выбор вызывают движок P1b с параметром `exclude` и подвыборкой строк."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_ctrl as CT, hp_engine as E, hp_feat as F, hp_hyb_core as C, hp_paths as P, hp_panel as PN, hp_sync as SY

TRAIN = list(P.MAIN17)
ENGINE_P17 = P.OUT / "engine_panel_p17.parquet"
CROSS_COLS = ["m_med", "m_disp", "m_share_up", "c_rank", "c_dev_med"]


# --- крючки правил ---
def cross_members(tickers) -> list:
    """Бумаги, по которым считаются поперечные величины дня (У-19): только из 17."""
    return [t for t in tickers if t in TRAIN]


def train_exclude(tickers) -> list:
    """Бумаги, исключаемые из обучения (У-19): все вне 17."""
    return sorted(t for t in set(tickers) if t not in TRAIN)


def sel_mask(tickers: np.ndarray) -> np.ndarray:
    """Строки, по которым выбираются параметры года (У-20): бумаги из 17."""
    return np.isin(np.asarray(tickers), TRAIN)


# --- поперечные признаки по 17 ---
def rank_outside(lnrel: pd.DataFrame, med: pd.Series, x: pd.Series) -> tuple:
    """Ранг и отклонение бумаги вне 17 в день t относительно 17: (число бумаг 17 с меньшим значением + 1)/(n + 1) — как место среди n + 1; x − медиана 17.
    Нет x, нет дня у 17 или n < MIN_CROSS — NaN."""
    L = lnrel.reindex(x.index)
    n = L.notna().sum(axis=1)
    less = L.lt(x, axis=0).sum(axis=1)
    ok = x.notna() & (n >= SY.MIN_CROSS)
    return ((less + 1.0) / (n + 1.0)).where(ok), (x - med.reindex(x.index)).where(ok)


def features_all(daily: dict, imo: pd.DataFrame, k1: dict) -> dict:
    """{бумага: таблица X/M/C (сырые + z)} на её датах; как hp_panel.features_all, но поперечные величины — по бумагам 17 (У-19)."""
    mem = cross_members(list(daily))
    lnrel, up = PN.cross_frames({t: daily[t] for t in mem})
    cs = SY.cross_stats(lnrel, up)
    rank, dev = SY.cross_rank(lnrel, cs["m_med"])
    out = {}
    for t, d in daily.items():
        f = pd.concat([F.individual(d, k1[t]), F.market_own(d, imo), F.structural(d)], axis=1)
        for c in ("m_med", "m_disp", "m_share_up"):
            f[c] = cs[c].reindex(d.index)
        if t in mem:
            f["c_rank"], f["c_dev_med"] = rank[t].reindex(d.index), dev[t].reindex(d.index)
        else:
            f["c_rank"], f["c_dev_med"] = rank_outside(lnrel, cs["m_med"], F.ln(d["rv"]) - F.ln(d["rv_m"]))
        out[t] = F.add_z(f[F.X_COLS + F.M_COLS + F.C_COLS])
    return out


def build_panel(daily: dict, imo: pd.DataFrame) -> pd.DataFrame:
    """Панель движка (28 бумаг), K1 и все прочие колонки — как `hp_ctrl.build_panel_ext`; поперечные признаки — по 17."""
    k1 = CT.k1_ext(daily)
    feats = features_all(daily, imo, k1)
    parts = [PN.ticker_rows(t, daily[t], k1[t], feats[t], imo["rv"]) for t in P.TICKERS]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def panel_diff(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """Сверка с `engine_panel`: на общих строках числовые колонки вне поперечных и их z — побитно; поперечные — отличаются."""
    m = new.merge(old, on=["ticker", "date"], suffixes=("", "_o"))
    cross = set(CROSS_COLS) | {"z_" + c for c in CROSS_COLS}
    same, moved = [], []
    for c in old.columns:
        if c in ("ticker", "date") or c + "_o" not in m.columns or old[c].dtype.kind != "f":
            continue
        a, b = m[c].to_numpy(float), m[c + "_o"].to_numpy(float)
        eq = bool(np.array_equal(a, b, equal_nan=True))
        if c in cross:
            moved.append((c, eq))
        elif not eq:
            same.append(c)
    return {"n_common": len(m), "n_new": len(new), "n_old": len(old), "differs_noncross": same, "cross_unchanged": [c for c, e in moved if e], "cross_n": len(moved)}


# --- проходы walk-forward и выбор на 17 ---
def cached_runs(module: str, name: str, panel: pd.DataFrame, feats: list, base, panel_sha: str, loss: str = "qlike", hp=None, depths=E.DEPTHS, nthread: int = 4, log=None) -> dict:
    """{глубина: walk_forward(exclude = вне 17)}: кэш `hyb_run_p17_*` (ключ — как у ядра P2 + хэш входов + список исключённых, Д5)."""
    excl = train_exclude(panel["ticker"].unique())
    extra = "p17|%s|%s" % (",".join(excl), C.data_sha(panel, feats, base))
    out = {}
    for dp in depths:
        path = P.OUT / ("hyb_run_p17_%s_%s_d%d_%s.parquet" % (module, name, dp, C.run_key(module, name, dp, loss, feats, hp, panel_sha, extra)))
        if path.exists():
            out[dp] = pd.read_parquet(path)
            continue
        out[dp] = E.walk_forward(panel, feats, base, loss, module, dp, exclude=excl, hp=hp, nthread=nthread, log=log)
        out[dp].to_parquet(path)
    return out


def finalize(panel: pd.DataFrame, base, runs: dict, loss: str = "qlike", lams=E.LAMS) -> tuple:
    """Как `E.finalize`, но (глубина, c, λ) на год Y выбираются по строкам 17 (У-20); применяются ко всем строкам панели. -> (прогнозы, таблица выбора)."""
    base = np.asarray(base, float)
    sel = sel_mask(panel["ticker"].to_numpy())
    ps, bs, rs = panel[sel], base[sel], {d: r[sel] for d, r in runs.items()}
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    parts, ch = [], []
    for Y in sorted(set(yr[yr >= E.FIRST_YEAR])):
        s = E.choose(ps, bs, rs, Y, loss, E.CS, lams)
        ch.append({"year": int(Y), **s})
        one, _ = E.finalize(panel, base, {s["depth"]: runs[s["depth"]]}, loss, years=[Y], cs=(s["c"],), lams=(s["lam"],), default=(s["depth"], s["c"], s["lam"]))   # применение выбранного набора
        parts.append(one)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(ch)
