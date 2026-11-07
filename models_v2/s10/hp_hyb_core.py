"""P2, G9/G10 (R6–R8, У-6…У-11): общее ядро гибридов HAR-J + панельная поправка поверх движка P1b (hp_engine).
Закрытые пары (R4): пара (бумага i, s) годна к оценке на дату d, если s⁺ <= d; окно «последние W закрытых пар бумаги» строит `trail_index` (крючок `closed_counts` — мутации «закрытые + 1»).
Проходы walk-forward кэшируются в `hyb_run_*.parquet`; ключ кэша = хэш входов, параметров и кода (Д5). Гибриды не обучают самостоятельных моделей (Ж-218)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_engine as E, hp_feat as F, hp_paths as P

Z28 = ["z_" + c for c in F.X_COLS + F.M_COLS + F.C_COLS]      # У-10: A1, B, C, D
ZX = ["z_" + c for c in F.X_COLS]                             # А3: только группа X
SEEDS_BAG = (43, 44, 45, 46)                                  # D2: копия seed 42 — сам A1
KEY = ["ticker", "date"]


def closed_counts(tp_sorted: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Сколько пар бумаги (t⁺ по возрастанию) закрыто к каждой дате d: t⁺ <= d. Мутант «закрытые + 1» подменяет эту функцию."""
    return np.searchsorted(tp_sorted, dates, side="right")


def trail_index(panel: pd.DataFrame, valid: np.ndarray, win: int) -> tuple:
    """-> (order, lo, hi): окно строки r = order[lo[r]:hi[r]] — последние win закрытых пар ТОЙ ЖЕ бумаги (valid[j], t⁺_j <= d_r)."""
    tk, tp, dt = panel["ticker"].to_numpy(), panel["t_plus"].to_numpy("datetime64[ns]"), panel["date"].to_numpy("datetime64[ns]")
    lo, hi, order, off = np.zeros(len(panel), int), np.zeros(len(panel), int), [], 0
    for t in pd.unique(tk):
        rows = np.flatnonzero(tk == t)
        v = rows[valid[rows]]
        v = v[np.argsort(tp[v], kind="stable")]
        k = closed_counts(tp[v], dt[rows])
        hi[rows], lo[rows] = off + k, off + np.maximum(k - win, 0)
        order.append(v)
        off += len(v)
    return (np.concatenate(order) if order else np.array([], int)), lo, hi


def trail_agg(vals: np.ndarray, idx: tuple, how: str = "mean") -> tuple:
    """(агрегат по окну строки, число пар в окне); пустое окно — NaN. Суммы — по самому окну (без накопленных сумм: будущее другой бумаги не меняет младшие биты)."""
    order, lo, hi = idx
    vs, out = np.asarray(vals, float)[order], np.full(len(lo), np.nan)
    for r in np.flatnonzero(hi > lo):
        w = vs[lo[r]:hi[r]]
        out[r] = w.mean() if how == "mean" else w.sum()
    return out, hi - lo


def align(panel: pd.DataFrame, fc: pd.DataFrame, col: str) -> np.ndarray:
    """Колонка таблицы прогнозов `fc` (ticker, date, ...) в порядке строк панели; нет строки — NaN."""
    m = panel[KEY].merge(fc[KEY + [col]], on=KEY, how="left")
    return m[col].to_numpy(float)


def mean_q(rv, f, tk, mask) -> float:
    """Средний по бумагам QLIKE на строках mask (нефинитные и f <= 0 пропускаются)."""
    ok = mask & np.isfinite(f) & (f > 0) & np.isfinite(rv) & (rv > 0)
    return E.mean_by_ticker(E.qlike_rows(rv[ok], f[ok]), tk[ok]) if ok.any() else float("nan")


def pick_min(cands: list, default: dict) -> dict:
    """cands — [(crit, params)]; минимум конечного crit (ничья — первый). Пусто — default (R8: первый год без закрытой валидации)."""
    fin = [(q, p) for q, p in cands if np.isfinite(q)]
    if not fin:
        return {**default, "crit": float("nan"), "source": "default"}
    q, p = min(fin, key=lambda x: x[0])
    return {**p, "crit": float(q), "source": "closed"}


def data_sha(panel: pd.DataFrame, feats: list, base) -> str:
    """Хэш входов прохода: значения признаков, цели RV_{t⁺}, базы, дат (Д5: подмена любого входа — другой ключ)."""
    h = hashlib.sha256()
    for a in (panel[feats].to_numpy(float), panel["rv_tplus"].to_numpy(float), np.asarray(base, float), panel["date"].to_numpy("datetime64[ns]").astype("int64")):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def run_key(module: str, name: str, depth: int, loss: str, feats: list, hp, panel_sha: str, extra: str = "") -> str:
    """Ключ кэша прохода: модуль, имя, глубина, функция потерь, признаки, HP, SHA панели, хэш значений входов (`data_sha`) и SHA кода движка (Д5)."""
    code = P.sha16(E.__file__)
    raw = json.dumps([module, name, depth, loss, list(feats), {**E.HP, **(hp or {})}, panel_sha, extra, code], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def cached_runs(module: str, name: str, panel: pd.DataFrame, feats: list, base, panel_sha: str, loss: str = "qlike", hp=None, depths=E.DEPTHS, nthread: int = 4, log=None) -> dict:
    """{глубина: walk_forward}: из кэша `hyb_run_<module>_<name>_d<k>_<ключ>.parquet` или счёт и запись. Модули XGB и LGBM не смешиваются (имя файла и ключ)."""
    out, ds = {}, data_sha(panel, feats, base)
    for dp in depths:
        path = P.OUT / ("hyb_run_%s_%s_d%d_%s.parquet" % (module, name, dp, run_key(module, name, dp, loss, feats, hp, panel_sha, ds)))
        if path.exists():
            out[dp] = pd.read_parquet(path)
            continue
        out[dp] = E.walk_forward(panel, feats, base, loss, module, dp, hp=hp, nthread=nthread, log=log)
        out[dp].to_parquet(path)
    return out


def pick_by_depth(runs: dict, col: str, depth: np.ndarray) -> np.ndarray:
    """Значение колонки прохода той глубины, которая выбрана для строки (depth[r])."""
    res = np.full(len(depth), np.nan)
    for dp, r in runs.items():
        m = depth == dp
        res[m] = r[col].to_numpy(float)[m]
    return res


def apply_lam(base, g, c, lam) -> np.ndarray:
    """σ̂² = base·exp(λ·clip(g, ±c)) со строчными c, λ (нет g — поправка 0); λ = 0 даёт base побитно (base·1.0)."""
    gc = np.clip(np.nan_to_num(np.asarray(g, float), nan=0.0), -np.asarray(c, float), np.asarray(c, float))
    return np.asarray(base, float) * np.exp(np.asarray(lam, float) * gc)
