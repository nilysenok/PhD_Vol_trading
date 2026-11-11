"""S-C-IMPL-1 (Ж-254 пп. 8–12; `docs/otchety/harj_struct/S_C_SPEC.md`, разд. 1.1–1.3, 2, 7): память компонент RV SBER (common c, specific p, uncovered u) — вход компонентного гибрида C.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Читает только `data/regen/harj_struct/C/C_sessions.parquet` (k_factors = 1, sha256 c71410531eec3275…) и бары SBER через `fwd_src.Src`; прогнозы K1–K3, cfg8, X1, W5 не трогает.
Публичное API:
- `SHA_C`, `C_PARQ`, `CM_DIR`, `FEAT_PATH` (= CM/feat.parquet), `SHIFT = 252` (сессий, S-CKL), `COLS_REAL`, `COLS_LAG` (префикс `L_`).
- `load_c(path=C_PARQ, check_sha=True) -> DataFrame`: строки k_factors = 1, индекс d (3124 сессии); отказ (SystemExit) при несовпадении sha.
- `components(tab) -> dict(c, p, u, rv, seg=None)`: массивы rv_common, rv_specific, rv_uncovered, rv; до первой покрытой сессии (n_cov > 0, 2014-08-27) c, p, u = NaN, а не 0 (Д4).
- `shift_decomp(c, p, u, rv, lag=SHIFT) -> (c~, p~, u~)`: q_s = c_{s−lag}/(c_{s−lag} + p_{s−lag}); c~ = (rv − u)·q, p~ = (rv − u)(1 − q), u~ = u; NaN, пока s − lag < первой покрытой (только прошлое).
- `harj_hat(x, seg, j, rows=None) -> ndarray`: Ĥ_X,d = `hp_ctrl.harj_at` над `har.make_panel` ряда уровней x (окна 5 и 22 строки сегмента — `hv2_dec.wmean`, МНК по парам t + 1 ≤ d, ≥ 252 пар); rows — позиции (остальные NaN).
- `memory(c, p, u, seg, j, rows=None) -> DataFrame`: колонки c, c_w, c_m, p, p_w, p_m, u, u_m, sc, sp, su (ŝ_X = X_m / (c_m + p_m + u_m)), Hc, Hp.
- `build(cut=LAST_SESSION, src=None, rows=None, tab=None) -> DataFrame`: индекс d (3124 сессии); rv, seg, j (из `daily.build_daily`), COLS_REAL и те же колонки с префиксом `L_` для сдвинутого разложения (S-CKL).
- `load_feat() -> DataFrame`: `feat.parquet`, если файл есть, иначе `build()`; `write()` — запись feat.parquet (только после регистрации S-C, разд. 6).
- `dm_dates()`: даты d окна D_M (2198, `x1_w3_core.load`).
Запуск: `python3 -W ignore -m models_v2.s18.cm_feat [--check | --check-k7 | --check-pit | --write]` (проверки только читают; код — `cm_feat_chk.py`)."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2 import har as H
from models_v2.s9 import hv2_dec as DC
from models_v2.s10 import hp_ctrl as HC
from models_v2.s17.fwd_src import LAST_SESSION, ROOT, Src

C_PARQ = ROOT / "data/regen/harj_struct/C/C_sessions.parquet"
CM_DIR = ROOT / "data/regen/harj_struct/CM"
FEAT_PATH = CM_DIR / "feat.parquet"
SHA_C, SHIFT = "c71410531eec3275", 252
BASE = ["c", "c_w", "c_m", "p", "p_w", "p_m", "u", "u_m", "sc", "sp", "su", "Hc", "Hp"]
COLS_REAL, COLS_LAG = BASE, ["L_" + k for k in BASE]


def load_c(path=C_PARQ, check_sha: bool = True) -> pd.DataFrame:
    if check_sha and not hashlib.sha256(path.read_bytes()).hexdigest().startswith(SHA_C):
        raise SystemExit("C_sessions.parquet: sha256 не c71410531eec3275 — вход изменён")
    t = pd.read_parquet(path)
    return t[t["k_factors"] == 1].sort_values("d").set_index("d")


def components(tab: pd.DataFrame) -> dict:
    first = int(np.argmax(tab["n_cov"].to_numpy() > 0))
    pre = np.arange(len(tab)) < first
    g = lambda k: np.where(pre, np.nan, tab[k].to_numpy(float))
    return dict(c=g("rv_common"), p=g("rv_specific"), u=g("rv_uncovered"), rv=tab["rv"].to_numpy(float))


def shift_decomp(c, p, u, rv, lag: int = SHIFT) -> tuple:
    c, p, u, rv = (np.asarray(a, float) for a in (c, p, u, rv))
    q = np.full(len(c), np.nan)
    n = len(c) - lag
    q[lag:] = c[:n] / (c[:n] + p[:n])
    rest = rv - u
    return rest * q, rest * (1.0 - q), np.where(np.isfinite(q), u, np.nan)


def harj_hat(x, seg, j, rows=None) -> np.ndarray:
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        panel = H.make_panel(pd.DataFrame({"rv": x, "rv_w": DC.wmean(x, seg, 5), "rv_m": DC.wmean(x, seg, 22), "j": np.asarray(j, float), "seg": seg}))
    out = np.full(len(x), np.nan)
    for i in (range(len(x)) if rows is None else rows):
        out[i] = HC.harj_at(panel, int(i))
    return out


def memory(c, p, u, seg, j, rows=None) -> pd.DataFrame:
    seg, o = np.asarray(seg), {}
    for k, x in (("c", c), ("p", p), ("u", u)):
        x = np.asarray(x, float)
        o[k], o[k + "_m"] = x, DC.wmean(x, seg, 22)
        if k != "u":
            o[k + "_w"] = DC.wmean(x, seg, 5)
    tot = o["c_m"] + o["p_m"] + o["u_m"]
    for k in "cpu":
        o["s" + k] = o[k + "_m"] / tot
    o["Hc"], o["Hp"] = (harj_hat(o[k], seg, j, rows) for k in "cp")
    return pd.DataFrame({k: o[k] for k in BASE})


def build(cut=LAST_SESSION, src: Src | None = None, rows=None, tab: pd.DataFrame | None = None) -> pd.DataFrame:
    src = src or Src()
    tab = load_c() if tab is None else tab
    dd, _ = D.build_daily(src.bars("SBER", cut), index_days=src.index_days(cut))
    if len(dd) != len(tab) or not (dd.index == tab.index).all():
        raise SystemExit("сетка C не совпадает с build_daily")
    k = components(tab)
    seg, j = dd["seg"].to_numpy(), dd["j"].to_numpy(float)
    real = memory(k["c"], k["p"], k["u"], seg, j, rows)
    ct, pt, ut = shift_decomp(k["c"], k["p"], k["u"], k["rv"])
    lag = memory(ct, pt, ut, seg, j, rows).rename(columns=dict(zip(BASE, COLS_LAG)))
    head = pd.DataFrame(dict(rv=k["rv"], seg=seg, j=j))
    return pd.concat([head, real, lag], axis=1).set_axis(tab.index)


def dm_dates() -> pd.DatetimeIndex:
    from models_v2.s16 import x1_w3_core as C
    return pd.DatetimeIndex(C.load()["d"])


def load_feat() -> pd.DataFrame:
    return pd.read_parquet(FEAT_PATH) if FEAT_PATH.exists() else build()


def write() -> pd.DataFrame:
    f = build()
    CM_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FEAT_PATH.with_suffix(".tmp")
    f.to_parquet(tmp)
    tmp.replace(FEAT_PATH)
    return f


if __name__ == "__main__":
    from models_v2.s18 import cm_feat_chk
    sys.exit(cm_feat_chk.main(sys.argv[1:]))
