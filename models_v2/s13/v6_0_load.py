"""V6-0, G1/G2: чтение СОХРАНЁННЫХ прогнозов SBER (h = 1) прежних программ. Ничего не обучает и не пишет. Единый кадр по датам d = общие даты H ∩ K2 ∩ K3 (2271), контроли K1, K2, K3, M, K8, D2 и все ряды
прежних гибридов (Ж-210/211/212/214, панель Ж-217–220, V4, V5, sber_h1_deep_v1) в колонках `<программа>:<id>`. Проверка цели: rv и t⁺ каждого файла сверяются с K1."""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as P
from models_v2.s11 import v4_fin_data as FD, v4_io as IO4, v4_rqc_calib as RQ

R = P.ROOT / "data/regen"
V5 = R / "harj_hybrid_v5"
DEEP = P.ROOT / "docs/otchety/sber_h1_deep_v1/PREDICTIONS.parquet"
CTRL = ["K1", "K2", "K3", "M", "K8", "D2"]
COMP_COLS = ["HAR", "XJ", "XF", "LF", "B2X", "B2L"]
SERIES_D = {"J211bound": R / "hybrid_final_v1/series_bound.parquet", "J211feat": R / "hybrid_final_v1/series_feat.parquet", "J211w": R / "hybrid_final_v1/series_weights.parquet",
            "J212B": R / "hybrid_struct_v1/B_series.parquet", "J212C": R / "hybrid_struct_v1/C_series.parquet",       # Cg_series.parquet хранит поправки g (positive=False, hs_cstage.py:97), не прогнозы — не читается
            "J212D": R / "hybrid_struct_v1/D_series.parquet", "J212F": R / "hybrid_struct_v1/F_series.parquet"}
V2_FILES = ["H", "H2", "MKT", "ML", "DEC", "DEC2", "Q", "REC", "SEA", "SEA2"]
V5_CFGS = ["GS%d" % i for i in range(1, 6)] + ["SP%d" % i for i in range(1, 5)]


def _a_path():
    from models_v2.s6.e import comp, ser
    return comp, ser.file("a")


def source_files() -> list:
    comp, a = _a_path()
    fs = [IO4.path_of("H", "SBER"), IO4.path_of("K2", "SBER"), IO4.path_of("D2", "x"), FD.OUT / "K8.parquet", comp.FILE, a, DEEP]
    fs += list(SERIES_D.values()) + [R / ("harj_hybrid_v2/%s_series.parquet" % f) for f in V2_FILES]
    fs += [R / "harj_panel_hybrid/hyb_xgb.parquet", R / "harj_panel_hybrid/hyb_lgbm_p17.parquet", R / "harj_panel_hybrid/k4567.parquet"]
    fs += [FD.OUT / FD.file_of(c) for c in FD.CFGS] + [V5 / (c + ".parquet") for c in V5_CFGS]
    return fs


def _d_series(df: pd.DataFrame, cols=None) -> tuple:
    """Ряды с колонками d/target_date/rv -> (date, t_plus, rv, {col: значения}); `__*` — служебные коэффициенты, не прогнозы."""
    cols = cols or [c for c in df.columns if c not in ("d", "target_date", "rv") and "__" not in c]
    return pd.to_datetime(df["d"]), pd.to_datetime(df["target_date"]), df["rv"].to_numpy(float), {c: df[c].to_numpy(float) for c in cols}


def _entries() -> list:
    """[(программа, семейство, файл, режим, префикс, колонки|None)] — реестр источников."""
    comp, a = _a_path()
    e = [("J210", "comp", comp.FILE, "d", "J210comp", COMP_COLS), ("J210", "series_a", a, "d", "J210a", None)]
    e += [("J211" if k.startswith("J211") else "J212", k, p, "d", k, None) for k, p in SERIES_D.items()]
    e += [("V2", f, R / ("harj_hybrid_v2/%s_series.parquet" % f), "d", "V2", None) for f in V2_FILES]
    e += [("PAN", "hyb_xgb", R / "harj_panel_hybrid/hyb_xgb.parquet", "panel", "PANxgb", None), ("PAN", "hyb_lgbm_p17", R / "harj_panel_hybrid/hyb_lgbm_p17.parquet", "panel", "PANlgbm", None),
          ("PAN", "k4567", R / "harj_panel_hybrid/k4567.parquet", "panel", "PAN", None), ("DEEP", "PREDICTIONS", DEEP, "deep", "DEEP", None)]
    e += [("V4", c, FD.OUT / FD.file_of(c), "v", "V4:" + c, None) for c in FD.CFGS] + [("V5", c, V5 / (c + ".parquet"), "v", "V5:" + c, None) for c in V5_CFGS]
    return e


def _read(prog, fam, path, mode, pref, cols) -> tuple:
    if mode == "v":
        f = pd.read_parquet(path, columns=["ticker", "date", "t_plus", "F"])
        f = f[f["ticker"] == "SBER"]
        return pd.to_datetime(f["date"]), pd.to_datetime(f["t_plus"]), None, {pref: f["F"].to_numpy(float)}
    f = pd.read_parquet(path)
    if mode == "panel":
        f = f[f["ticker"] == "SBER"]
        cs = [c for c in f.columns if c.startswith("sigma2_") and c not in ("sigma2_k1",)]
        pre = pref + ("" if pref == "PAN" else ":")
        return pd.to_datetime(f["date"]), pd.to_datetime(f["t_plus"]), f["rv_tplus"].to_numpy(float), {pre + c[7:]: f[c].to_numpy(float) for c in cs}
    if mode == "deep":
        d, tp, rv, s = _d_series(f, ["s2_b2", "s2_harj"])
        return d, tp, rv, {pref + ":" + k: v for k, v in s.items()}
    if pref == "J210a":          # шесть базовых колонок series_a дублируют components.parquet (J210comp)
        cols = [c for c in f.columns if c not in ("d", "target_date", "rv", *COMP_COLS) and "__" not in c]
    d, tp, rv, s = _d_series(f, cols)
    pre = {"V2": "V2:", "J210a": "J210a:", "J210comp": "J210comp:"}.get(pref, pref + ":")
    return d, tp, rv, {pre + k: v for k, v in s.items()}


@functools.lru_cache(maxsize=1)
def load() -> tuple:
    """-> (wide, meta). wide: индекс — дата d, колонки t_plus, rv, контроли и все ряды; meta: по ряду программа, семейство, файл, SHA, совпадение rv и t⁺ с K1."""
    from models_v2.s13.v6_0_io import ROOT, sha_file
    a = IO4.aligned("SBER")
    w = pd.DataFrame({"t_plus": a["t_plus"].to_numpy(), "rv": a["rv"].to_numpy(float), "K1": a["h"].to_numpy(float), "K2": a["k2"].to_numpy(float), "K3": a["k3"].to_numpy(float), "D2": a["d2"].to_numpy(float)}, index=pd.DatetimeIndex(a["date"]))
    w["M"] = np.sqrt(w["K2"] * w["K3"])
    w["K8"] = RQ.k8(w["K1"], w["K2"], w["K3"])
    cols, meta = {}, []
    for prog, fam, path, mode, pref, sel in _entries():
        d, tp, rv, s = _read(prog, fam, path, mode, pref, sel)
        sha = sha_file(path)
        for name, v in s.items():
            x = pd.Series(v, index=pd.DatetimeIndex(d))
            x = x[~x.index.duplicated(keep="last")].reindex(w.index)
            tpm = pd.Series(tp.to_numpy(), index=pd.DatetimeIndex(d))
            tpm = tpm[~tpm.index.duplicated(keep="last")].reindex(w.index)
            rvm = None if rv is None else pd.Series(rv, index=pd.DatetimeIndex(d)).reindex(w.index)
            cols[name] = x.to_numpy(float)
            meta.append(dict(series=name, program=prog, family=fam, source=str(path.relative_to(ROOT)), sha256=sha, n_file=len(d),
                             tplus_mismatch=int((tpm.dropna() != w.loc[tpm.dropna().index, "t_plus"]).sum()),
                             rv_maxdiff=float("nan") if rvm is None else float(np.nanmax(np.abs(rvm.to_numpy() - w["rv"].to_numpy())))))
    wide = pd.concat([w, pd.DataFrame(cols, index=w.index)], axis=1)
    return wide, pd.DataFrame(meta)
