"""PREDICTIONS.parquet: единая таблица Б-2 / HAR-J(б) + режимы (PIT) + значения признаков Б-2 (вход модели, форма 'm')."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from models_v2 import daily as dly
from models_v2 import metrics, prov
from models_v2.s2 import features as ft
from models_v2.s4 import mkt, prov4, select
from models_v2.s4.fit import pair_index
from models_v2.s5 import common, pit

FEAT = tuple(ft.NAMES[:ft.N_BASE]) + tuple(mkt.BASE) + tuple(mkt.IDX)
DZK = "m_mkx"


def _key() -> str:
    h = hashlib.sha256()
    for part in (common.sha_file(prov.BARS), common.sha_file(common.IMOEX), prov4.sha_papers(), prov.code_sha(), prov4.code_sha4(),
                 common.sha_file(__file__)):
        h.update(part.encode())
    return h.hexdigest()[:16]


def features_b2() -> tuple:
    """(DataFrame f_* по сетке с индексом d + aux_imoex_rv, {год: число пар обучения на 1 января}); кэш с ключом входов и кода (Д5)."""
    p, k, j = common.DEEP / "features_b2.parquet", common.DEEP / "features_b2.key", common.DEEP / "n_pairs.json"
    key = _key()
    if p.exists() and k.exists() and k.read_text() == key and j.exists():
        return pd.read_parquet(p), {int(a): b for a, b in json.loads(j.read_text()).items()}
    from models_v2 import run_fix   # импорт здесь: тяжёлый, нужен только при пересчёте
    F, grid, DZ, _, _ = run_fix.prepare(str(common.IMOEX))
    dz = DZ[DZK]
    df = pd.DataFrame(dz.X[grid], columns=["f_" + n for n in FEAT], index=pd.DatetimeIndex(F.dates[grid], name="d"))
    idly, _ = dly.build_daily(pd.read_parquet(common.IMOEX), index_days=None)
    df["aux_imoex_rv"] = idly["rv"].reindex(df.index).to_numpy(float)
    npairs = {Y: int(len(pair_index(F.seg, dz.X, dz.y, (0, 1, 2, 3), select.cut_row(F, Y)))) for Y in range(2017, 2027)}
    common.DEEP.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    j.write_text(json.dumps(npairs))
    k.write_text(key)
    return df, npairs


def daily_p1() -> pd.DataFrame:
    return pd.read_parquet(common.OUT / "daily_p1.parquet")


def regimes(dl: pd.DataFrame) -> pd.DataFrame:
    """Терцили и квинтили RV_w по 252 предыдущим строкам дневного ряда (индекс — дата d)."""
    w = dl["rv_w"].to_numpy(float)
    return pd.DataFrame({"tercile_w": pit.past_bucket(w, [1 / 3, 2 / 3]), "quintile_w": pit.past_bucket(w, [.2, .4, .6, .8])}, index=dl.index)


def build(c: pd.DataFrame, dl: pd.DataFrame, stress: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = c.copy()
    for m in ("b2", "harj"):
        s2 = df["s2_" + m]
        df["err_ln_" + m] = np.log(df["rv"]) - np.log(s2)
        df["err_lvl_" + m] = df["rv"] - s2
        df["qlike_" + m] = metrics.qlike(df["rv"], s2)
        df["sign_" + m] = np.sign(df["err_ln_" + m]).astype(int)   # +1: RV больше прогноза (недооценка)
    df["dloss"] = df["qlike_b2"] - df["qlike_harj"]
    for col in ("rv", "rv_w", "rv_m"):
        df[col + "_d" if col == "rv" else col] = dl[col].reindex(df["d"]).to_numpy()
    rg = regimes(dl).reindex(df["d"])
    df["tercile_w"], df["quintile_w"] = rg["tercile_w"].to_numpy(), rg["quintile_w"].to_numpy()
    st = stress.set_index("d")["stress"].reindex(df["d"])
    df["stress_g4"] = st.to_numpy()
    td = df["target_date"]
    df["mar_jun_2020"] = ((td >= "2020-03-01") & (td <= "2020-06-30")).to_numpy()
    df["y2022"] = (td.dt.year == 2022).to_numpy()
    f = feats.reindex(df["d"])
    for col in f.columns:
        df[col] = f[col].to_numpy()
    return df


def make() -> pd.DataFrame:
    feats, _ = features_b2()
    stress = pd.read_parquet(common.OUT / "stress_p3.parquet")
    df = build(common.controls(), daily_p1(), stress, feats)
    common.REP.mkdir(parents=True, exist_ok=True)
    df.to_parquet(common.REP / "PREDICTIONS.parquet")
    return df
