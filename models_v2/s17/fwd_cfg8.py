"""Л4: cfg8 (Residual MoE с якорем на бустинг, V7-лист 3) SBER на склейке «архив + новые» (протокол HARJ_HYBRID_SBER_FWD, раздел 11).

Только замороженные функции архива: `v7_3_cfg.run_cfg` (полный ход) и `v7_1m_wf._refit` (один блок месяца), `v7_1m_feat.build_Z`, `v7_1b_cfg.z_of_year`.
Таблица `tab` — как `v7_1m_data.load_tab`, но из продлённых K1 (Л1), K2/K3 (Л2) и дневного ряда: `make_tab`. Строка d = cut без цели (rv_next = NaN, t⁺ из календаря)
допустима только последней: у неё нет закрытой пары, в обучение и валидацию она не входит.
z года переоценки — энкодер ТОГО ЖЕ года: годы 2018…2025 из архивных артефактов, 2026 — из Л3 (`fwd_z.build_z`, веса и PCA артефакта; обучения нет).
`build_cfg8(cut, src, inp)` — прогноз d = cut по блоку месяца (одна переоценка, 5 seeds); `full=True` — полный ход (все строки). Новые строки требуют замка (Л1).
Запись: `python3 -m models_v2.s17.fwd_cfg8 --write [--check]` — только архивные строки (cut = 2026-02-27) в data/fwd/models/cfg8.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  (первым: общий OpenMP с lightgbm/xgboost, класс № 152)

from models_v2.s10 import hp_har as HH
from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1b_cfg as CFG1, v7_1b_sta as STA, v7_1m_feat as FE, v7_1m_fit as FT, v7_1m_wf as WF, v7_2_cfg as C2, v7_3_cfg as C8
from models_v2.s17 import fwd_cal, fwd_k1, fwd_z as FZ
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, ROOT, Src

OUT = ROOT / "data/fwd/models/cfg8.parquet"
K23 = ROOT / "data/fwd/models/k23.parquet"
KIND, SEEDS = "tcn", FT.SEEDS
TABCOLS = ["H", "K2", "K3", "rv_next", "t_plus", "rv_d", "jshare"]


def pad_daily(daily: pd.DataFrame, t_next) -> pd.DataFrame:
    """Дневной ряд + строка-заглушка t⁺ (rv, bv, j, rv_w, rv_m = NaN, тот же сегмент): без неё `hp_har.grid` не даёт строки d последнего дня (цели нет)."""
    row = daily.iloc[[-1]].copy()
    row.index = pd.DatetimeIndex([pd.Timestamp(t_next)], name=daily.index.name)
    row[["rv", "bv", "j", "rv_w", "rv_m"]] = np.nan
    return pd.concat([daily, row])


def k1_at(cut, src: Src) -> tuple:
    """K1 по парам SBER до cut вместе со строкой d = cut (цель NaN): (таблица `hp_har.series`, дневной ряд без заглушки).
    t⁺ заглушки — следующий канонический день календаря Л1 (только begin); для последнего дня склейки его нет — строки d нет."""
    daily = fwd_k1.build_daily_sber(cut, src)
    nxt = fwd_cal.build_cal(LAST_SESSION, src).set_index("date")["t_plus"].get(daily.index[-1])
    return (HH.series(pad_daily(daily, nxt)) if pd.notna(nxt) else HH.series(daily)), daily


def mask_k23(k23: pd.DataFrame, cut) -> pd.DataFrame:
    """Рамка K2/K3 на момент cut: строки d <= cut; цель неизвестна у пар с t⁺ > cut (rv = NaN). Для тестов обрезки по сохранённому `k23.parquet`."""
    out = k23[k23["date"] <= pd.Timestamp(cut)].copy()
    out.loc[out["t_plus"] > pd.Timestamp(cut), "rv"] = np.nan
    return out.reset_index(drop=True)


def make_tab(k1: pd.DataFrame, k23: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Таблица как `load_tab`: пересечение K1 ∩ K2 ∩ K3 (даты, t⁺ и цель равны, σ̂² конечны и > 0), rv_d и jshare — дневной ряд. Последняя строка может быть без цели (rv NaN)."""
    h = pd.DataFrame({"date": pd.to_datetime(k1["d"]), "t_plus": pd.to_datetime(k1["target_date"]), "rv": k1["rv"].to_numpy(float), "K1": k1["sigma2"].to_numpy(float)})
    k = pd.DataFrame({"date": pd.to_datetime(k23["date"]), "t_plus_k": pd.to_datetime(k23["t_plus"]), "rv_k": k23["rv"].to_numpy(float), "K2": k23["sigma2_k2"].to_numpy(float), "K3": k23["sigma2_k3"].to_numpy(float)})
    m = h.merge(k, on="date").sort_values("date").reset_index(drop=True)
    last = np.arange(len(m)) == len(m) - 1
    ok = (m["t_plus"] == m["t_plus_k"]) & ((m["rv"] == m["rv_k"]) | (np.isnan(m["rv"]) & np.isnan(m["rv_k"]) & last))
    for c in ("K1", "K2", "K3"):
        ok &= np.isfinite(m[c]) & (m[c] > 0)
    ok &= (np.isfinite(m["rv"]) & (m["rv"] > 0)) | last
    m = m[ok].set_index("date")
    rv_d, js = daily["rv"].reindex(m.index), (daily["j"] / daily["rv"]).reindex(m.index)
    tab = pd.DataFrame({"H": m["K1"].to_numpy(float), "K2": m["K2"].to_numpy(float), "K3": m["K3"].to_numpy(float), "rv_next": m["rv"].to_numpy(float), "t_plus": m["t_plus"].to_numpy("datetime64[ms]"), "rv_d": rv_d.to_numpy(float),
                        "jshare": js.to_numpy(float)}, index=m.index)
    if not tab["t_plus"].is_monotonic_increasing or not np.isfinite(tab[["H", "K2", "K3"]].to_numpy()).all() or not np.isfinite(tab["rv_next"].to_numpy()[:-1]).all():
        raise ValueError("таблица SBER: t⁺ не возрастает или есть нечисловые значения")
    return tab


def z_from_frame(z: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    """(n, 4): z склейки на датах index (как `z_of_year`: до 31.12.2026, иначе NaN)."""
    pos = index.get_indexer(pd.DatetimeIndex(z["d"]))
    use = (pos >= 0) & (pd.DatetimeIndex(z["d"]) < pd.Timestamp(FZ.YEAR + 1, 1, 1))
    out = np.full((len(index), 4), np.nan)
    out[pos[use]] = z[[f"z{i}" for i in range(4)]].to_numpy(float)[use]
    return out


def z_maps(index, cut, src: Src, seeds=SEEDS, store=None, xw=None, load=STA.load, years=STA.YEARS) -> dict:
    """{seed: {год: (n, 4)}}: годы < 2026 — архивные энкодеры, 2026 — z Л3 (артефакт 2026, без обучения)."""
    out = {}
    for s in seeds:
        zm = {y: CFG1.z_of_year(KIND, s, index, y, load) for y in years if y < FZ.YEAR}
        zm[FZ.YEAR] = z_from_frame(FZ.build_z(cut, src, s, store, xw), index)
        out[s] = zm
    return out


def tab_at(cut, src: Src, k23=None) -> pd.DataFrame:
    """Таблица на момент cut. k23 — рамка K2/K3 (по умолчанию `k23.parquet` Л2: только архивные строки)."""
    k1, daily = k1_at(cut, src)
    return make_tab(k1, pd.read_parquet(K23) if k23 is None else k23, daily)


def inputs_at(cut, src: Src, k23=None, store=None, xw=None, seeds=SEEDS) -> dict:
    """Входы cfg8 на момент cut: tab и z по seeds."""
    tab = tab_at(cut, src, k23)
    return dict(tab=tab, zmap=z_maps(tab.index, cut, src, seeds, store, xw))


def block_of(tab: pd.DataFrame, a: int | None = None) -> tuple:
    """(позиция переоценки, конец блока): последний блок месяца таблицы (a = None) или блок с позиции a."""
    pos = W6.refit_positions(tab.index, WF.FIRST_REFIT)
    a = pos[-1] if a is None else a
    return a, (pos + [len(tab)])[pos.index(a) + 1]


def seed_block(tab: pd.DataFrame, Zg, z_by_year: dict, a: int, b: int, s: int) -> dict:
    """Одна переоценка (блок a..b) одного seed — тот же `WF._refit`, что внутри walk_forward. -> {G, w, pi, gk, lam} (массивы по блоку)."""
    sink = dict(keep=True)
    with C2.aux_patch(sink):
        g, _ = WF._refit(CFG1.wf_inputs(tab), Zg, z_by_year[tab.index[a].year], C8.SPEC8, tab.index[a], a, b, (s,), C8.GRID8)
    if g is None:
        raise ValueError(f"cfg8: мало закрытых пар на {tab.index[a].date()}")
    k = sink[a]
    return dict(G=g["G"], pi=k["pi"], gk=k["gk"], w=float(np.mean([float(x.w) for x in k["raw"][0]])), lam=tuple(g["lam"]))


def build_cfg8(cut, src: Src, inp: dict | None = None, full: bool = False, seeds=SEEDS, at: int | None = None) -> pd.DataFrame:
    """Прогноз cfg8 (колонки `v7_3_cfg.COLS`): строки блока месяца d = cut (последняя строка — d = cut; at — позиция переоценки другого блока);
    full=True — все строки (полный ход, как `run_prod`). λ блока по seeds — в `df.attrs["lam"]` (seeds × (λ_g, λ_LB))."""
    inp = inp or inputs_at(cut, src, seeds=seeds)
    tab, zm = inp["tab"], inp["zmap"]
    Zg = FE.build_Z(tab).to_numpy(float)
    if full:
        return C8.run_cfg(tab, Zg, lambda s: zm[s], seeds)[0]
    a, b = block_of(tab, at)
    res = [seed_block(tab, Zg, zm[s], a, b, s) for s in seeds]
    out = assemble_block(tab, res, a)
    out.attrs["lam"] = lam_by_seed(res).tolist()
    return out


def assemble_block(tab: pd.DataFrame, res: list, a: int) -> pd.DataFrame:
    """Среднее по seeds блока a..a+len (тот же `v7_3_cfg.assemble`, строки вне блока NaN и отбрасываются)."""
    n, m = len(tab), len(res[0]["G"])
    per = []
    for r in res:
        p = dict(G=np.full(n, np.nan), fitted=np.full(n, np.nan), w=np.full(n, np.nan), pi=np.full((n, 3), np.nan), gk=np.full((n, 2), np.nan))
        p["G"][a:a + m], p["fitted"][a:a + m], p["w"][a:a + m], p["pi"][a:a + m], p["gk"][a:a + m] = r["G"], 1.0, r["w"], r["pi"], r["gk"]
        per.append(p)
    return C8.assemble(tab, per)


def lam_by_seed(res: list) -> np.ndarray:
    """(seeds, 2): (λ_g, λ_LB) переоценки по seeds."""
    return np.array([r["lam"] for r in res], float)


def main(argv: list) -> int:
    src = Src()
    inp = inputs_at(ARCHIVE_END, src)
    df = build_cfg8(ARCHIVE_END, src, inp, full=True)
    print(f"cfg8 archive rows={len(df)} d {df['d'].iloc[0].date()}..{df['d'].iloc[-1].date()}")
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(OUT) + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(OUT)
    if "--check" in argv:
        pd.testing.assert_frame_equal(pd.read_parquet(OUT), df)
        pd.testing.assert_frame_equal(df, pd.read_parquet(ROOT / "data/regen/harj_hybrid_v7/3/cfg8.parquet"))
        print("cfg8.parquet == пересчёт == архив V7: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
