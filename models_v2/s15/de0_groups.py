"""DE0, G3: восемь группирующих признаков (объявлены в задании `2026-11-09_harj_deep_0.md`; после просмотра не меняются) и группы по ПРОШЛОМУ.
Признак дня d известен на закрытии d. Терцили (порог) в день i — только по значениям признака в днях < i (расширяющееся окно); пока прошлых значений меньше MIN_PAST = 250 — дата без группы (-1).
Знак доходности — две группы (вниз/вверх), экстремальный бар — две (нет/да); порог бара — 99-й перцентиль |r| ВСЕХ баров дней < i."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAST = 250
EWMA_SPAN = 20
Z_WIN = 60
FEATURES = {"a_uroven": "уровень ln RV_d", "b_izmenenie": "ln RV_d − среднее ln RV за 5 предыдущих дней", "c_znak": "знак доходности дня d", "d_ekstrem": "экстремальный M10-бар дня d",
            "e_herfindal": "концентрация r² по барам (Херфиндаль)", "f_raskhozhd": "|ln(K2/H)|", "g_oshibki": "|EWMA₂₀ прошлых закрытых ln(RV/H)|", "h_neustojch": "расстояние z_d (TCN-w5) от среднего z прошлых 60 дат"}
BINS = {"c_znak": ["вниз", "вверх"], "d_ekstrem": ["нет", "да"]}


def terciles_past(x: np.ndarray, min_past: int = MIN_PAST) -> np.ndarray:
    """Метки 0/1/2 (T1 низ … T3 верх), -1 — нет группы. Пороги дня i — по x[:i] (конечные значения), не меньше `min_past` значений."""
    x = np.asarray(x, float)
    out = np.full(len(x), -1, int)
    for i in range(len(x)):
        past = x[:i][np.isfinite(x[:i])]
        if len(past) < min_past or not np.isfinite(x[i]):
            continue
        q1, q2 = np.quantile(past, [1 / 3, 2 / 3])
        out[i] = int(x[i] > q1) + int(x[i] > q2)
    return out


def day_stats(store: dict, dates: pd.DatetimeIndex) -> dict:
    """По барам M10 дня d: сумма доходностей, max|r|, Херфиндаль долей r², набор |r| бара (для порога). Нет баров — NaN."""
    pos = store["days"].get_indexer(dates)
    if (pos < 0).any():
        raise SystemExit("в складе M10 нет дней таблицы")
    r = store["r"][pos]
    fin = np.isfinite(r)
    r0 = np.where(fin, r, 0.0)
    r2 = r0 ** 2
    tot = r2.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        herf = np.where(tot > 0, ((r2 / tot[:, None]) ** 2).sum(1), np.nan)
    has = fin.any(1)
    return dict(ret=np.where(has, r0.sum(1), np.nan), maxabs=np.where(has, np.abs(r0).max(1), np.nan), herf=herf, pos=pos)


def extreme_flag(store: dict, dates: pd.DatetimeIndex, st: dict) -> np.ndarray:
    """Метка 1/0: max|r| дня d > 99-го перцентиля |r| баров дней, СТРОГО предшествующих d (в складе; -1 — нет группы, i < MIN_PAST или нет баров)."""
    absr = np.abs(store["r"])
    out = np.full(len(dates), -1, int)
    for i in range(MIN_PAST, len(dates)):
        p = st["pos"][i]
        past = absr[:p][np.isfinite(absr[:p])]
        if len(past) and np.isfinite(st["maxabs"][i]):
            out[i] = int(st["maxabs"][i] > np.quantile(past, 0.99))
    return out


def ewma_closed_err(err: np.ndarray, tp: np.ndarray, dates: np.ndarray, span: int = EWMA_SPAN) -> np.ndarray:
    """EWMA (α = 2/(span+1)) прошлых ЗАКРЫТЫХ ln(RV/H): в день i учтены пары j с t⁺_j ≤ d_i. NaN — пока закрытых нет."""
    a, out, j, cur = 2.0 / (span + 1), np.full(len(dates), np.nan), 0, np.nan
    for i in range(len(dates)):
        while j < len(err) and tp[j] <= dates[i]:
            cur = err[j] if not np.isfinite(cur) else (1 - a) * cur + a * err[j]
            j += 1
        out[i] = cur
    return out


def z_distance(dates: pd.DatetimeIndex, loader, enc_year=lambda d: d.year, seeds=(0, 1, 2, 3, 4), win: int = Z_WIN) -> np.ndarray:
    """(з): расстояние z_d энкодера года `enc_year(d)` от среднего z 60 предыдущих дат ТОГО ЖЕ энкодера (по осям в единицах их σ за те 60 дат); среднее по seeds.
    loader(год, seed) → (дни, z). Нет дня в энкодере / меньше `win` прошлых дат — NaN."""
    out = np.full(len(dates), np.nan)
    ey = np.array([enc_year(d) for d in dates])
    for y in sorted(set(ey)):
        acc = []
        for s in seeds:
            try:
                days, z = loader(int(y), s)
            except FileNotFoundError:
                break
            pos = days.get_indexer(dates[ey == y])
            v = np.full(len(pos), np.nan)
            for k, p in enumerate(pos):
                if p >= win:
                    pr = z[p - win:p]
                    sd = pr.std(0)
                    v[k] = float(np.sqrt((((z[p] - pr.mean(0)) / np.where(sd > 1e-12, sd, np.nan)) ** 2).sum()))
            acc.append(v)
        if acc:
            out[ey == y] = np.mean(acc, axis=0)
    return out


def feature_table(d: dict, store: dict, loader, enc_year=lambda d: d.year) -> dict:
    """-> {имя признака: массив значений (a, b, e, f, g, h — числа; c, d — готовые метки 0/1/-1)} на датах таблицы дней `d['tab']`."""
    tab = d["tab"]
    dates = tab.index
    st = day_stats(store, dates)
    lrv = np.log(tab["rv_d"].to_numpy(float))
    prev5 = pd.Series(lrv).shift(1).rolling(5).mean().to_numpy()
    ret = st["ret"]
    c = np.where(np.isfinite(ret) & (ret != 0), (ret > 0).astype(int), -1)
    c[:MIN_PAST] = -1                                    # первые 250 дат истории — без группы (как у остальных признаков)
    err = np.log(tab["rv_next"].to_numpy(float) / tab["H"].to_numpy(float))
    return {"a_uroven": lrv, "b_izmenenie": lrv - prev5, "c_znak": c, "d_ekstrem": extreme_flag(store, dates, st), "e_herfindal": st["herf"],
            "f_raskhozhd": np.abs(np.log(tab["K2"].to_numpy(float) / tab["H"].to_numpy(float))),
            "g_oshibki": np.abs(ewma_closed_err(err, tab["t_plus"].to_numpy("datetime64[ns]"), dates.to_numpy("datetime64[ns]"))),
            "h_neustojch": z_distance(dates, loader, enc_year)}


def labels(feat: dict) -> dict:
    """Метки групп по каждому признаку: терцили по прошлому, для c и d — готовые."""
    return {k: (v if k in BINS else terciles_past(v)) for k, v in feat.items()}


def group_table(lab: dict, pos_idx: np.ndarray, rv, f: dict, year: np.ndarray, series=("K1", "K2", "A2", "cfg8")) -> pd.DataFrame:
    """По признаку, группе и окну (2018_2021, 2022_2025): n, средний u, QLIKE по рядам, Δ (A2, cfg8) − K2. `pos_idx` — позиции дат idx в таблице дней. Строка `все` — все даты окна."""
    from models_v2.s15 import de0_core as C
    u = {s: C.u_of(rv, f[s]) for s in series}
    q = {s: C.qlike_day(rv, f[s]) for s in series}
    rows = []
    for feat, lb in [("все", None)] + list(lab.items()):
        bins = [(-9, "—")] if lb is None else list(enumerate(BINS.get(feat, ["T1 (низ)", "T2", "T3 (верх)"])))
        for b, name in bins:
            for w in ("2018_2021", "2022_2025"):
                m = C.wmask(year, w) & (True if lb is None else lab[feat][pos_idx] == b)
                if not m.any():
                    continue
                r = dict(feature=feat, bin=name, window=w, n=int(m.sum()))
                r.update({"u_" + s: float(u[s][m].mean()) for s in series})
                r.update({"q_" + s: float(q[s][m].mean()) for s in series})
                r.update(d_A2=r["q_A2"] - r["q_K2"], d_cfg8=r["q_cfg8"] - r["q_K2"])
                rows.append(r)
    return pd.DataFrame(rows).rename(columns=lambda c: c.replace("K1", "H"))
