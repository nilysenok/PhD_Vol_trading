"""W4-B1 (Ж-250, W4_SPEC 2.2, 2.4): продолжение признаков S (9) и Σ (12 площадей) V6 на d нового периода 2026-02-27…2026-11-09.
Только импорт замороженного кода V6 (`s13.v6_1_feat`, `s13.v6_1b_sig`) и источника `s17.fwd_src`/`fwd_k1`; бары — «архив + новые» с обрезкой begin <= cut 18:40.
Признаки — не прогнозы: M⁺ и потери на окне M не считаются. `build()` пишет ТОЛЬКО в `data/regen/harj_x1/W4/feat/`.
Мутации (тестировщик): `cut` и `src` — параметры `rows_at`; окно s7 и 5-дневное среднее — `FE.vol_ratio`, `SG.five_mean` (вызываются через модуль).
--check-volume один пишет `W4_VOLUME.csv`; внутри `--check` перечень только пересчитывается и сверяется с файлом (read-only)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s13 import v6_1_feat as FE
from models_v2.s13 import v6_1b_sig as SG
from models_v2.s17.fwd_k1 import build_daily_sber
from models_v2.s17.fwd_src import ROOT, Src

V6 = ROOT / "data/regen/harj_hybrid_v6"
FEAT = ROOT / "data/regen/harj_x1/W4/feat"
NEW_K1, NEW_X1 = ROOT / "data/fwd/models/new/k1.parquet", ROOT / "data/fwd/models/new/x1.parquet"
VOLCSV = ROOT / "docs/otchety/harj_x1/W4_VOLUME.csv"
ARCH_LAST, NEW_LAST = pd.Timestamp("2026-02-26"), pd.Timestamp("2026-11-09")
TOL_SIGMA, VOL_JUMP, VOL_WIN = 1e-12, 10.0, 20


def outputs() -> list:
    return [FEAT / "S_ext.parquet", FEAT / "SIGMA_ext.parquet", VOLCSV]


def sha(paths) -> dict:
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in paths}


def new_dates(x1_path: Path = NEW_X1) -> pd.DatetimeIndex:
    """Даты d нового периода — ровно d файла x1 продолжения."""
    return pd.DatetimeIndex(pd.read_parquet(x1_path, columns=["d"])["d"])


def load_h(k1_path: Path = NEW_K1, x1_path: Path = NEW_X1) -> pd.Series:
    """H по d: архив — `H_table.h` (V6), продолжение — sigma2 K1 из `new/k1.parquet`; d продолжения обязаны совпасть с d файла x1."""
    tab = pd.read_parquet(V6 / "H_table.parquet")
    arch = tab.set_index("d")["h"]
    k1 = pd.read_parquet(k1_path).set_index("d")["sigma2"]
    if not k1.index.equals(new_dates(x1_path)):
        raise ValueError("d нового K1 не совпадают с d x1 продолжения")
    if k1.index.min() <= arch.index.max():
        raise ValueError("продолжение K1 пересекается с архивом H")
    return pd.concat([arch, k1]).astype(float).rename("h")


def rows_at(cut, src: Src, h: pd.Series) -> tuple:
    """-> (S, Σ-таблица) на днях h.index <= cut по барам begin <= cut 18:40. Σ-таблица = 12 площадей + 8 sig3 + 2 флага (как SIGMA_features)."""
    hh = h[h.index <= pd.Timestamp(cut)]
    bars = src.bars("SBER", cut)
    daily = build_daily_sber(cut, src)
    S = FE.assemble(FE.raw_features(bars, daily), hh)
    sigma, sig3, flags = SG.features(bars, daily)
    return S, pd.concat([sigma, sig3, flags], axis=1).reindex(S.index)


def _flat(x: pd.DataFrame) -> pd.DataFrame:
    return x.rename_axis("d").reset_index()


def full_run(src: Src | None = None, cut=NEW_LAST, h: pd.Series | None = None) -> tuple:
    src = src or Src()
    return rows_at(cut, src, load_h() if h is None else h)


def stored() -> tuple:
    S = pd.read_parquet(V6 / "S_features.parquet").set_index("d")
    return S, pd.read_parquet(V6 / "SIGMA_features.parquet").set_index("d")


def assemble_ext(S_full: pd.DataFrame, G_full: pd.DataFrame) -> tuple:
    """Архивные строки — сохранённые файлы V6, продолжение — строки полного прогона с d > 2026-02-26."""
    S0, G0 = stored()
    new = S_full.index > ARCH_LAST
    return pd.concat([S0, S_full[new]])[S0.columns], pd.concat([G0, G_full[new]])[G0.columns]


def atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def build(out_dir: Path = FEAT) -> tuple:
    S, G = assemble_ext(*full_run())
    atomic(_flat(S), out_dir / "S_ext.parquet")
    atomic(_flat(G), out_dir / "SIGMA_ext.parquet")
    return S, G


def read_ext(out_dir: Path = FEAT) -> tuple:
    return pd.read_parquet(out_dir / "S_ext.parquet").set_index("d"), pd.read_parquet(out_dir / "SIGMA_ext.parquet").set_index("d")


def maxdiff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))) if a.size else 0.0


def bitwise(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Побитное равенство значений, индекса по датам (без учёта единицы datetime64) и колонок; NaN == NaN."""
    ia, ib = pd.DatetimeIndex(a.index).astype("datetime64[ns]"), pd.DatetimeIndex(b.index).astype("datetime64[ns]")
    return bool(ia.equals(ib) and list(a.columns) == list(b.columns) and np.array_equal(a.to_numpy(float).view(np.int64), b.to_numpy(float).view(np.int64)))


def same_nan(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.shape == b.shape and bool((np.isnan(a.to_numpy(float)) == np.isnan(b.to_numpy(float))).all())


def check_archive(src: Src | None = None) -> dict:
    """Архивные строки S побитно = S_features.parquet, Σ: max|Δ| <= 1e-12 — и для ext-файлов, и для пересчёта по cut = 2026-02-26."""
    (S0, G0), (Se, Ge), (Sr, Gr) = stored(), read_ext(), rows_at(ARCH_LAST, src or Src(), load_h())
    ea, ga = Se.loc[S0.index], Ge.loc[G0.index]
    res = {"S_ext_bitwise": bitwise(ea, S0), "S_recompute_bitwise": bitwise(Sr, S0),
           "G_ext_maxdiff": maxdiff(ga, G0), "G_recompute_maxdiff": maxdiff(Gr.reindex(G0.index)[G0.columns], G0), "n_arch": len(S0), "n_ext": len(Se)}
    res["ok"] = bool(res["S_ext_bitwise"] and res["S_recompute_bitwise"] and res["G_ext_maxdiff"] <= TOL_SIGMA and res["G_recompute_maxdiff"] <= TOL_SIGMA
                     and same_nan(ga, G0) and same_nan(Gr.reindex(G0.index)[G0.columns], G0) and ea.shape == S0.shape)
    return res


def check_pit(src: Src | None = None, dates=None) -> dict:
    """На каждом новом d строка по cut = d равна строке полного прогона (из ext-файлов); d ≡ d x1 продолжения."""
    src, h, Se, Ge = src or Src(), load_h(), *read_ext()
    ds = new_dates() if dates is None else pd.DatetimeIndex(dates)
    bad, worst = [], 0.0
    for d in ds:
        S, G = rows_at(d, src, h)
        a, b = S.loc[[d]], Se.loc[[d]]
        g, ge = G.loc[[d]][Ge.columns], Ge.loc[[d]]
        diff = max(maxdiff(a, b), maxdiff(g, ge))
        worst = max(worst, diff)
        if not (bitwise(a, b) and bitwise(g, ge) and S.index[-1] == d):
            bad.append(str(d.date()))
    return {"n_days": len(ds), "n_bad": len(bad), "bad": bad, "max_abs_diff": worst, "d_equals_x1": bool(Se.index[Se.index > ARCH_LAST].equals(new_dates())), "ok": not bad}


def volume_table(src: Src | None = None, cut=NEW_LAST) -> pd.DataFrame:
    """Дни (архив и новые), где медиана объёма бара основной сессии отличается от медианы дневных медиан прошлых 20 сессий в >= 10 раз (вверх: лоты → штуки)."""
    src = src or Src()
    main = D.main_session(src.bars("SBER", cut))
    med = main.groupby(main["begin"].dt.normalize())["volume"].median()
    med = med[med.index.isin(build_daily_sber(cut, src).index)]    # только канонические дни (без выходных сессий IMOEX-календаря)
    prev = med.shift(1).rolling(VOL_WIN, min_periods=VOL_WIN).median()
    r = (med / prev).replace([np.inf, -np.inf], np.nan)
    hit = (r >= VOL_JUMP) | (r <= 1.0 / VOL_JUMP)
    out = pd.DataFrame({"d": r.index[hit], "period": np.where(r.index[hit] > ARCH_LAST, "new", "archive"), "bar_vol_median": med[hit].to_numpy(),
                        "prev20_median": prev[hit].to_numpy(), "ratio": r[hit].to_numpy(), "direction": np.where(r[hit] >= VOL_JUMP, "up", "down")})
    return out.reset_index(drop=True)


def csv_text(tab: pd.DataFrame) -> str:
    return tab.to_csv(index=False, lineterminator="\n")


def check_volume(write: bool, path: Path = VOLCSV, src: Src | None = None) -> dict:
    tab = volume_table(src)
    text = csv_text(tab)
    if write:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
    same = path.exists() and path.read_text() == text
    return {"n_days": len(tab), "days": [str(d.date()) for d in tab["d"]], "file_matches": bool(same), "ok": bool(same)}


def check_all() -> dict:
    before = sha(outputs())
    res = {"archive": check_archive(), "pit": check_pit(), "volume": check_volume(False)}
    after = sha(outputs())
    res["hash_unchanged"] = before == after and all(v is not None for v in before.values())
    res["ok"] = all(res[k]["ok"] for k in ("archive", "pit", "volume")) and res["hash_unchanged"]
    return res


def main(argv: list) -> int:
    if "--check-archive" in argv:
        res = check_archive()
    elif "--check-pit" in argv:
        res = check_pit()
    elif "--check-volume" in argv:
        res = check_volume(True)
    elif "--check" in argv:
        res = check_all()
    else:
        S, G = build()
        res = {"ok": True, "rows": len(S), "d": f"{S.index.min().date()}..{S.index.max().date()}", "new_days": int((S.index > ARCH_LAST).sum())}
    print(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
