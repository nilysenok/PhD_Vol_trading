"""Семья E: ряды σ̂ для лестницы (E-const, E-naive, E-harj, E-hyb) в дневной шкале σ, только 2017–2021.

Каждая строка (date, ticker, sigma) — σ̂ для d+1 (d0), известная на закрытии d.
 - const: σ̄_d = среднее √RV тикера по строкам d−252…d−1 (день d не входит);
 - naive: √(среднее RV дней d−4…d), w = 5 (строка ch4_dev/sigma_naive_dev.parquet — тот же ряд, сверка в build());
 - harj_h1, hyb_h1: sigma_pred_h1 из ch4_dev как есть;
 - harj_h5d, hyb_h5d: «дневной эквивалент» σ̂₅. В файлах *_h5_dev лежит σ = √(среднее RV за d+1…d+5)
   (медиана σ̂₅/√(фактич. среднее RV) = 1,15; после /√5 было бы 0,51), т.е. УЖЕ дневная шкала: σ_экв = σ̂₅,
   деление на √5 дало бы вдвое заниженный ряд (↺ к тексту задания «σ̂₅/√5»).
Данные ≥ 2022 не читаются: фильтр при чтении + assert.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
SRC = MAIN / "data" / "regen"
CH4 = SRC / "ch4_dev"
OUT = Path(__file__).resolve().parent.parent / "data" / "fastbt" / "sigma_e"
LAST = pd.Timestamp("2021-12-31")
FIRST = pd.Timestamp("2017-01-03")
WIN_CONST, WIN_NAIVE = 252, 5
NAMES = ("const", "naive", "harj_h1", "hyb_h1", "harj_h5d", "hyb_h5d")
TIP = {"const": "okno", "naive": "okno", "harj_h1": "model", "hyb_h1": "model", "harj_h5d": "model", "hyb_h5d": "model"}
KNOWN = {n: (-1,) if n == "const" else (0,) for n in NAMES}   # последний известный день входов ряда
_SRC_FILES = {"harj_h1": ("sigma_harj_b_h1_dev", "sigma_pred_h1"), "hyb_h1": ("sigma_hyb_h1_dev", "sigma_pred_h1"),
              "harj_h5d": ("sigma_harj_b_h5_dev", "sigma_pred_h5"), "hyb_h5d": ("sigma_hyb_h5_dev", "sigma_pred_h5")}


def read_dev(path, cols) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=cols, filters=[("date", "<=", LAST)])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= LAST].reset_index(drop=True)
    assert len(d) == 0 or d["date"].max() <= LAST, "дата > 2021-12-31"
    return d


def rv_dev() -> pd.DataFrame:
    return read_dev(SRC / "features_pit_v3.parquet", ["date", "ticker", "rv"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _out(rv: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    o = rv[["date", "ticker"]].assign(sigma=s.astype("float64").values)
    o = o[(o["date"] >= FIRST) & (o["date"] <= LAST)].dropna()
    return o.sort_values(["date", "ticker"]).reset_index(drop=True)


def sigma_const(rv: pd.DataFrame) -> pd.DataFrame:
    """σ̄_d: среднее √RV по 252 строкам тикера, оканчивающимся днём d−1 (shift(1) — текущий день не входит)."""
    r = rv.sort_values(["ticker", "date"]).reset_index(drop=True)
    s = r.groupby("ticker")["rv"].transform(lambda x: np.sqrt(x).rolling(WIN_CONST, min_periods=WIN_CONST).mean().shift(1))
    return _out(r, s)


def sigma_naive(rv: pd.DataFrame) -> pd.DataFrame:
    """√(среднее RV d−4…d) — окно включает день d (известен на закрытии d), как в ch4_dev."""
    r = rv.sort_values(["ticker", "date"]).reset_index(drop=True)
    s = r.groupby("ticker")["rv"].transform(lambda x: x.rolling(WIN_NAIVE, min_periods=WIN_NAIVE).mean())
    return _out(r, np.sqrt(s))


def load_model(name: str) -> pd.DataFrame:
    f, c = _SRC_FILES[name]
    d = read_dev(CH4 / f"{f}.parquet", ["date", "ticker", c]).rename(columns={c: "sigma"})
    d["sigma"] = d["sigma"].astype("float64")   # daily-эквивалент h5: σ̂₅ как есть (см. шапку)
    return d.sort_values(["date", "ticker"]).reset_index(drop=True)


def build() -> dict:
    """Пишет data/fastbt/sigma_e/{name}.parquet; сверка naive с sigma_naive_dev (макс. |Δ|)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rv = rv_dev()
    res = {"const": sigma_const(rv), "naive": sigma_naive(rv), **{n: load_model(n) for n in _SRC_FILES}}
    old = read_dev(CH4 / "sigma_naive_dev.parquet", ["date", "ticker", "sigma_pred_h1"])
    j = res["naive"].merge(old, on=["date", "ticker"], how="outer", indicator=True)
    sv = {"naive_vs_ch4_strok": len(j), "naive_tolko_odin": int((j["_merge"] != "both").sum()),
          "naive_max_abs": float((j["sigma"] - j["sigma_pred_h1"]).abs().max())}
    for n, d in res.items():
        d.to_parquet(OUT / f"{n}.parquet", index=False)
    return sv


def load(name: str) -> pd.DataFrame:
    """Ряд E из data/fastbt/sigma_e (date, ticker, sigma); нет файла — build()."""
    assert name in NAMES, name
    p = OUT / f"{name}.parquet"
    if not p.exists():
        build()
    return read_dev(p, ["date", "ticker", "sigma"])


def proverka_sdvig(df: pd.DataFrame, rv: pd.DataFrame, tip: str, K: tuple = (0,)) -> dict:
    """Проверка сдвига тем же кодом, что ch4_dev/proverka_sdvig.csv (r2_loyo.sdvig + r2_features.shift_table).
    model — решает уровень; okno — приращение (уровень в окне имеет пик в середине окна).
    K — последний известный день ряда: 0 для всех, кроме const (окно оканчивается d−1 → K=(−1,)). ->{argmax_uroven, argmax_prir, ok, corr_k±}."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))
    import r2_features as RF
    import r2_loyo as O
    d = df[(df["date"] >= pd.Timestamp("2017-01-01")) & (df["date"] <= LAST)]
    s = O.sdvig(d[["date", "ticker"]].assign(rv_pred=d["sigma"].astype(float) ** 2), rv, K=(0,))
    F = d[["date", "ticker"]].assign(ln_sigma2=np.log(d["sigma"].astype(float) ** 2))
    p = RF.shift_table(F, {"B": rv}, "E").iloc[0]
    dam = None if p["d_argmax"] is None or p["d_argmax"] != p["d_argmax"] else int(p["d_argmax"])
    lev_ok = bool(s["ok"])
    if tip == "model":
        ok = lev_ok
    else:
        # приращение: пик строго в диапазоне известных дней (sdvig_ok при K=(−1,) пропустил бы и 0 — окно с днём d)
        ok = dam is not None and min(K) <= dam <= max(K) and bool(O.sdvig_ok(dam, K)[0])
    return {"argmax_uroven": s["argmax"], "uroven_ok": lev_ok, "argmax_prir": dam, "ok": bool(ok),
            **{f"corr_k{k:+d}": s[f"corr_s{k:+d}"] for k in (-2, -1, 0, 1)},
            **{f"rho_prir_k{k:+d}": p[f"drho_{k}"] for k in (-2, -1, 0, 1)}, "tip": tip}


def main() -> None:
    sv = build()
    print("сверка naive:", sv)
    rv = rv_dev()
    rv_s = rv.assign(date=pd.to_datetime(rv["date"]))
    rows = []
    for n in NAMES:
        d = load(n)
        for start in ("2017", "2018"):
            r = proverka_sdvig(d[d["date"] >= pd.Timestamp(f"{start}-01-01")], rv_s, TIP[n], KNOWN[n])
            rows.append({"ryad": n, "period": f"{start}–2021", "n_strok": len(d[d["date"] >= pd.Timestamp(f"{start}-01-01")]), **r})
    t = pd.DataFrame(rows)
    p = Path(__file__).resolve().parent.parent / "docs" / "tables" / "semya_e_proverka_sdvig.csv"
    t.to_csv(p, index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
