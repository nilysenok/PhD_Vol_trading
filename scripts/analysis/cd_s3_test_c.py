"""C/D-разрез для S3 строго по замороженной TEST_C (Ж-116 п. 3): hyb_h5, C1 W504 p70 (fastbt/s9.py: TEST_C, c_points, _c_gate).
Запуск (один процесс): python3 scripts/analysis/cd_s3_test_c.py   [CD_S3_DATA=<копия с data/>]
Правило затвора = s9: ранг σ_t в окне [t−504, t−1] (текущий бар не входит; доля строго ниже; нужно >= 252 значений, past_rank/_minp);
S3 — категория 0 (тренд), её затвор G[0]: торгуем, если ранг NaN или ранг > 1−p = 0,30 (тренду нужна не-низкая σ̂; ВЫСОКУЮ σ̂ не режем).
Затвор побаровый: позиция * маска (как e.A_row * g в s9) — закрывает и открытую позицию, в отличие от #483 (вход-only, ранг >= p67).
База — A_strogaya «только S3», окно 2018–2019 (σ̂ с 2017). Данные ≥ 2020 не читаются, метрики 2020–2021 не считаются."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cd_s3_sigma as cd                      # noqa: E402  (чтение σ, schitat, k2_mask, Д1 — те же, что в #483)

W, MINP, P_TEST = 504, 252, 0.70
OUT, REPORT = cd.ROOT / "docs/tables/cd_s3", cd.ROOT / "docs/otchety/2026-10-28_cd_s3_test_c.md"
FILES = {"hyb": "sigma_hyb_h5_dev", "harj": "sigma_harj_b_h5_dev", "naive": "sigma_naive_dev"}   # naive: один ряд для h1 и h5
STROKI = {"hyb": ("затвор C1 W504 p70 по σ̂ гибрида h5 (H2, TEST_C)", "H2"),
          "harj": ("контроль: тот же затвор по HAR-J (б) h5", "kontrol"),
          "naive": ("контроль: тот же затвор по наивной √RV", "kontrol")}


def rang(arr: np.ndarray, i: int) -> float:
    """Ранг arr[i] в окне arr[i−504..i−1] (текущий бар не входит; доля строго ниже). < 252 значений или nan в arr[i] — nan."""
    w = arr[max(i - W, 0):i]
    w = w[np.isfinite(w)]
    if not np.isfinite(arr[i]) or len(w) < MINP:
        return float("nan")
    return float((w < arr[i]).sum()) / len(w)


def allow_c1(arr, p: float = P_TEST) -> np.ndarray:
    """S3 (категория 0): allow[i] = ранг nan или ранг > 1−p, как G[0] в s9._c_gate. Строки после i не читаются."""
    arr = np.asarray(arr, float)
    r = np.array([rang(arr, i) for i in range(len(arr))])
    return np.isnan(r) | (r > 1 - p)


def mask_s9(arr, p: float = P_TEST) -> np.ndarray:
    """Та же маска из готовых функций s9 (past_rank, _c_gate, CATS) — для сверки."""
    from fastbt import s9
    from fastbt.s9_stats import past_rank
    R = past_rank(np.asarray(arr, float).copy(), W, MINP)[None, :]
    return s9._c_gate(R, p)[s9.CATS["S3"], 0] > 0.5


def sig_fn_poz(allow_of):
    """sig_fn для evaluate_v2: позиция * маска (побаровый затвор s9)."""
    def fn(t, sig, dates):
        a = np.asarray(allow_of(t, dates), bool)
        if len(a) != len(sig):
            raise ValueError("маска затвора и позиции разной длины")
        return np.asarray(sig, float) * a
    return fn


def kesh(ser: dict):
    """allow_of(t, dates) с кэшем маски по (тикер, длина дат); σ̂ выровнена как s9.align (последнее значение с датой <= бара)."""
    memo = {}

    def f(t, dates):
        k = (t, len(dates))
        if k not in memo:
            s = ser[t].reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy(float)
            memo[k] = allow_c1(s)
        return memo[k]
    return f


def sharpe(x) -> float:
    from fastbt import metriki_e as me
    return float(me.sr_dnevnoy(np.asarray(x, float)) * np.sqrt(252))


def tablica(res: dict, k2: list, p_k2: float) -> pd.DataFrame:
    """Строки по годам и 2018–2019: mSR, Sharpe (ddof=1, √252), сделки, Δ против базы и против HAR-J (б) как разность табличных mSR."""
    from fastbt import metriki_e as me
    def ms(r, g):
        yr = r["dates"].astype("datetime64[Y]").astype(int) + 1970
        m = np.isin(yr, cd.GODY) if g == "2018-2019" else yr == g
        mr = np.isin(r["god"], cd.GODY) if g == "2018-2019" else r["god"] == g
        return me.msr(r["x"][m]), sharpe(r["x"][m]), float(r["cnt"][mr].sum()) / 2
    names = [("A_strogaya только S3 (база)", "baza", res["baza"])]
    names += [(STROKI[k][0], STROKI[k][1], res[k]) for k in ("hyb", "harj", "naive")]
    rows = []
    for g in (*cd.GODY, "2018-2019"):
        base, harj = ms(res["baza"], g)[0], ms(res["harj"], g)[0]
        for nm, tip, r in names:
            a, b, c = ms(r, g)
            rows.append(dict(stroka=nm, tip=tip, god=g, msr=a, sharpe=b, sdelok=c, d_vs_baza=a - base, d_vs_harj=a - harj,
                             p_k2=p_k2 if tip == "H2" else np.nan))
        v = np.array([ms(r, g) for r in k2])
        rows.append(dict(stroka="контроль: к2 (сдвиг маски σ̂, среднее 100)", tip="kontrol", god=g, msr=v[:, 0].mean(), sharpe=v[:, 1].mean(),
                         sdelok=v[:, 2].mean(), d_vs_baza=v[:, 0].mean() - base, d_vs_harj=v[:, 0].mean() - harj, p_k2=np.nan))
    return pd.DataFrame(rows)


def main() -> int:
    from fastbt import a_strogaya as AS
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = AS.evaluatory("S3")[1]
    baza = cd.schitat(ev)
    ry = pd.read_parquet(AS.OUT / "ryad_S3.parquet")
    assert np.allclose(baza["x"], ry["excess_zaem5"].to_numpy()[np.isin(pd.to_datetime(ry["date"]).dt.year, AS.GODY)], atol=1e-9), \
        "база не совпала с ryad_S3.parquet"
    ser = {k: cd.citat_sigma(v) for k, v in FILES.items()}
    allow = {k: kesh(s) for k, s in ser.items()}
    res = {"baza": baza, **{k: cd.schitat(ev, sig_fn_poz(a)) for k, a in allow.items()}}
    rng, T = np.random.default_rng(cd.SEED), ev.sk.tickers
    k2 = [cd.schitat(ev, sig_fn_poz(cd.k2_mask(allow["hyb"], rng))) for _ in range(cd.N_K2)]
    msr_k2 = np.array([cd.stroki_msr(r, "2018-2019") for r in k2])
    p_k2 = float((msr_k2 >= cd.stroki_msr(res["hyb"], "2018-2019")).mean())
    df = tablica(res, k2, p_k2)
    df.to_csv(OUT / "test_c.csv", index=False)
    dts = {t: ev.inp.meta[t]["dates"][ev.inp.meta[t]["dates"] <= np.datetime64("2019-12-31")] for t in T}
    yr = lambda t: pd.DatetimeIndex(dts[t]).year.isin(cd.GODY)                              # noqa: E731
    arr = lambda k, t: ser[k][t].reindex(pd.DatetimeIndex(dts[t]), method="ffill").to_numpy(float)   # noqa: E731
    svod = {k: dict(zakryto=int(sum((~allow[k](t, dts[t]))[yr(t)].sum() for t in T)), dney=int(sum(yr(t).sum() for t in T)),
                    nan=int(sum(np.isnan([rang(arr(k, t), i) for i in np.flatnonzero(yr(t))]).sum() for t in T))) for k in allow}
    sverka = sum(int((allow["hyb"](t, dts[t]) != mask_s9(arr("hyb", t))).sum()) for t in T)
    cd.FILES = {"hyb": FILES["hyb"], "harj": FILES["harj"]}
    sd = cd.sdvig()
    sd.to_csv(OUT / "test_c_sdvig.csv", index=False)
    from cd_s3_test_c_report import napisat
    napisat(REPORT, df, sd, msr_k2, p_k2, svod, sverka, time.time() - t0)
    print(df.to_string(), p_k2, svod, "расхождений с s9:", sverka, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
