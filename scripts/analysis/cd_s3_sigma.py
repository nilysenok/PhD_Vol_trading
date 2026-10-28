"""C/D-разрез по σ для S3 (Ж-096 п. 5г; H2 Ж-088 п. 2). Разработка train4, окно 2018–2019, потолок, H2-строка.
Запуск (один процесс): python3 scripts/analysis/cd_s3_sigma.py   [FASTBT_MAIN=<копия с data/>]
База — A_strogaya «только S3» (fastbt.a_strogaya, точка A). Затвор: не входить, если ранг σ тикера в день сигнала >= p67
(доля значений окна 252, строго ниже; нужно >= 126; ряд до дня сигнала включительно — решение на close d). Открытую позицию затвор не закрывает.
Строки: реализованная σ (rv22, справочно) · σ̂ гибрида (H2) · контроли HAR-J (б), наивная √RV, к2 (круговой сдвиг маски по тикеру, 100 повторов).
Данные ≥ 2020 не читаются (фильтр при чтении). Метрики 2020–2021 не считаются."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MAIN = Path(os.environ.get("CD_S3_DATA", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))   # data/regen: только чтение
CH4, LAST = MAIN / "data" / "regen" / "ch4_dev", pd.Timestamp("2019-12-31")
OUT, REPORT = ROOT / "docs/tables/cd_s3", ROOT / "docs/otchety/2026-10-28_cd_s3_sigma.md"
P, OKNO, MIN_R, PLATA, N_K2, SEED, LAG_MIN = 0.67, 252, 126, 0.05, 100, 20261028, 21
GODY, SDVIGI = (2018, 2019), (-2, -1, 0, 1)
FILES = {"hyb": "sigma_hyb_h1_dev", "harj": "sigma_harj_b_h1_dev"}


def rang(arr: np.ndarray, i: int) -> float:
    """Ранг arr[i] в окне arr[i−251..i] (доля строго ниже). Строки после i не читаются. < 126 значений или nan — nan."""
    w = arr[max(i - OKNO + 1, 0):i + 1]
    w = w[np.isfinite(w)]
    if i < 0 or len(w) < MIN_R or not np.isfinite(arr[i]):
        return float("nan")
    return float((w < arr[i]).sum()) / len(w)


def zatvor(arr, p: float = P) -> np.ndarray:
    """allow[i] = вход по решению на close i разрешён: ранг < p (ранг не определён — разрешён, число таких дней считает nan_dni)."""
    arr = np.asarray(arr, float)
    return np.array([not (rang(arr, i) >= p) for i in range(len(arr))], bool)


def nan_dni(arr, dates) -> int:
    """Дни 2018–2019, где ранг не определён (затвор в них открыт по умолчанию)."""
    yr = pd.DatetimeIndex(dates).year.to_numpy()
    arr = np.asarray(arr, float)
    return int(sum(np.isnan(rang(arr, i)) for i in np.flatnonzero(np.isin(yr, GODY))))


def sig_fn(allow_of):
    """sig_fn для evaluate_v2: allow_of(t, dates) -> bool[len(dates)]; сделка пропускается целиком, если вход запрещён."""
    from fastbt.etap1_data import gate_trades

    def fn(t, sig, dates):
        a = np.asarray(allow_of(t, dates), bool)
        if len(a) != len(sig):
            raise ValueError("маска затвора и позиции разной длины")
        return gate_trades(sig, a)
    return fn


def citat_sigma(name: str) -> dict:
    """{тикер: Series(date → σ̂ на d+1, известная на close d)}; только <= 2019-12-31."""
    d = pd.read_parquet(CH4 / f"{name}.parquet", filters=[("date", "<=", LAST)])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= LAST]
    col = [c for c in d.columns if c.startswith("sigma")][0]
    assert d["date"].max() <= LAST, "дата > 2019"
    return {t: g.set_index("date")[col].sort_index() for t, g in d.groupby("ticker")}


def na_dates(ser: dict):
    return lambda t, dates: ser[t].reindex(pd.DatetimeIndex(dates)).to_numpy(float)


def schitat(ev, fn=None) -> dict:
    from fastbt import a_strogaya as AS
    row, x, tik = ev.evaluate_v2({"t_vyhod": "open2"}, AS.PER, {"sig_fn": fn} if fn else {})
    _, _, dts, _, _, rdates = ev._series(AS.PER, ev.comm)
    rs = ev._last[4]
    return dict(x=x - PLATA * np.asarray(ev.zaem), dates=dts.to_numpy().astype("datetime64[D]"),
                cnt=np.concatenate([r.cnt for r in rs]), god=rdates.year.to_numpy())


def stroki(imya: str, tip: str, r: dict, baza: dict | None, **extra) -> list:
    from fastbt import metriki_e as me
    yr = r["dates"].astype("datetime64[Y]").astype(int) + 1970
    out = []
    for g in (*GODY, "2018-2019"):
        m, mr = ((yr == g, r["god"] == g) if g != "2018-2019" else (np.isin(yr, GODY), np.isin(r["god"], GODY)))
        msr, sd = me.msr(r["x"][m]), float(r["cnt"][mr].sum()) / 2
        d = np.nan if baza is None else msr - stroki_msr(baza, g)
        out.append(dict(stroka=imya, tip=tip, god=g, msr=msr, sdelok=sd, d_vs_baza=d, **extra))
    return out


def stroki_msr(r: dict, g) -> float:
    from fastbt import metriki_e as me
    yr = r["dates"].astype("datetime64[Y]").astype(int) + 1970
    return me.msr(r["x"][np.isin(yr, GODY) if g == "2018-2019" else yr == g])


def k2_mask(allow_of, rng):
    """к2: круговой сдвиг маски затвора тикера на lag ∈ [21, n−21] дней; последний день не сдвигается (как k2_sig_fn)."""
    def fn(t, dates):
        a = np.asarray(allow_of(t, dates), bool)
        n = len(a)
        k = int(rng.integers(LAG_MIN, max(n - LAG_MIN, LAG_MIN + 1)))
        out = a[(np.arange(n) - k) % n] if n else a.copy()
        if k:
            out[-1] = a[-1]
        return out
    return fn


def sdvig() -> pd.DataFrame:
    """Д1: корреляция σ̂ (строка d) с √RV дня d+1+k при k ∈ −2…+1, пул по тикерам, 2018–2019; уровни и приращения (оконные ряды — по приращениям).
    Последний известный день на close d — сам d, т.е. k = −1: максимум обязан быть там."""
    rv = pd.read_parquet(MAIN / "data/regen/features_pit_v3.parquet", columns=["date", "ticker", "rv"], filters=[("date", "<=", LAST)])
    rv["date"], rv["s"] = pd.to_datetime(rv["date"]), np.sqrt(rv["rv"].astype(float))
    rv = rv.sort_values(["ticker", "date"])
    ser = {**{k: citat_sigma(v) for k, v in FILES.items()}, "naive": citat_sigma("sigma_naive_dev")}
    rows = []
    for nm, d in ser.items():
        f = pd.concat([s.rename("sg").rename_axis("date").reset_index().assign(ticker=t) for t, s in d.items()])
        f = f.sort_values(["ticker", "date"]).assign(dsg=lambda z: z.groupby("ticker")["sg"].diff())
        for k in SDVIGI:
            g = rv.groupby("ticker")["s"]
            rv["tgt"], rv["dtgt"] = g.shift(-(1 + k)), g.shift(-(1 + k)) - g.shift(-k)
            j = f.merge(rv[["date", "ticker", "tgt", "dtgt"]], on=["date", "ticker"]).dropna()
            j = j[j["date"].dt.year.isin(GODY)]
            rows.append(dict(ryad=nm, sdvig=k, corr=float(j["sg"].corr(j["tgt"])), corr_prirasch=float(j["dsg"].corr(j["dtgt"])), n=len(j)))
    return pd.DataFrame(rows)


def main() -> int:
    from fastbt import a_strogaya as AS
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = AS.evaluatory("S3")[1]
    baza = schitat(ev)
    ry = pd.read_parquet(AS.OUT / "ryad_S3.parquet")
    ref = ry["excess_zaem5"].to_numpy()[np.isin(pd.to_datetime(ry["date"]).dt.year, AS.GODY)]
    assert np.allclose(baza["x"], ref, atol=1e-9), "база не совпала с ryad_S3.parquet"
    rv22 = lambda t, dates: zatvor(ev.inp.sigma["rv22"][t][:len(dates)])                         # noqa: E731
    mod = {k: na_dates(citat_sigma(v)) for k, v in {**FILES, "naive": "sigma_naive_dev"}.items()}
    zat = {k: (lambda t, d, f=f: zatvor(f(t, d))) for k, f in mod.items()}
    res = {"rv22": schitat(ev, sig_fn(rv22)), **{k: schitat(ev, sig_fn(z)) for k, z in zat.items()}}
    rng, T = np.random.default_rng(SEED), ev.sk.tickers
    k2 = []
    for _ in range(N_K2):
        k2.append(schitat(ev, sig_fn(k2_mask(zat["hyb"], rng))))
    msr_k2 = np.array([stroki_msr(r, "2018-2019") for r in k2])
    p_k2 = float((msr_k2 >= stroki_msr(res["hyb"], "2018-2019")).mean())
    names = {"rv22": ("затвор по реализованной σ (rv22), справочно", "spravochno"), "hyb": ("затвор по σ̂ гибрида (H2)", "H2"),
             "harj": ("контроль: затвор по HAR-J (б)", "kontrol"), "naive": ("контроль: затвор по наивной √RV", "kontrol")}
    rows = stroki("A_strogaya только S3 (база)", "baza", baza, None)
    for k, (nm, tip) in names.items():
        rows += stroki(nm, tip, res[k], baza, p_k2=p_k2 if k == "hyb" else np.nan)
    kr = [dict(stroka="контроль: к2 (сдвиг маски σ̂, среднее 100)", tip="kontrol", god=g, msr=float(np.mean([stroki_msr(r, g) for r in k2])),
               sdelok=float(np.mean([r["cnt"][np.isin(r["god"], GODY if g == "2018-2019" else (g,))].sum() / 2 for r in k2])))
          for g in (*GODY, "2018-2019")]
    for r in kr:
        r["d_vs_baza"] = r["msr"] - stroki_msr(baza, r["god"])
    df = pd.DataFrame(rows + kr)
    df.to_csv(OUT / "sigma_zatvor.csv", index=False)
    pd.DataFrame({"povtor": range(N_K2), "msr_2018_2019": msr_k2}).to_csv(OUT / "sigma_zatvor_k2.csv", index=False)
    sd = sdvig()
    sd.to_csv(OUT / "sigma_sdvig.csv", index=False)
    dts = {t: ev.inp.meta[t]["dates"][ev.inp.meta[t]["dates"] <= np.datetime64("2019-12-31")] for t in T}
    nan = {k: sum(nan_dni(f(t, dts[t]), dts[t]) for t in T) for k, f in mod.items()}
    from cd_s3_report import napisat
    napisat(REPORT, df, sd, msr_k2, p_k2, nan, time.time() - t0)
    print(df.to_string(), p_k2, nan, sep="\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
