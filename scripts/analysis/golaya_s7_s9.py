"""Голая идея S7–S9 на train4 (Ж-096 п. 5б): центр каждой, строгое исполнение, к2 — 20 циклических сдвигов.

Только 2016–2019 (прогрев 2015). 10-минутки читаются с фильтром timestamp < 2020-01-01 (данные ≥ 2020 не читаются).
Запуск: python3 scripts/analysis/golaya_s7_s9.py [--out DIR]; таблицы — docs/tables/golaya_s7_s9/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import daily_1850 as d18, dividendy, kross_seksiya as ks, metriki_e as me, msp  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402

END, START, T0 = pd.Timestamp("2020-01-01"), pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01")
IMOEX = Path(d18.default_main()) / "data" / "iss_mix_2015_2019" / "IMOEX.csv"
SEKTORY = ROOT / "docs" / "issledovaniya" / "2026-10-27_sektory.csv"
POVTOROV, SEED = 20, 20261028


def chitat_imoex() -> pd.DataFrame:
    """Дневной IMOEX (индекс TRADEDATE, колонки OPEN, CLOSE): csv ISS по пути IMOEX или готовый DataFrame (test1, Z2)."""
    if isinstance(IMOEX, pd.DataFrame):
        return IMOEX[["OPEN", "CLOSE"]].copy()
    return pd.read_csv(IMOEX, usecols=["TRADEDATE", "OPEN", "CLOSE"], parse_dates=["TRADEDATE"]).set_index("TRADEDATE")


def zagruzit() -> dict:
    """close/open 10:00/дивиденд по тикерам из 10-минуток до 2019-12-31; сигналы — на ряду с поправкой на дивиденд."""
    src = Path(d18.default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m"
    cl, op, dv = {}, {}, {}
    for t in TICKERS:
        b = d18.read_bars(src / f"{t}.parquet", END)
        b = b[b["timestamp"] >= START]
        dd = d18.aggregate(b, t).set_index("date")
        dates = dd.index.values
        cl[t] = pd.Series(dividendy.adjust_ohlc(t, dates, dd["close"].to_numpy(), dd["close"].to_numpy())[0], dd.index)
        dv[t] = pd.Series(dividendy.gap_divs(t, dates) * dividendy.DIV_NALOG, dd.index)
        b10 = b[b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute == 600]
        op[t] = b10.groupby(b10["timestamp"].dt.normalize())["open"].first()
    c = pd.DataFrame(cl)[TICKERS]
    c = c[c.notna().sum(axis=1) >= 8]
    im = chitat_imoex()
    im = im[im.index < END].reindex(c.index)
    return dict(c=c, o=pd.DataFrame(op)[TICKERS].reindex(c.index), d=pd.DataFrame(dv)[TICKERS].reindex(c.index).fillna(0.0), im=im)


def karta() -> pd.DataFrame:
    k = pd.read_csv(SEKTORY, usecols=["ticker", "indeks"])
    k["s"] = pd.Timestamp("2000-01-01")                         # ❓ Д-4: единый состав (2017-03) на весь train4
    return k


def sigmy(c: pd.DataFrame) -> pd.DataFrame:
    return np.log(c).diff().rolling(22, min_periods=15).std()   # RV22 до d включительно


def skory(D: dict, metki: np.ndarray) -> dict:
    c, im = D["c"], D["im"]["CLOSE"]
    return {"S7": ks.skor_s7(c, im), "S8": ks.skor_s8(c), "S9": ks.skor_s9(c, im, metki)}


def schet(sid: str, skor: np.ndarray, D: dict, metki, sig, stavka, i0: int) -> dict:
    tau, n = ks.TAU[sid]
    ok = D["o"].notna().shift(-ks.LAG, fill_value=False).to_numpy()
    W = ks.vesa(skor, sig, metki, ok)
    return ks.kniga(W, D["o"].ffill().to_numpy(), D["d"].to_numpy(), D["c"].index.values, stavka, tau, n, i0)


def metriki(o: dict, dates: pd.DatetimeIndex, win: np.ndarray, rm: np.ndarray) -> dict:
    x, yr = o["excess"][win], dates.year[win]
    ok = np.isfinite(rm[win])
    beta = np.cov(x[ok], rm[win][ok])[0, 1] / np.var(rm[win][ok], ddof=1)
    r = {"mSR": me.msr(x), "mSR_min_god": min(me.msr(x[yr == y]) for y in range(2016, 2020))}
    r.update({f"mSR_{y}": me.msr(x[yr == y]) for y in range(2016, 2020)})
    r["mSR_bez_stavki_zanyatogo"] = me.msr((o["excess"] + o["stavka"])[win])        # справочно: порог ставки снят
    r.update(dolya_vremeni=float((o["gross"][win] > 0).mean()), sdelok_v_god=float(o["vhody"][win].sum() / 4),
             beta=float(beta), dolya_dispersii_rynka=float(np.corrcoef(x[ok], rm[win][ok])[0, 1] ** 2))
    for k in ("pnl", "kom", "zaem", "stavka", "excess"):
        r[k + "_pct_god"] = float(o[k][win].sum() / 4 * 100)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "tables" / "golaya_s7_s9"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    D = zagruzit()
    dates = pd.DatetimeIndex(D["c"].index)
    metki = ks.sektor_na_d(karta(), dates, TICKERS)
    sig, sk = sigmy(D["c"]).to_numpy(), skory(D, metki)
    stavka = msp.rate_on(dates).to_numpy()
    i0 = int(np.searchsorted(dates.values, np.datetime64(T0))) - 1
    win = np.asarray((dates >= T0) & (dates <= "2019-12-31"))
    imo = D["im"]["OPEN"]
    rm = (imo.shift(-1) / imo - 1).to_numpy()                   # open→open, как день книги
    rows, curves, k2 = {}, {}, []
    rng = np.random.default_rng(SEED)
    lags = rng.integers(60, len(dates) - 60, POVTOROV)
    for sid, s in sk.items():
        o = schet(sid, s.to_numpy(), D, metki, sig, stavka, i0)
        rows[sid] = metriki(o, dates, win, rm)
        curves[sid] = o["excess"][win]
        sh = [me.msr(schet(sid, ks.k2_skor(s.to_numpy(), int(g)), D, metki, sig, stavka, i0)["excess"][win]) for g in lags]
        k2.append(dict(strategiya=sid, centr=rows[sid]["mSR"], k2_mediana=float(np.nanmedian(sh)), k2_min=float(np.nanmin(sh)),
                       k2_max=float(np.nanmax(sh)), dolya_k2_ne_huzhe_centra=float(np.mean(np.array(sh) >= rows[sid]["mSR"]))))
    kr = pd.DataFrame(curves, index=dates[win])
    pd.DataFrame(rows).T.rename_axis("strategiya").to_csv(out / "svod.csv")
    pd.DataFrame(k2).to_csv(out / "k2.csv", index=False)
    pd.concat([kr, kr.cumsum().add_suffix("_nakopl")], axis=1).rename_axis("date").to_csv(out / "krivaya.csv")
    kr.corr().to_csv(out / "korr_S7_S9.csv")
    print(pd.DataFrame(rows).T.round(4).to_string(), "\n", pd.DataFrame(k2).round(3).to_string(), "\n", kr.corr().round(3), sep="")
    print("пропуски open 10:00 (дней×тикеров, заполнено ffill):", int(D["o"].isna().sum().sum()))


if __name__ == "__main__":
    main()
