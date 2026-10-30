"""Л1.6 (Ж-146): драйвер голых идей S7–S12 (центры — docs/tables/golaya_v7/centry.csv, коммит до кода) с к2 и плацебо.

Периоды train4 и v2 (< 2020-01-01; перед чтением — proverit_chtenie; wf запрещён). Статус: разработка, до аудита, потолок.
Цены и D (замечание 1): исполнитель получает НЕскорректированные open 10:00 и close 18:50 и D = дивиденд ПОСЛЕ НДФЛ (gap_divs × 0,87,
свод Д8) на день отсечки; D дня без цены тикера переносится на ближайший день тикера. Сигналы S7–S9 и S10–S12 — на скорректированных
ценах (adjust_ohlc; 10-минутки × множитель дня дневки, mutations № 33), исполнение — на реальных. Вес капитала на тикер 1/N. S7–S9 — состояние: каждый
отрезок знака — сделка, выход по смене знака (open дня после). S10–S12: выход через τ по close; флаги exit обёртки v7_impuls НЕ
подаются (замечание 2: флаг пропущенного сигнала закрывает позицию нового по open раньше τ — тест). Справочная книга S7–S9 —
kross_seksiya.kniga, сверена с golaya_s7_s9/svod.csv. Контроли, шапка, таблицы и запуск — golaya_v7_tablicy.py (лимит 200 строк).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import bars_agg, daily_1850 as d18, dividendy, impuls_v7 as imp, kross_seksiya as ks, kross_v7 as kv  # noqa: E402
from fastbt import metriki_diag as md, metriki_e as me, msp, progon_kontrol as pk, t_reshenie_v7 as tr  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402
from fastbt.golaya_v7_isp import KOMISSIYA, ZAEM, ispolnit  # noqa: E402
from fastbt.period_v2 import PERIODY, proverit_chtenie  # noqa: E402

END, PERIODY_OK, T_DOP, N_POVT, SEED = pd.Timestamp("2020-01-01"), ("train4", "v2"), ("17:00", "18:00", "18:40"), 20, 20261031
SEKTORY, SECH, IMPULS = ROOT / "docs" / "issledovaniya" / "2026-10-27_sektory.csv", ("S7", "S8", "S9"), ("S10", "S11", "S12")
_C = dict(porog=0.5, bufer=0.0, cap=3)
CENTR = {"S7": dict(rezhim="den", **_C), "S8": dict(rezhim="nedelya", **_C), "S9": dict(rezhim="nedelya", **_C),
         **{s: dict(tau=2, vhod="d1", T="18:30") for s in IMPULS}}


def chitat_imoex(period: str) -> pd.DataFrame:
    d = Path(d18.default_main()) / "data" / "iss_mix_2015_2019"
    fs = ([d / "IMOEX_2014.csv"] if PERIODY[period].progrev.year < 2015 else []) + [d / "IMOEX.csv"]
    x = pd.concat([pd.read_csv(f, usecols=["TRADEDATE", "OPEN", "CLOSE"], parse_dates=["TRADEDATE"]).set_index("TRADEDATE") for f in fs])
    return x[~x.index.duplicated()].sort_index()


def _korr10(b: pd.DataFrame, f: pd.Series) -> pd.DataFrame:
    """10-минутки × множитель корректировки дня (тот же, что у дневки): сигналы S10–S12 на скорректированном ряду."""
    k = f.reindex(pd.DatetimeIndex(b["timestamp"].dt.normalize())).fillna(1.0).to_numpy()
    return b.assign(**{c: b[c] * k for c in ("open", "high", "low", "close")})


def iz_barov(raw: dict, im: pd.DataFrame, period: str, tk=None) -> dict:
    """Входы драйвера из 10-минуток (timestamp, OHLCV) по тикерам и IMOEX (OPEN, CLOSE)."""
    P, tk = PERIODY[period], list(tk or TICKERS)
    cl, op, cr, dv, ist, rw = {}, {}, {}, {}, {}, {}
    for t in tk:
        b = raw[t][lambda x: (x["timestamp"] >= P.progrev) & (x["timestamp"] < END)]
        dd = d18.aggregate(b, t).set_index("date")
        dts, cr[t] = dd.index.values, dd["close"]
        f = pd.Series(dividendy.adjust_ohlc(t, dts, dd["close"].to_numpy(), np.ones(len(dd)))[0], dd.index)    # множитель дня
        cl[t], dv[t] = dd["close"] * f, pd.Series(dividendy.gap_divs(t, dts) * dividendy.DIV_NALOG, dd.index)
        b10 = b[b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute == 600]
        op[t], ist[t], rw[t] = b10.groupby(b10["timestamp"].dt.normalize())["open"].first(), bars_agg.prep_10m(_korr10(b, f)), b
    c = pd.DataFrame(cl)[tk]
    c = c[c.notna().sum(axis=1) >= 8]
    daty, im = pd.DatetimeIndex(c.index), im[im.index < END].reindex(c.index)
    f = lambda d: pd.DataFrame(d)[tk].reindex(c.index)  # noqa: E731
    karta, o, y0 = pd.read_csv(SEKTORY, usecols=["ticker", "indeks"]).assign(s=pd.Timestamp("2000-01-01")), f(op), P.y0
    return dict(daty=daty, tk=tk, c=c, o=o, cr=f(cr), div=f(dv).fillna(0.0), im=im, ist=ist, raw=rw, period=period,
                div_nalog=dividendy.DIV_NALOG, metki=ks.sektor_na_d(karta, daty, tk), stavka=msp.rate_on(daty).to_numpy(),
                sigma=np.log(c).diff().rolling(22, min_periods=15).std().to_numpy(), rm=(im["OPEN"].shift(-1) / im["OPEN"] - 1).to_numpy(),
                ok=o.notna().shift(-ks.LAG, fill_value=False).to_numpy(), gody=tuple(range(y0, P.konec.year + 1)),
                win=np.asarray((daty >= pd.Timestamp(f"{y0}-01-01")) & (daty <= P.konec)),
                i0=int(np.searchsorted(daty.values, np.datetime64(f"{y0}-01-01"))) - 1, _sig={}, _pT={})


def zagruzit(period: str) -> dict:
    if period not in PERIODY_OK:
        raise ValueError(f"период {period!r}: в этом листе только {PERIODY_OK}")
    proverit_chtenie(period)
    src = Path(d18.default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m"
    return iz_barov({t: d18.read_bars(src / f"{t}.parquet", END) for t in TICKERS}, chitat_imoex(period), period)


def shag(sid: str, vhod: str = "d1", T: str | None = None) -> dict:
    """Параметры исполнения точки: S7–S9 — состояние; S10–S12 — выход через τ."""
    if sid in SECH:
        return dict(vhod="d1", tau=None, sost=True, T=None)
    return dict(vhod=vhod, tau=CENTR[sid]["tau"], sost=False, T=T if vhod == "T" else None)


def signaly(D: dict, sid: str, ctl=None, tochka: dict | None = None) -> np.ndarray:
    """int8[T, N] ±1/0 в день решения. ctl: None | ("k2", лаг) | ("placebo", seed). Только данные ≤ дня решения."""
    if sid in SECH:
        c = CENTR[sid]
        sk = kv.rebal(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], sid), D["daty"], c["rezhim"])
        if ctl:
            sk = kv.k2(sk, ctl[1]) if ctl[0] == "k2" else kv.placebo_sechenie(sk, ctl[1])
        return kv.storona(sk, D["sigma"], D["metki"], D["ok"], c["porog"], c["bufer"], c["cap"])
    tochka = tochka or shag(sid)
    ts = CENTR[sid]["T"] if tochka["vhod"] == "d1" else tochka["T"]
    if (sid, ts) not in D["_sig"]:
        D["_sig"][(sid, ts)] = np.column_stack([imp.signal_dnya(D["ist"][t], sid, ts).reindex(D["daty"]).fillna(0).to_numpy()
                                                for t in D["tk"]]).astype(np.int8)
    S = D["_sig"][(sid, ts)]
    if ctl:
        fn = pk.k2_sig_fn(ctl[1]) if ctl[0] == "k2" else pk.placebo_sig_fn(ctl[1])
        S = np.column_stack([fn(t, S[:, j], D["daty"]) for j, t in enumerate(D["tk"])]).astype(np.int8)
    return S


def _izb(sd, dts, rate, Dv) -> list:
    """Сделки на единицу цены входа: (до издержек: ход + D; комиссия на обе стороны; заём шорта; ставка) за ночи e…x−1."""
    g = np.append(np.diff(dts.values.astype("datetime64[D]")).astype(float), 1.0) / ks.DNEI
    cd, cn, zs = (np.cumsum(np.r_[0.0, a]) for a in (Dv, rate * g, g * ZAEM))
    return [(s * ((px - pe) + cd[x + 1] - cd[e + 1]) / pe, KOMISSIYA * (1 + px / pe), (zs[x] - zs[e]) if s < 0 else 0.0, cn[x] - cn[e])
            for s, e, x, pe, px in sd]


def _runy(s, o, c, dts, rate, Dv, pT, vhod, tau, sost) -> list:
    kw = dict(dates=dts, rate=rate, D=Dv)
    if not sost:
        return [ispolnit(s, o, c, pT, vhod, tau, **kw)]
    runs, res = pk.runs_of(s), []
    for par in (0, 1):                                      # соседние отрезки — в разных вызовах (смена знака в тот же open)
        m, fin = np.zeros(len(s), np.int8), np.zeros(len(s), bool)
        for k, (st, ln, val) in enumerate(runs):
            if k % 2 == par:
                m[st], fin[min(st + ln, len(s) - 1)] = int(val), st + ln < len(s)
        res.append(ispolnit(m, o, c, None, "d1", None, fin, fin, **kw))
    return res


def schet(D: dict, sig: np.ndarray, vhod: str = "d1", tau=None, sost: bool = False, T: str | None = None) -> dict:
    """Дневные ряды портфеля (вес тикера 1/N) через исполнитель; sd — сделки (день входа, избыточное PnL, дней)."""
    n, N = len(D["daty"]), len(D["tk"])
    o = {k: np.zeros(n) for k in ("pnl", "kom", "zaem", "stavka", "gross", "vhody")}
    o["sd"], o["tr"], cs = [], [], np.cumsum(D["div"].to_numpy(), axis=0)
    if vhod == "T" and T not in D["_pT"]:
        D["_pT"][T] = np.column_stack([tr.cena_T(D["raw"][t], T).set_index("date")["cena"].reindex(D["daty"]).to_numpy() for t in D["tk"]])
    pT = D["_pT"][T] if vhod == "T" else None
    for j in range(N):
        oo, cc = D["o"].iloc[:, j].to_numpy(), D["cr"].iloc[:, j].to_numpy()
        idx = np.flatnonzero((oo > 0) & (cc > 0))                       # дни без цены у тикера пропущены (Д4: без ffill)
        if not len(idx) or not sig[idx, j].any():
            continue
        dts, rate, Dv = D["daty"][idx], D["stavka"][idx], np.diff(np.r_[0.0, cs[idx, j]])
        for r in _runy(sig[idx, j], oo[idx], cc[idx], dts, rate, Dv, None if pT is None else pT[idx, j], vhod, tau, sost):
            for k in ("pnl", "kom", "zaem", "stavka"):
                o[k][idx] += r[k] / N
            o["gross"][idx] += np.abs(r["poz"]) / N
            for (s, e, x, pe, px), z in zip(r["sdelki"], _izb(r["sdelki"], dts, rate, Dv)):       # tr: журнал сделок (на единицу, вес 1/N)
                o["sd"].append((int(idx[e]), z[0] - z[1] - z[2] - z[3], x - e))
                o["tr"].append((j, int(idx[e - 1] if vhod == "d1" else idx[e]), int(idx[e]), int(idx[x]), s, *z))
                o["vhody"][idx[e]] += 1
    o["excess"] = o["pnl"] - o["kom"] - o["zaem"] - o["stavka"]
    return o


def sila(D: dict, sid: str):
    """|скор| S7–S9 на день решения [T, N] (после ребалансировки); у бинарных S10–S12 силы нет — None."""
    if sid not in SECH:
        return None
    return np.abs(kv.rebal(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], sid), D["daty"], CENTR[sid]["rezhim"]))


def kniga_spravka(D: dict, sid: str) -> dict:
    """Нейтральная книга S7–S9 (kross_seksiya.kniga, траншевая) — как в golaya_s7_s9."""
    tau, n = ks.TAU[sid]
    W = ks.vesa(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], sid), D["sigma"], D["metki"], D["ok"])
    return {**ks.kniga(W, D["o"].ffill().to_numpy(), D["div"].to_numpy(), D["daty"].values, D["stavka"], tau, n, D["i0"]), "sd": []}


def mera(o: dict, D: dict) -> float:
    """Мера контролей: дневной mSR, если определён, иначе недельный (Ж-124 п. 3)."""
    x, m = o["excess"][D["win"]], me.msr(o["excess"][D["win"]])
    return float(m) if np.isfinite(m) else float(me.metriki_ned(x, D["daty"][D["win"]])["msr_ned"])


def stroka(o: dict, D: dict) -> dict:
    w, yr, x, L = D["win"], D["daty"].year[D["win"]], o["excess"][D["win"]], len(D["gody"])
    rm = D["rm"][w]
    ok = np.isfinite(rm)
    r = {"mSR": me.msr(x), "msr_ned": me.metriki_ned(x, D["daty"][w])["msr_ned"]}
    r.update({f"mSR_{y}": me.msr(x[yr == y]) for y in D["gody"]})
    v = [r[f"mSR_{y}"] for y in D["gody"] if np.isfinite(r[f"mSR_{y}"])]
    var = np.var(rm[ok], ddof=1) if ok.sum() > 2 else 0.0
    r.update(mSR_min_god=min(v) if v else float("nan"), dolya_vremeni=float((o["gross"][w] > 0).mean()), sdelok_v_god=float(o["vhody"][w].sum() / L),
             beta=float(np.cov(x[ok], rm[ok])[0, 1] / var) if var > 0 else float("nan"),
             dolya_dispersii_rynka=float(np.corrcoef(x[ok], rm[ok])[0, 1] ** 2) if var > 0 and x[ok].std() > 0 else float("nan"))
    for k in ("pnl", "kom", "zaem", "stavka", "excess"):
        r[k + "_pct_god"] = float(o[k][w].sum() / L * 100)
    sd = [t for t in o["sd"] if w[t[0]]]
    dg = md.panel_diag(D["daty"][w], x, o["gross"][w], [D["daty"][t[0]].year for t in sd], [t[1] for t in sd], [t[2] for t in sd])
    return {**r, **{k: v for k, v in dg.loc["итого"].items() if k in (md.K_NORM, md.K_SDELOK, md.K_CENA, "причина NaN")}}


