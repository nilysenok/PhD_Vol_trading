"""Л2.2 (Ж-146 п. 2): голая идея S1–S4 × otnositelno ∈ {net, sektor, imoex} на центрах v5, с к2 и плацебо.

Центры — docs/tables/golaya_otnositelno_v7/centry.csv (коммит до кода; средний элемент оси, golaya_tf_v5.tsentr). Периоды train4 и v2
(< 2020-01-01, перед чтением — proverit_chtenie). Статус: разработка, до аудита, потолок. Сигналы — по скорректированным на дивиденды
дневным OHLCV (adjust_ohlc, mutations № 33), исполнение — реальные open/close + D (golaya_v7_isp); вход d+1 (Д2), выход — флаг exit
семьи (open дня после) либо последний close; вес тикера 1/N. Ось (fastbt.otnositelno_v7) меняет только ряд, на котором считается сигнал.
↺ Контроли двигают только входы (k2 — сдвиг сигнала, плацебо — случайные дни того же года); флаги выхода остаются правилом семьи.
↺ Относительный ряд тикера начинается после его последнего дня с неопределённым фоном (Д4: без подстановок); net — весь ряд.
Свой schet — потому что gv.schet не принимает флаги выхода; без флагов он ему равен побайтно (тест). Метрики, журнал (Ж-150) и шапка —
golaya_v7 / golaya_v7_tablicy.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7 as gv  # noqa: E402
import golaya_v7_tablicy as gt  # noqa: E402
from fastbt import daily_1850 as d18, dividendy, otnositelno_v7 as ot, progon_kontrol as pk  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402
from fastbt.golaya_v7_isp import ispolnit  # noqa: E402
from fastbt.indicators import compute_base  # noqa: E402
from fastbt.period_v2 import PERIODY, proverit_chtenie  # noqa: E402
from fastbt.signals import v5_s12, v5_s34  # noqa: E402,F401  (регистрация семей v5)
from fastbt.signals.interface import TickerData, get_family  # noqa: E402

SIDS = ("S1", "S2", "S3", "S4")
CENTRY = ROOT / "docs" / "tables" / "golaya_otnositelno_v7" / "centry.csv"
CENTR = {r.sid: json.loads(r.tochka) for r in pd.read_csv(CENTRY, dtype=str).itertuples()}


def iz_barov(raw: dict, im: pd.DataFrame, period: str, tk=None) -> dict:
    """D драйвера golaya_v7 + td[тикер] = (скорректированный TickerData на своих днях, позиции дней в D['daty'])."""
    D, P = gv.iz_barov(raw, im, period, tk), PERIODY[period]
    D.update(td={}, _sig={}, _fon={}, obrez={})
    for t in D["tk"]:
        b = raw[t][lambda x: (x["timestamp"] >= P.progrev) & (x["timestamp"] < gv.END)]
        dd = d18.aggregate(b, t).set_index("date").reindex(D["daty"])
        ok = dd[["open", "high", "low", "close", "volume"]].notna().all(axis=1).to_numpy()
        dts, d = D["daty"].values[ok], dd[ok]
        o, h, lo, c = dividendy.adjust_ohlc(t, dts, d["close"].to_numpy(), *(d[k].to_numpy() for k in ("open", "high", "low", "close")))
        D["td"][t] = (TickerData(t, dts.astype("datetime64[ns]"), o, h, lo, c, d["volume"].to_numpy(float),
                                 np.r_[np.log(c[1:] / c[:-1]), 0.0]), np.flatnonzero(ok))
    return D


def zagruzit(period: str) -> dict:
    if period not in gv.PERIODY_OK:
        raise ValueError(f"период {period!r}: в этом листе только {gv.PERIODY_OK}")
    proverit_chtenie(period)
    src = Path(d18.default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m"
    return iz_barov({t: d18.read_bars(src / f"{t}.parquet", gv.END) for t in TICKERS}, gv.chitat_imoex(period), period)


def _fon(D: dict, rezhim: str):
    if rezhim not in D["_fon"]:
        D["_fon"][rezhim] = ot.fon_ryad(D["c"], D["im"]["CLOSE"], D["metki"], rezhim)
    return D["_fon"][rezhim]


def _srez(td: TickerData, k: int) -> TickerData:
    return dataclasses.replace(td, dates=td.dates[k:], open=td.open[k:], high=td.high[k:], low=td.low[k:], close=td.close[k:],
                               volume=td.volume[k:], log_ret=td.log_ret[k:])


def _nachalo(td: TickerData, fon_t: np.ndarray) -> int:
    """Первый день относительного ряда: фон со 2-го дня должен быть определён (Д4)."""
    bad = np.flatnonzero(~np.isfinite(fon_t[1:]))
    return int(bad[-1]) + 1 if len(bad) else 0


def signaly(D: dict, sid: str, rezhim: str = "net", ctl=None):
    """(sig int8, exit_l bool, exit_s bool) [T, N] в день решения. ctl: None | ("k2", лаг) | ("placebo", seed). Только данные ≤ дня."""
    if (sid, rezhim) not in D["_sig"]:
        fon, flag = (None, None) if rezhim == "net" else _fon(D, rezhim)
        fn = ot.semya(get_family(sid, "v5"), rezhim, fon, flag)
        S, EL, ES = (np.zeros(D["c"].shape, a) for a in (np.int8, bool, bool))
        for j, t in enumerate(D["tk"]):
            td, ix = D["td"][t]
            k = 0 if rezhim == "net" else _nachalo(td, fon[t].reindex(pd.DatetimeIndex(td.dates)).to_numpy())
            D["obrez"][(rezhim, t)] = k
            td, ix = _srez(td, k), ix[k:]
            ss = fn(td, compute_base(td.close, td.high, td.low, td.open, td.volume), [CENTR[sid]])
            el, es = ss.exit_info(0) if ss.split else (ss.exit_l[0], ss.exit_l[0])
            S[ix, j], EL[ix, j], ES[ix, j] = ss.sig[0], el, es
        D["_sig"][(sid, rezhim)] = (S, EL, ES)
    S, EL, ES = D["_sig"][(sid, rezhim)]
    if ctl:
        f = pk.k2_sig_fn(ctl[1]) if ctl[0] == "k2" else pk.placebo_sig_fn(ctl[1])
        S = np.column_stack([f(t, S[:, j], D["daty"]) for j, t in enumerate(D["tk"])]).astype(np.int8)
    return S, EL, ES


def schet(D: dict, sig: np.ndarray, EL=None, ES=None) -> dict:
    """Как gv.schet (vhod d1, не состояние), но с флагами выхода семьи. sd, tr — журнал сделок (вес 1/N)."""
    n, N = len(D["daty"]), len(D["tk"])
    o = {k: np.zeros(n) for k in ("pnl", "kom", "zaem", "stavka", "gross", "vhody")}
    o["sd"], o["tr"], cs = [], [], np.cumsum(D["div"].to_numpy(), axis=0)
    for j in range(N):
        oo, cc = D["o"].iloc[:, j].to_numpy(), D["cr"].iloc[:, j].to_numpy()
        idx = np.flatnonzero((oo > 0) & (cc > 0))                       # дни без цены пропущены (Д4: без ffill)
        if not len(idx) or not sig[idx, j].any():
            continue
        dts, rate, Dv = D["daty"][idx], D["stavka"][idx], np.diff(np.r_[0.0, cs[idx, j]])
        fl = [None if x is None else x[idx, j] for x in (EL, ES)]
        r = ispolnit(sig[idx, j], oo[idx], cc[idx], None, "d1", None, *fl, dates=dts, rate=rate, D=Dv)
        for k in ("pnl", "kom", "zaem", "stavka"):
            o[k][idx] += r[k] / N
        o["gross"][idx] += np.abs(r["poz"]) / N
        for (s, e, x, pe, px), z in zip(r["sdelki"], gv._izb(r["sdelki"], dts, rate, Dv)):
            o["sd"].append((int(idx[e]), z[0] - z[1] - z[2] - z[3], x - e))
            o["tr"].append((j, int(idx[e - 1]), int(idx[e]), int(idx[x]), s, *z))
            o["vhody"][idx[e]] += 1
    o["excess"] = o["pnl"] - o["kom"] - o["zaem"] - o["stavka"]
    return o


def stroka_ryad(D: dict, S, EL=None, ES=None) -> dict:
    return gv.stroka(schet(D, S, EL, ES), D)


def kontrol(D: dict, sid: str, rezhim: str, tip: str, n: int) -> list:
    par = pk.k2_lags(n) if tip == "k2" else [gv.SEED + i for i in range(n)]
    return [gv.mera(schet(D, *signaly(D, sid, rezhim, (tip, p))), D) for p in par]


def tablicy(D: dict, period: str, out: Path, n: int = gv.N_POVT) -> None:
    svod, k2, pl, zh = [], [], [], []
    for sid in SIDS:
        for rz in ot.OTNOSITELNO:
            S, EL, ES = signaly(D, sid, rz)
            o = schet(D, S, EL, ES)
            zh.append(gt.zhurnal(D, sid, o, f"centr:{rz}", {"vhod": "d1"}))
            svod.append({"strategiya": sid, "otnositelno": rz, "variant": "centr", "obrezano_dn": sum(D["obrez"][(rz, t)] for t in D["tk"]),
                         **gv.stroka(o, D)})
            fakt = gv.mera(o, D)
            for tip, rows in (("k2", k2), ("placebo", pl)):
                v = np.array(kontrol(D, sid, rz, tip, n))
                rows.append(dict(strategiya=sid, otnositelno=rz, centr=fakt, mediana=float(np.nanmedian(v)), min=float(np.nanmin(v)),
                                 max=float(np.nanmax(v)), dolya_ne_huzhe_centra=float(np.mean(v >= fakt)), p_perm=pk.p_perm(fakt, v), n=n))
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(CENTRY.read_bytes()).hexdigest()
    h = "# " + json.dumps({**gt.shapka(D, period), "v": "7.0-gol-otn", "centry_sha": sha, "oci": list(ot.OTNOSITELNO)}, ensure_ascii=False) + "\n"
    for nm, rows in (("svod", svod), ("k2", k2), ("placebo", pl)):
        (out / f"{nm}_{period}.csv").write_text(h + pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    (out / f"sdelki_{period}.csv").write_text(h + pd.concat(zh, ignore_index=True).to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=gv.PERIODY_OK)
    ap.add_argument("--out", default=str(ROOT / "docs" / "tables" / "golaya_otnositelno_v7"))
    a = ap.parse_args()
    tablicy(zagruzit(a.period), a.period, Path(a.out))
