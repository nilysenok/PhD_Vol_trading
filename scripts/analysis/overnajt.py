"""Овернайт-разложение по категориям (Ж-103 пп. 2, 4), train4 (2016–2019, прогрев 2015), справочно, без выбора.
Запуск: FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 scripts/analysis/overnajt.py  → docs/tables/overnajt/razlozhenie.csv.
S1–S6 — A_strogaya (fastbt/a_strogaya.py, точка A, движок не правится: подкласс запоминает сделки и считает части по ним, сверка с gross движка);
S7–S9 — центр кросс-секции (golaya_s7_s9 + fastbt/kross_seksiya.py, части — overnajt_lib.kniga_razlozhenie, сверка с pnl книги).
Определения: Шарп — ddof=1, √252 над excess с платой за шорт 5 % (как кросс-секция); «в деле» — только дни с позицией; без ФДР — над
торговой компонентой (g − комиссия − плата за заём, без ставки и без ФДР). Последняя дата окна A (день b — один бар) в долях не участвует."""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastbt import a_strogaya as AS, daily_1850 as d18, dividendy, etap1, exec_e_vremya as vr, kross_seksiya as ks, msp  # noqa: E402
from fastbt import progon_urovni as urovni  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402
from fastbt.etap1_eval import years_of  # noqa: E402
import overnajt_lib as L  # noqa: E402

OUT = ROOT / "docs" / "tables" / "overnajt" / "razlozhenie.csv"
KATEGORII = [("A_strogaya (S1–S6)", AS.SIDS), ("контртренд (S1, S2)", ("S1", "S2")), ("диапазон (S5, S6)", ("S5", "S6")),
             ("контртренд + диапазон (S1, S2, S5, S6)", ("S1", "S2", "S5", "S6")), ("тренд (S3, S4)", ("S3", "S4")),
             ("кросс-секция (S7–S9)", ("S7", "S8", "S9"))]


def open10(d) -> float:
    """open бара 10:00 дня (первый бар не раньше 10:00; времени нет — первый бар)."""
    k = vr.bar_at(d, vr.HHMM["open2"])
    return float(d.o[k if k is not None and k < len(d.o) else 0])


def klass():
    class Rec(AS.AStrogaya):
        """A_strogaya + запись частей по сделкам (движок не правится): komp[id(rows)] = (rows, intra, noch, max|intra+noch−g|)."""
        komp: dict = {}

        def _ladder(self, t, days, sig, sg, p, w, lest, kw):
            rows = super()._ladder(t, days, sig, sg, p, w, lest, kw)
            a, b = kw["win"]
            ix = [i for i, d in enumerate(days) if not d.pusto]
            o = np.array([np.nan if d.pusto else open10(d) for d in days])
            I, N = L.razlozhit_sdelki(rows.trades, w, o, kw["close"], kw["div"], dividendy.DIV_NALOG, ix, len(days))
            ecc = float(np.abs(I[a:b - 1] + N[a:b - 1] - rows.g[:b - a - 1]).max()) if b - a > 1 else 0.0
            self.komp[id(rows)] = (rows, I[a:b], N[a:b], ecc)
            return rows
    return Rec


def setka(ev, parts) -> np.ndarray:
    """Строки тикеров [b−a] → сетка [тикеров, T] дат train4 — как EvaluatorV2._series (NaN вне окна)."""
    df, inv, _, _, rs = ev._last
    keep = np.isin(df.index.year, years_of(AS.PER))
    col = np.cumsum(keep) - 1
    out, lo = np.full((len(rs), int(keep.sum())), np.nan), 0
    for ti, r in enumerate(rs):
        sl = slice(lo, lo + len(r.g))
        lo = sl.stop
        ok = keep[inv[sl]]
        out[ti, col[inv[sl]][ok]] = np.asarray(parts[ti])[ok]
    return out


def schitat_sk(sid: str) -> dict:
    inp, base = etap1.build_inputs(sid, period=AS.PER), etap1.base_point(sid)
    ev = klass()(sid, inp, base, AS.COMM)
    r = AS.schitat(ev, True)
    rs = ev._last[4]
    kp = [ev.komp[id(x)] for x in rs]
    assert all(k[0] is x for k, x in zip(kp, rs)), "части записаны не для тех строк, что в _last"
    r.update(I=setka(ev, [k[1] for k in kp]), N=setka(ev, [k[2] for k in kp]), g=setka(ev, [x.g for x in rs]),
             sides=setka(ev, [x.sides for x in rs]), eks=setka(ev, [x.expo for x in rs]), ecc=max(k[3] for k in kp))
    print(sid, "готово; макс. |части − g движка| =", f"{r['ecc']:.2e}", flush=True)
    return {k: r[k] for k in ("x", "tik", "baza", "dates", "I", "N", "g", "sides", "eks", "ecc")}


def stroki_a(R: dict, god: float = 4.0) -> list:
    """Строки категорий S1–S6 (и каждого Sk): средние по парам, как a_shtrih с равными весами; сверка excess с a_shtrih."""
    rows = []
    for imya, sids in [(s, (s,)) for s in AS.SIDS] + KATEGORII[:5]:
        tik = np.stack([R[s]["tik"] for s in sids])
        nn = np.isfinite(tik).sum((0, 1)).astype(float)
        agg = lambda k: np.stack([np.nan_to_num(R[s][k]) for s in sids]).sum((0, 1)) / nn      # noqa: E731
        baza = np.mean([R[s]["baza"] for s in sids], axis=0)
        x = np.nan_to_num(tik).sum((0, 1)) / nn - AS.PLATA * baza
        ref = urovni.a_shtrih({s: R[s]["tik"] for s in sids}, urovni.setka()["umolchaniya"], R[sids[0]]["dates"]) - AS.PLATA * baza
        assert np.allclose(x, ref, atol=1e-12), f"{imya}: excess категории ≠ a_shtrih"
        rows.append(L.stroka(imya, ",".join(sids), agg("I")[:-1], agg("N")[:-1], x, agg("eks") > 0, god))
    return rows


def syroi_close(idx) -> pd.DataFrame:
    import golaya_s7_s9 as G
    src = Path(d18.default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m"
    cl = {}
    for t in TICKERS:
        b = d18.read_bars(src / f"{t}.parquet", G.END)
        cl[t] = d18.aggregate(b[b["timestamp"] >= G.START], t).set_index("date")["close"]
    return pd.DataFrame(cl)[TICKERS].reindex(idx)


def kross() -> dict:
    import golaya_s7_s9 as G
    D = G.zagruzit()
    dates = pd.DatetimeIndex(D["c"].index)
    metki = ks.sektor_na_d(G.karta(), dates, TICKERS)
    sig, sk, stavka = G.sigmy(D["c"]).to_numpy(), G.skory(D, metki), msp.rate_on(dates).to_numpy()
    i0, win = int(np.searchsorted(dates.values, np.datetime64(G.T0))) - 1, np.asarray((dates >= G.T0) & (dates <= "2019-12-31"))
    C, out = syroi_close(dates).to_numpy(), {}
    for sid, s in sk.items():
        tau, n = ks.TAU[sid]
        ok = D["o"].notna().shift(-ks.LAG, fill_value=False).to_numpy()
        W, P, Dv = ks.vesa(s.to_numpy(), sig, metki, ok), D["o"].ffill().to_numpy(), D["d"].to_numpy()
        o = ks.kniga(W, P, Dv, dates.values, stavka, tau, n, i0)
        I, N = L.kniga_razlozhenie(W, P, C, Dv, tau, n, i0, ks.LAG)
        ecc = float(np.abs(I + N - o["pnl"]).max())
        assert ecc < 1e-12, f"{sid}: части ≠ pnl книги ({ecc})"
        out[sid] = {k: v[win] for k, v in dict(I=I, N=N, x=o["excess"], d=o["gross"] > 0).items()}
        print(sid, "кросс-секция готово; макс. |части − pnl| =", f"{ecc:.1e}", "; дней×тикеров без close:", int(np.isnan(C[win]).sum()), flush=True)
    return out


def stroki_kross(K: dict, god: float = 4.0) -> list:
    rows = [L.stroka(f"{s}", s, k["I"], k["N"], k["x"], k["d"], god) for s, k in K.items()]
    m = lambda key: np.mean([K[s][key] for s in K], axis=0)                       # noqa: E731
    rows.append(L.stroka(KATEGORII[5][0], "S7,S8,S9", m("I"), m("N"), m("x"), np.any([K[s]["d"] for s in K], axis=0), god))
    return rows


def main() -> None:
    R = {AS.SIDS[0]: schitat_sk(AS.SIDS[0])}              # первый — один: кэш 5-мин баров пишется общим файлом (гонка tmp → parquet)
    with Pool(2) as p:
        R.update(zip(AS.SIDS[1:], p.map(schitat_sk, AS.SIDS[1:])))
    rows = stroki_a(R) + stroki_kross(kross())
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
