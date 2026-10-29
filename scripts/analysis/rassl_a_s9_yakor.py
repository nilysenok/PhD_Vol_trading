"""Расследование: якорь A на test1 — «A (s9)» (scripts/revers/r1_a_cd.py) против «yakor» (scripts/revers/r1_a.py). Поденные ряды
обоих по (Sk, тикер, день): gross, комиссия, итог; первый день расхождения; разложение по тикерам. Только чтение; код fastbt не меняется.
Запуск: FASTBT_Z2_DAILY=… FASTBT_Z2_DIV=… FASTBT_Z2_R2FINAL=… FASTBT_BARS1M=… python3 scripts/analysis/rassl_a_s9_yakor.py \
    --period {train4,test1} --out DIR
Пишет DIR/s9_<period>.pkl и DIR/yakor_<period>.pkl (матрицы дата×тикер по Sk) и печатает сравнение. 2 процесса не нужны — один."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
from fastbt import a_strogaya as AS, etap1, s9  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402

COMM = AS.COMM


def pair_matrices(eng, comm: float) -> dict:
    """s9: {sid: {'gross','comm','net'} → DataFrame[дата × тикер]} — вклад пары в сумму по барам (до деления на число пар)."""
    pos, nd, nt = eng.A_row, len(eng.dates), len(eng.tickers)
    dp = np.diff(pos, prepend=0.0)
    dp[eng.first] = pos[eng.first]
    last = np.r_[eng.first[1:], True]
    sides = np.abs(dp) + np.where(last, np.abs(pos), 0.0)
    gross = pos * eng.lr_row
    out = {}
    for si, sid in enumerate(s9.SIDS):
        m = np.array([eng.pairs[p][1] == sid for p in eng.r["pair"]])
        mat = {}
        for nm, v in (("gross", gross), ("comm", sides * comm)):
            mat[nm] = pd.DataFrame(np.bincount(eng.r["tick"][m] * nd + eng.r["date"][m], v[m], nt * nd).reshape(nt, nd).T,
                                   index=eng.dates, columns=eng.tickers)
        mat["net"] = mat["gross"] - mat["comm"]
        out[sid] = mat
    return out


def s9_side(period: str) -> dict:
    s9.set_period(period)
    import r1_a_cd as CD
    eng = s9.Engine(CD.kesh(period))
    return dict(pairs=pair_matrices(eng, COMM), cnt=pd.Series(eng.cnt, index=eng.dates),
                port=eng.port(eng.A_row, comm=COMM), tickers=eng.tickers)


def yakor_side(period: str) -> dict:
    out, cnt, port = {}, None, None
    for sid in AS.SIDS:
        inp = etap1.build_inputs(sid, period=period)
        ev = EvaluatorV2(sid, inp, etap1.base_point(sid), COMM)
        row, x, tik = ev.evaluate_v2({}, period)
        df, inv, eng, _, rs = ev._last
        nd = len(eng.dates)
        g = np.zeros((len(rs), nd))
        sd = np.zeros((len(rs), nd))
        lo = 0
        for ti, r in enumerate(rs):
            sl = slice(lo, lo + len(r.g))
            lo = sl.stop
            g[ti] = np.bincount(inv[sl], r.g, nd)
            sd[ti] = np.bincount(inv[sl], r.sides, nd) * COMM
        tk = list(ev.sk.tickers)             # порядок = порядок rs (см. EvaluatorV2: enumerate(self.sk.tickers))
        assert len(tk) == len(rs), (len(tk), len(rs))
        gross = pd.DataFrame(g.T, index=eng.dates, columns=tk)
        cm = pd.DataFrame(sd.T, index=eng.dates, columns=tk)
        out[sid] = dict(gross=gross, comm=cm, net=gross - cm, port=df, cnt_pairs=pd.Series(eng.cnt, index=eng.dates),
                        div=pd.DataFrame({t: pd.Series(inp.div[t], index=pd.DatetimeIndex(inp.meta[t]["dates"])) for t in tk}))
        print(sid, "готово: дней", nd, "тикеров", len(tk), flush=True)
    return out


def sravnit(a: dict, b: dict) -> None:
    """a — s9, b — yakor: по Sk первый день, где net по (дата, тикер) расходится > 1e-12; числа дней и тикеров."""
    for sid in AS.SIDS:
        na, nb = a["pairs"][sid]["net"], b[sid]["net"]
        print(f"{sid}: s9 дней {len(na)} ({na.index.min().date()}…{na.index.max().date()}), тикеров {na.shape[1]}; "
              f"yakor дней {len(nb)} ({nb.index.min().date()}…{nb.index.max().date()}), тикеров {nb.shape[1]}")
        only_a, only_b = sorted(set(na.columns) - set(nb.columns)), sorted(set(nb.columns) - set(na.columns))
        if only_a or only_b:
            print("   тикеры только в s9:", only_a, "только в yakor:", only_b)
        di = na.index.intersection(nb.index)
        print("   дат только в s9:", [str(d.date()) for d in na.index.difference(nb.index)][:5],
              "только в yakor:", [str(d.date()) for d in nb.index.difference(na.index)][:5])
        cols = [c for c in na.columns if c in nb.columns]
        d = (na.loc[di, cols] - nb.loc[di, cols]).fillna(0.0)
        bad = d.abs().max(axis=1) > 1e-12
        print(f"   дней с расхождением: {int(bad.sum())} из {len(di)}; первый: {d.index[bad][0].date() if bad.any() else '-'}")
        if bad.any():
            per_t = d.abs().sum().sort_values(ascending=False).head(5)
            print("   топ тикеров по Σ|Δ|:", {k: round(float(v), 5) for k, v in per_t.items()})


def pervyj(d: pd.Series, eps: float = 1e-12) -> str:
    bad = d.abs() > eps
    return f"дней {int(bad.sum())}, первый {d.index[bad][0].date() if bad.any() else '-'}, Σ|Δ|={float(d.abs().sum()):.6g}"


def komponenty(a: dict, b: dict) -> None:
    """Портфель A (среднее по 6 Sk) по компонентам: gross, комиссия (gross−r), ставка/плечо (excess−r), итог excess — s9 минус yakor.
    Ставка: s9 `_finish` и yakor `port_rows` — один код; расхождение в `excess−r` значит разные доли занятости (exposure) или rf·cal."""
    pa, pb = a["port"], pd.concat({s: b[s]["port"] for s in AS.SIDS}).groupby(level=1).mean()
    cb = pd.concat({s: b[s]["cnt_pairs"] for s in AS.SIDS}).groupby(level=1).mean()
    print(f"A портфель: s9 дней {len(pa)}, yakor (среднее по Sk) дней {len(pb)}; среднее cnt s9 {a['cnt'].mean():.2f} yakor {cb.mean():.2f}")
    di = pa.index.intersection(pb.index)
    for nm, fa, fb in (("gross", pa["gross"], pb["gross"]), ("комиссия gross−r", pa["gross"] - pa["r"], pb["gross"] - pb["r"]),
                       ("ставка/плечо excess−r", pa["excess"] - pa["r"], pb["excess"] - pb["r"]),
                       ("exposure", pa["exposure"], pb["exposure"]), ("excess", pa["excess"], pb["excess"])):
        print(f"   {nm}: {pervyj((fa.loc[di] - fb.loc[di]).fillna(0.0))}")
    from fastbt import dividendy, zona
    zona.set_zone("test1")
    dd = []
    for t in a["tickers"]:
        ya = b[AS.SIDS[0]]["div"].get(t)
        if ya is None:
            continue
        sa = pd.Series(dividendy.gap_divs(t, pd.DatetimeIndex(a["port"].index).values), index=a["port"].index)
        dd.append((t, float((sa.loc[sa.index.intersection(ya.index)] - ya.loc[sa.index.intersection(ya.index)]).abs().sum())))
    print("   дивиденды (gap_divs на датах s9 против yakor), Σ|Δ| по тикерам:", [x for x in dd if x[1] > 0] or "совпадают")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=("train4", "test1"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pa, pb = a.out / f"s9_{a.period}.pkl", a.out / f"yakor_{a.period}.pkl"
    if not pa.exists():
        pa.write_bytes(pickle.dumps(s9_side(a.period)))
    if not pb.exists():
        pb.write_bytes(pickle.dumps(yakor_side(a.period)))
    sa, sb = pickle.loads(pa.read_bytes()), pickle.loads(pb.read_bytes())
    sravnit(sa, sb)
    komponenty(sa, sb)


if __name__ == "__main__":
    main()
