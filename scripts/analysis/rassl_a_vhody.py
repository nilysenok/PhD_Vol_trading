"""Расследование A s9 vs yakor, шаг 2: ВХОДЫ на test1 по тикеру — цены и позиции одной стратегии Sk (по умолчанию S3).
Печатает: (1) max|Δ close| между дневным рядом зоны Z2 (daily_df — его видит s9 и сигналы yakor), `inp.close` yakor и закрытием
последней 5-минутки дня из `inp.bars` (то, по чему yakor исполняет); первые 5 дней расхождения; (2) позиции: s9 (кэш A[sid][Y])
против сигнала yakor (`_positions`, до исполнения) и против исполненной экспозиции yakor (`r.expo`): число расходящихся дней,
первые 10 дней таблицей. Только чтение.
Запуск: те же FASTBT_* , что у rassl_a_s9_yakor.py:
  python3 scripts/analysis/rassl_a_vhody.py --sid S3 --tickers AFLT SBER"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
from fastbt import a_strogaya as AS, data, etap1, s9  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402


def pos_s9(eng, t: str, sid: str) -> pd.Series:
    """Позиция s9 по датам годов периода: срезы A[sid][Y] (как Engine.rows_from)."""
    c, parts = eng.cache[t], []
    for Y in s9.YEARS:
        ys, ye = eng._slice(t, Y)
        parts.append(pd.Series(np.asarray(c["A"][sid][Y][ys:ye], float), index=pd.DatetimeIndex(c["dates"][ys:ye])))
    return pd.concat(parts)


def ceny(inp, t: str, n: int = 5) -> None:
    d = data.daily_df()
    d = d[d["ticker"] == t].set_index("date")["close"]
    m = inp.meta[t]
    idx = pd.DatetimeIndex(m["dates"])
    rc = pd.Series(inp.close_n6[t], index=idx)
    bc = pd.Series([np.nan if (b.pusto or len(b.c) == 0) else b.c[-1] for b in inp.bars[t]], index=idx[:len(inp.bars[t])])
    print(f"[{t}] дней в окне yakor {len(idx)} ({idx[0].date()}…{idx[-1].date()}); бары без данных (pusto): {int(bc.isna().sum())}")
    for nm, s in (("inp.close_n6 − daily", rc - d.reindex(idx)), ("посл. 5-мин. close − daily", bc - d.reindex(bc.index))):
        s = s.dropna()
        bad = s.abs() > 1e-9
        print(f"   {nm}: max|Δ|={float(s.abs().max()):.6g}, дней с Δ {int(bad.sum())} из {len(s)}; первые: "
              f"{[(str(k.date()), round(float(v), 4)) for k, v in s[bad].head(n).items()]}")


def pozicii(ev, rs, eng_y, inv, t: str, ps9: pd.Series, n: int = 10) -> None:
    ti = list(ev.sk.tickers).index(t)
    m = ev.inp.meta[t]
    s0, b = m["s0"], ev._b(t, "test1")
    sp = {k: ev.base[k] for k in ev.sk.sig_names}
    sig = pd.Series(ev._positions(t, sp, 0.0, True, s0 + b)[s0:s0 + b + 1].astype(float), index=pd.DatetimeIndex(m["dates"][:b + 1]))
    r = rs[ti]
    ex = pd.Series(r.expo, index=eng_y.dates[inv[sum(len(x.g) for x in rs[:ti]):][:len(r.g)]])
    di = ps9.index.intersection(sig.index)
    t_ = pd.DataFrame({"s9_pos": ps9.loc[di], "yakor_signal": sig.loc[di], "yakor_expo": ex.reindex(di)})
    print(f"[{t}] позиции {ev.sid}: s9 дней {len(ps9)}, общих с yakor {len(di)}; "
          f"s9≠сигнал yakor в {int((t_.s9_pos != t_.yakor_signal).sum())} днях; s9≠|expo| yakor в {int((t_.s9_pos.abs() != t_.yakor_expo.abs()).sum())}")
    print(t_.head(n).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="S3", choices=AS.SIDS)
    ap.add_argument("--tickers", nargs="+", default=["AFLT", "SBER"])
    a = ap.parse_args()
    s9.set_period("test1")
    import r1_a_cd as CD
    eng = s9.Engine(CD.kesh("test1"))
    inp = etap1.build_inputs(a.sid, period="test1")
    ev = EvaluatorV2(a.sid, inp, etap1.base_point(a.sid), AS.COMM)
    ev.evaluate_v2({}, "test1")
    _, inv, eng_y, _, rs = ev._last
    for t in a.tickers:
        ceny(inp, t)
        pozicii(ev, rs, eng_y, inv, t, pos_s9(eng, t, a.sid))


if __name__ == "__main__":
    main()
