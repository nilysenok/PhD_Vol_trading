"""Расследование A s9 vs yakor, шаг 3: ОТКУДА параметры сигнала. s9 A = ApproachA (fastbt/approach_a.py): на год Y из сетки сигнал×RM×исполнение
берётся top-10 по Шарпу на окне до 1 января Y (walk-forward), позиция = голосование ансамбля. yakor A = ОДНА фиксированная точка
`etap1.base_point(sid)` (ekviv_20.yaml#S{k}_a), позиции `EvaluatorV2._positions`. Скрипт для тикера/дня печатает: параметры yakor;
top-10 s9 на год дня (параметры сигнала u.ss.params[i], RM u.rm[r], исполнение); сигнал S3 каждой из них и yakor на день; Donchian
(close, max high / min low за окно); и число дней, где позиция s9 (кэш) ≠ позиция yakor (сигнал), по периоду.
Запуск (те же FASTBT_*, что у rassl_a_s9_yakor.py), по одному процессу на период:
  python3 scripts/analysis/rassl_a_parametry.py --period test1 --sid S3 --ticker AFLT --day 2022-01-04
  python3 scripts/analysis/rassl_a_parametry.py --period train4 --sid S3 --ticker AFLT --day 2016-01-12"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from fastbt import a_strogaya as AS, dev, etap1, s9  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402
import rassl_a_vhody as V  # noqa: E402


def s9_top10(ticker: str, sid: str, Y: int):
    """Тот же путь, что s9._job для одной пары: top-10 ApproachA.year(Y) и юнит."""
    from fastbt import design as dsg
    from fastbt.approach_a import ApproachA
    from fastbt.data import load_ticker
    from fastbt.unit import make_unit
    d, _ = dev.load_design(s9.MAIN_YAML)
    cfg = s9.period_cfg(dev.make_config(d, s9.SCOPE))
    dsg.apply(cfg.design)
    u = make_unit(ticker, sid, load_ticker(ticker, cfg.corp_actions))
    a = ApproachA(u, cfg)
    return a, u, a.year(Y), cfg


def donchian(td, idx: int, w: int) -> str:
    lo, hi = max(idx - w, 0), idx
    return (f"w={w}: close={td.close[idx]:.4f}, max high[{lo}:{hi}]={td.high[lo:hi].max():.4f}, "
            f"min low[{lo}:{hi}]={td.low[lo:hi].min():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=("train4", "test1"), required=True)
    ap.add_argument("--sid", default="S3", choices=AS.SIDS)
    ap.add_argument("--ticker", default="AFLT")
    ap.add_argument("--day", required=True)
    a = ap.parse_args()
    s9.set_period(a.period)
    day, t = pd.Timestamp(a.day), a.ticker
    base = etap1.base_point(a.sid)
    print(f"[yakor] точка {a.sid}_a (фиксированная, одна на все годы и тикеры): {base}")
    aa, u, top, cfg = s9_top10(t, a.sid, day.year)
    print(f"[s9] год {day.year}: граница обучения = индекс {int(np.searchsorted(u.dates, np.datetime64(f'{day.year}-01-01')))} "
          f"(1 января {day.year}); прогрев WARMUP; top-10 (i_sig, i_rm, i_exec) = {top}")
    idx = int(np.searchsorted(u.dates, np.datetime64(day)))
    for (i, r, e) in (top or []):
        print(f"   i={i} сигнал {u.ss.params[i]} | RM {u.rm[r]} | исполнение {aa.exec_grid[e]} | sig[{day.date()}]={int(u.sigs[i][idx])}")
    ev_sp = {k: base[k] for k in ("dc_window",) if k in base}
    iy = [i for i, p in enumerate(u.ss.params) if all(p.get(k) == v for k, v in ev_sp.items())]
    print(f"[yakor] индекс сигнала по сетке s9 {iy}; sig на день: {[int(u.sigs[i][idx]) for i in iy]}")
    for w in sorted({int(u.ss.params[i].get('dc_window', 0)) for i, _, _ in (top or [])} | {int(base.get('dc_window', 0))} - {0}):
        print("   Donchian", donchian(u.td, idx, w))
    inp = etap1.build_inputs(a.sid, period=a.period)
    ev = EvaluatorV2(a.sid, inp, base, AS.COMM)
    m, b = inp.meta[t], ev._b(t, a.period)
    sp = {k: base[k] for k in ev.sk.sig_names}
    sig = pd.Series(ev._positions(t, sp, 0.0, True, m["s0"] + b)[m["s0"]:m["s0"] + b + 1].astype(float),
                    index=pd.DatetimeIndex(m["dates"][:b + 1]))
    eng = s9.Engine(pickle.loads((s9.OUT / (s9.OUT / s9._latest_name()).read_text()).read_bytes()))
    ps = V.pos_s9(eng, t, a.sid)
    di = ps.index.intersection(sig.index)
    diff = ps.loc[di] != sig.loc[di]
    print(f"[{a.period}] {t} {a.sid}: общих дней {len(di)}, позиция s9 ≠ сигнал yakor в {int(diff.sum())}; первые: "
          f"{[(str(k.date()), float(ps[k]), float(sig[k])) for k in di[diff.to_numpy()][:5]]}")


if __name__ == "__main__":
    main()
