"""Скорость этапа 1 (Ж-083): эквивалентность быстрого слоя (`fastbt.fast_etap1.patched()`) и медленного пути
(`etap1_eval.Evaluator` как есть — эталон) на train4, ПОБАЙТНО.

200 случайных точек × 6 стратегий (S1…S6) из сеток `fastbt/designs/sweep/zh047_S{k}.yaml` плюс ось `vyhod`
(net | lim, в yaml её нет — Ж-065 п. 3). Точки сделаны так, чтобы в каждой стратегии были: k_s ≠ net, R ≠ net,
трейлинг (k_tr ≠ net), vyhod = lim (с вход ≠ рынок и bars5m — иначе ось не действует), vhod ≠ rynok.
Сравнивается всё: словарь `evaluate` (excess_d, by_year, sdelok_y, …), excess портфеля (df), лента сделок.
Печать: число расхождений и первая расходящаяся точка. Нет `fastbt.fast_etap1` — падает с текстом «нет fast_etap1».

python3 scripts/analysis/fastbt_equiv_etap1.py [--n 200] [--sids S1,S2,...] [--seed 20261028]
Нужны 1-минутки D25 (FASTBT_BARS1M). Данные ≥ 2022 не читаются (окно train4 = 2016–2019).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PERIOD = "train4"
SIDS = ["S1", "S2", "S3", "S4", "S5", "S6"]
SKIP_SOBYTIYA = ("div", "cbr_div")              # «не считано» (❓5): точка вырожденная, сравнивать нечего


def load_fast():
    """(fast_etap1, patched) или SystemExit с понятным текстом."""
    try:
        from fastbt import fast_etap1
    except ImportError as e:
        raise SystemExit(f"нет fast_etap1 ({e}): ветка skorost-etap1 не слита; сверять нечего") from e
    return fast_etap1, fast_etap1.patched


def sample_points(sid: str, n: int, seed: int) -> list:
    """n точек: равномерно по осям zh047_{sid}.yaml + vyhod; первые 6 — обязательные классы (см. docstring)."""
    from fastbt.etap1_poisk import load_setka
    axes = {k: list(v) for k, v in load_setka(sid)["axes"].items()}
    axes["sobytiya"] = [v for v in axes["sobytiya"] if v not in SKIP_SOBYTIYA]
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        pt = {k: v[int(rng.integers(len(v)))] for k, v in axes.items()}
        pt["vyhod"] = "lim" if rng.random() < 0.5 else "net"
        if pt["vyhod"] == "lim":                    # vyhod_lim действует только при k и check = bary
            pt["vhod"] = [v for v in axes["vhod"] if v != "rynok"][int(rng.integers(len(axes["vhod"]) - 1))]
            pt["proverka"] = "bars5m"
        if i < 6:
            pt.update(_FORCED[i])
            if pt["vyhod"] == "lim" and (pt["vhod"] == "rynok" or pt["proverka"] != "bars5m"):
                pt["vyhod"] = "net"                 # vyhod_lim: нужны лимитный вход и check = bary (exec_e.Params)
        pts.append(pt)
    return pts


_FORCED = [dict(k_s=1.5, R=2.0, k_tr="net", vhod="rynok", proverka="bars5m", tau="net"),
           dict(k_s=2.0, R="net", k_tr=2.0, vhod=0.5, proverka="bars5m", tau=5),
           dict(k_s=1.0, R=1.0, k_tr="net", vhod=0.25, proverka="bars5m", tau=3, vyhod="net"),
           dict(k_s=3.0, R=3.0, k_tr=1.5, vhod=0.75, proverka="bars5m", tau="net", vyhod="lim"),
           dict(k_s="net", R="net", k_tr="net", vhod=1.0, proverka="zakrytie", tau=2, vyhod="net"),
           dict(k_s=2.5, R=1.5, k_tr=3.0, vhod=0.5, proverka="bars5m", tau=10, vyhod="lim")]


def canon(x) -> bytes:
    """Канонические байты значения: float — IEEE-754 как есть (NaN сравнивается по битам), без допусков."""
    if isinstance(x, (bool, np.bool_)):
        return b"b1" if x else b"b0"
    if isinstance(x, (float, np.floating)):
        return b"f" + struct.pack("<d", float(x))
    if isinstance(x, (int, np.integer)):
        return b"i" + str(int(x)).encode()
    if isinstance(x, dict):
        return b"{" + b",".join(str(k).encode() + b":" + canon(x[k]) for k in sorted(x, key=str)) + b"}"
    if isinstance(x, (list, tuple)):
        return b"[" + b",".join(canon(v) for v in x) + b"]"
    if isinstance(x, np.ndarray):
        return b"a" + str(x.dtype).encode() + str(x.shape).encode() + np.ascontiguousarray(x).tobytes()
    return b"s" + repr(x).encode()


def trades_of(rs) -> list:
    return [(tr.ticker, tr.d_sig, tr.entry_day, tr.entry_bar, tr.entry_px, tr.exit_day, tr.exit_bar, tr.exit_px,
             tr.reason, tr.side) for r in rs for tr in r.trades]


def run_point(ev, pt: dict, period: str = PERIOD) -> dict:
    """Всё, что обязано совпасть побайтно: строка evaluate, excess портфеля, стороны, лента сделок."""
    out = ev.evaluate(pt, period)
    df, _, _, cnt, rs = ev.excess(pt, period)
    return dict(out=out, excess=df["excess"].to_numpy(), index=df.index.to_numpy().astype("int64"), cnt=cnt,
                trades=trades_of(rs))


def diff_keys(a: dict, b: dict) -> list:
    """Какие части результата расходятся: ключи evaluate с префиксом out. и excess/index/cnt/trades."""
    bad = [f"out.{k}" for k in sorted(set(a["out"]) | set(b["out"]), key=str)
           if canon(a["out"].get(k)) != canon(b["out"].get(k))]
    return bad + [k for k in ("excess", "index", "cnt", "trades") if canon(a[k]) != canon(b[k])]


def compare(slow: list, fast: list, pts: list, sid: str) -> list:
    """[(sid, номер, точка, ключи)] для расходящихся точек."""
    return [(sid, i, p, d) for i, (s, f, p) in enumerate(zip(slow, fast, pts)) if (d := diff_keys(s, f))]


def wide(inp):
    """Те же входы, но каждый 5-мин бар раздут до ±20 % от open: стоп, тейк и лимит попадают в один бар
    (на реальных барах при R ≥ 1 стоп и тейк в одном баре не встречаются — порядок событий иначе не виден)."""
    import dataclasses
    from fastbt.exec_e_types import Day
    mk = lambda d: d if d.pusto else Day(d.o, np.maximum(d.h, d.o * 1.2), np.minimum(d.l, d.o * 0.8), d.c, d.pusto)  # noqa: E731
    return dataclasses.replace(inp, bars={t: [mk(d) for d in ds] for t, ds in inp.bars.items()})


def perturbed(inp):
    """Подмена входов (Д5): σ ×1,5 и бары ×(1 + 0,001·(i mod 7)); результат медленного пути обязан поменяться."""
    import dataclasses
    from fastbt.exec_e_types import Day
    f = lambda i: 1.0 + 0.001 * (i % 7)                                                         # noqa: E731
    bars = {t: [d if d.pusto else Day(d.o * f(i), d.h * f(i), d.l * f(i), d.c * f(i), d.pusto)
                for i, d in enumerate(ds)] for t, ds in inp.bars.items()}
    sig = {n: {t: v * 1.5 for t, v in tv.items()} for n, tv in inp.sigma.items()}
    return dataclasses.replace(inp, bars=bars, sigma=sig, sk=perturbed_sk(inp.sk))


def perturbed_sk(sk):
    """Подмена рядов Sk (Д5): цены ×(1 + 0,08·sin(i/9)) → другие сигналы и позиции. Кэш позиций быстрого слоя зависит
    ТОЛЬКО от рядов Sk, кода и параметров (не от баров и σ оценщика), поэтому проверять его ключ можно лишь так."""
    import copy
    import dataclasses
    from fastbt.indicators import compute_base
    sk2, td2 = copy.copy(sk), {}
    for t, (td, _) in sk._td.items():
        f = 1.0 + 0.08 * np.sin(np.arange(td.n) / 9.0)
        o, h, l, c = (np.ascontiguousarray(x * f) for x in (td.open, td.high, td.low, td.close))
        td2[t] = (dataclasses.replace(td, open=o, high=h, low=l, close=c), compute_base(c, h, l, o, td.volume))
    sk2._td = td2
    return sk2


def evaluator(sid: str):
    from fastbt import etap1
    return etap1.Evaluator(sid, etap1.build_inputs(sid, period=PERIOD), etap1.base_point(sid))


def run_sid(sid: str, n: int, seed: int, patched) -> tuple:
    """(число точек, расхождения) для стратегии; медленный и быстрый — на раздельных Evaluator (свои кэши)."""
    pts = sample_points(sid, n, seed)
    ev_s = evaluator(sid)
    slow = [run_point(ev_s, p) for p in pts]
    ev_f = evaluator(sid)
    with patched():
        fast = [run_point(ev_f, p) for p in pts]
    return len(pts), compare(slow, fast, pts, sid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sids", default=",".join(SIDS))
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(argv)
    _, patched = load_fast()
    total, bad = 0, []
    for k, sid in enumerate(a.sids.split(",")):
        n, b = run_sid(sid, a.n, a.seed + k, patched)
        total += n
        bad += b
        print(f"{sid}: точек {n}, расхождений {len(b)}", flush=True)
    print(f"{PERIOD}: точек {total}, расхождений {len(bad)}")
    if bad:
        sid, i, pt, keys = bad[0]
        print(f"первая расходящаяся: {sid} #{i} {pt}\n  ключи: {keys}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
