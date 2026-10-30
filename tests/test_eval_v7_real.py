"""E1 куба v7, G1 на реальных точках train4: две точки из ledger train4 v6 (только чтение) — `EvaluatorV7` без осей v7 даёт строку ledger,
excess портфеля, excess тикеров и суммы port (ledger_v2.summy) побайтно как `EvaluatorV6`. Тяжёлый: FASTBT_V6_REAL=1 + дневной ряд 18:50."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fastbt import canon_v5, canon_v6, ledger_v2
from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7 import EvaluatorV7

real = pytest.mark.skipif(os.environ.get("FASTBT_V6_REAL") != "1" or not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(),
                          reason="тяжёлый реальный тест (FASTBT_V6_REAL=1, дневной ряд 18:50, ledger train4 v6 — только чтение)")
LEDGER = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/server_2021/train4_v6_31-10/ledger/train4_v6_s1")


def _znacheniya(x):
    """Значения осей из кубов: setka и v6_dobavleno на любой глубине."""
    if isinstance(x, dict):
        for k in ("setka", "v6_dobavleno"):
            yield from (x[k] if isinstance(x.get(k), list) else [])
        for v in x.values():
            yield from _znacheniya(v)


def _pool(sid):
    """{ось: {ledger_v2.tok(v): v}} — тип значения восстанавливается по кубам v5/v6 (в ledger всё строки)."""
    k5, k6, pool = canon_v5.load_kub(sid), canon_v6._kub(sid), {}
    for n, vs in canon_v5.osi(k5):
        pool.setdefault(n, {}).update({ledger_v2.tok(v): v for v in vs})
    for kub in (k5, k6):
        for d in kub.values():
            for n, spec in (d.items() if isinstance(d, dict) else ()):
                pool.setdefault(n, {}).update({ledger_v2.tok(v): v for v in _znacheniya(spec)})
                if isinstance(spec, (str, int, float)):
                    pool[n][ledger_v2.tok(spec)] = spec
    return pool


def tochki_ledger(sid, k=2):
    """Первые k точек p00 с ≥ 2 сделками: значения восстановлены по кубу, klyuch пересобран и равен записанному."""
    pool, out = _pool(sid), []
    for line in (LEDGER / sid / "p00" / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        kl = {n: v for n, v in r["klyuch"].items() if n not in ("period", "komissiya", "fdr")} if r["status"] == "ok" else {}
        if sum(r.get("sdelok_y", {}).values()) < 2 or any(n not in pool or v not in pool[n] for n, v in kl.items()):
            continue
        pt = {n: pool[n][v] for n, v in kl.items()}
        if ledger_v2.klyuch(sid, pt, "train4", 0.0004, "t0d") == r["klyuch"]:
            out.append((r, pt))
        if len(out) == k:
            break
    return out


@real
def test_g1_realnye_tochki_ledger_v7_ravno_v6_pobaytno():
    from fastbt import etap1
    sid = "S1"
    inp, base = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
    e6, e7 = EvaluatorV6(sid, inp, base), EvaluatorV7(sid, inp, base)
    tk = tochki_ledger(sid)
    assert len(tk) == 2, "не удалось восстановить 2 точки ledger (тип значений или klyuch)"
    daty = inp.meta[inp.sk.tickers[0]]["dates"]
    for r, pt in tk:
        r6, x6, t6 = e6.evaluate_v2(pt, "train4")
        r7, x7, t7 = e7.evaluate_v2(pt, "train4")
        r6.pop("t_sek", None), r7.pop("t_sek", None)
        b = lambda x: None if x is None else x.tobytes()      # noqa: E731
        assert x7 is not None and np.abs(x7).sum() > 0, "точка без сделок — тест ничего не доказывает"
        assert repr(r6) == repr(r7) and b(x6) == b(x7) and b(t6) == b(t7)
        assert ledger_v2.summy(daty, x7).tobytes() == ledger_v2.summy(daty, x6).tobytes()      # суммы port
