"""Л2.3, G1 на реальных точках: `EvaluatorV7` с otnositelno=net даёт строку ledger, excess портфеля и тикеров побайтно как `EvaluatorV6`.
Тяжёлый (FASTBT_V6_REAL=1, дневной ряд 18:50); период train 2018–2019 (прогрев 2017), данные ≤ 2019; по одной точке куба v5 на S1–S4."""
import os

import numpy as np
import pytest

from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7 import EvaluatorV7


@pytest.mark.skipif(os.environ.get("FASTBT_V6_REAL") != "1", reason="тяжёлый реальный тест (FASTBT_V6_REAL=1, дневной ряд 18:50)")
@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4"])
def test_g1_realnaya_tochka_net_ravno_v6_pobajtno(sid):
    from fastbt import canon_v5, etap1
    inp, base = etap1.build_inputs(sid, period="train"), etap1.base_point(sid)
    e6, e7, osi = EvaluatorV6(sid, inp, base), EvaluatorV7(sid, inp, base), canon_v5.osi(canon_v5.load_kub(sid))
    otkazy = []
    for k in range(300):                                      # детерминированный обход куба v5 до первой годной точки со сделками
        rng = np.random.default_rng(k)
        pt = {n: vs[rng.integers(len(vs))] for n, vs in osi}
        if canon_v5.canon_v5(sid, pt) is None:
            continue
        try:
            r6, x6, t6 = e6.evaluate_v2(pt, "train")
        except ValueError as ex:
            otkazy.append(str(ex)[:120])
            continue
        if x6 is not None and np.abs(x6).sum() > 0:
            break
    else:
        pytest.fail(f"годная точка со сделками не найдена; отказы: {otkazy[:3]}")
    r7, x7, t7 = e7.evaluate_v2({**pt, "otnositelno": "net"}, "train")
    r6.pop("t_sek", None), r7.pop("t_sek", None)
    assert repr(r6) == repr(r7) and x6.tobytes() == x7.tobytes() and t6.tobytes() == t7.tobytes()
