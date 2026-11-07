"""Р-1, находка 1: `select.run_loyo` берёт результаты из `Pool.imap_unordered` -> порядок строк не фиксирован.
Подмена `runner.map_jobs` на выдачу в обратном порядке (боевой `run_loyo`, реальный пул): сравнение списком score падает, по метке — проходит."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s4 import mkt, select
from models_v2.s4.fit import HP
from models_v2.s10 import hp_k23_world as KW, hp_synth as SS

SMALL = (HP(2, 30.0, 2.0, 0.1, 0.8, 0.5, 40), HP(3, 30.0, 2.0, 0.1, 0.8, 0.5, 40))


@pytest.fixture(scope="module")
def world():
    bars, imo, _ = SS.world(900, ("A", "B", "C"))
    return KW.build_world(bars["A"], imo, [mkt.paper(bars[t]) for t in ("A", "B", "C")])


def _scores(w, monkeypatch, reverse: bool) -> pd.DataFrame:
    monkeypatch.setattr(select, "GRID132", SMALL)
    orig = select.runner.map_jobs
    # порядок «а» фиксируем (по ключу задания), «б» — обратный: естественный порядок imap_unordered сам недетерминирован
    def ordered(*a, **k):
        r = sorted(orig(*a, **k), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2]))
        return r[::-1] if reverse else r
    monkeypatch.setattr(select.runner, "map_jobs", ordered)
    return KW.loyo_scores(w, (2018,), 2)


def test_poryadok_prihoda_obratnyj_spisok_padaet_a_metka_prohodit(world, monkeypatch):
    a = _scores(world, monkeypatch, False)
    b = _scores(world, monkeypatch, True)
    assert len(a) == len(b) == 2 and np.isfinite(a["score"]).all()
    assert a["label"].tolist() != b["label"].tolist()  # порядок строк действительно сменился (подмена не пустая)
    assert a["score"].tolist() != b["score"].tolist()  # старое сравнение списком упало бы
    sa, sb = a.set_index("label")["score"].to_dict(), b.set_index("label")["score"].to_dict()
    assert set(sa) == set(sb) and all(abs(sa[k] - sb[k]) <= 1e-12 for k in sa)  # новое — по метке
