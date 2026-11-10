"""Этап Z (Ж-246), боевые входы `x1_z_stage.compute`: e_d и s_d, которые стадия отдаёт `walk` для даты d, побитно равны независимому пересчёту (e — колонка `s` x1.parquet по ДАТЕ d; s_d — бары дня d склада M10 по `x1_ya_feat.day_features`).
Мутации на боевом пути ловятся: (а) e_d со строки t⁺ (np.roll(e, −1)); (б) s_d сдвинут на +1 строку (np.roll(s, −1))."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_feat as YF, x1_z_stage as S


class Stop(Exception):
    pass


def captured(inp: dict, stage=S) -> tuple:
    """-> (D, d): входы, которые `compute` передаёт первому вызову walk."""
    got = {}

    def spy(x1, D, kind):
        got["D"], got["d"] = D, pd.DatetimeIndex(x1["d"])
        raise Stop

    orig = stage.W.walk
    stage.W.walk = spy
    try:
        stage.compute(inp)
    except Stop:
        pass
    finally:
        stage.W.walk = orig
    return got["D"], got["d"]


def independent(d) -> tuple:
    """e_d по дате d из файла X1; s_d по барам дня d склада (не через stage)."""
    x1 = pd.read_parquet(S.SA.X1_FILE).set_index("d")
    st = YF.load_store()
    pos = st["days"].get_indexer(d)
    s = np.array([YF.day_features(st["r2"][p], st["present"][p])[0] for p in pos])
    return x1.loc[d, "s"].to_numpy(int), s


@pytest.fixture(scope="module")
def inp():
    return S.load()


def test_e_and_s_equal_independent_recount_by_date_d(inp) -> None:
    D, d = captured(inp)
    e, s = independent(d)
    assert np.array_equal(D["e"], e) and np.array_equal(D["s"], s) and D["e"].sum() == 335 and np.array_equal(D["Z"][:, 0], e.astype(float)) and np.array_equal(D["Z"][:, 2], s)


@pytest.mark.parametrize("which", ["e", "s"])
def test_roll_mutations_on_battle_path_are_caught(inp, monkeypatch, which) -> None:
    orig = S.SB.load

    def mutant():
        i = orig()
        if which == "e":
            i["x1"] = i["x1"].assign(s=np.roll(i["x1"]["s"].to_numpy(), -1))          # (а) e_d со строки t⁺
        else:
            i["ft"] = dict(i["ft"], s=np.roll(np.asarray(i["ft"]["s"]), -1))          # (б) s_d на +1 строку
        return i

    monkeypatch.setattr(S.SB, "load", mutant)
    D, d = captured(S.load())
    e, s = independent(d)
    assert not (np.array_equal(D["e"], e) and np.array_equal(D["s"], s))
