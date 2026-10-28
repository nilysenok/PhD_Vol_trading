"""Ж-043: «было» (A только Sk) считается полным движком, как S9Baza, а не окном по end. Причина расхождения с bazy_sverka на 0,001–0,004:
окно по end платит выход на последнем дне окна, полный прогон — нет; разница целиком в одном дне (последний день года).
Нужны данные s9 (кэш v2); без них тест пропускается."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [".", "scripts"]
from fastbt import s9  # noqa: E402

if not (s9.OUT / "cache_latest_v2.txt").exists():
    pytest.skip("нет кэша s9 v2", allow_module_level=True)

import zh043_lib as L  # noqa: E402
from fastbt import ekviv_s9 as ek  # noqa: E402
from fastbt import shag0  # noqa: E402

BAZY = pd.read_csv("docs/tables/sweep_e1v2/bazy_sverka.csv").set_index("baza")


def _okno(sid: str, god: int):
    ys = ek.WINDOWS[god]
    eng = ek.s9_engine(ek.a_cache(2019), ys)
    pos = shag0.keep_sids(np.asarray(eng.A_row, float), shag0.sid_row(eng), [sid])
    return eng.port(pos, comm=0.0004)["excess"]


def test_byli_sovpadaet_s_bazy_sverka():
    for sid in ("S1", "S3"):
        r = {x["god"]: x["sharpe_posle_004"] for x in L.main_rows(sid)}
        assert abs(r[2018] - BAZY.loc[sid, "2018"]) < 1e-4 and abs(r[2019] - BAZY.loc[sid, "2019"]) < 1e-4


def test_mutatsiya_okno_po_end_daet_drugoe_chislo_v_odin_den():
    x = _okno("S1", 2018)
    x = x[x.index.year == 2018]
    assert abs(s9.sharpe(x) - BAZY.loc["S1", "2018"]) > 1e-3         # окно по end — −0,4919 против −0,4942: тест выше его ловит
    full = L.main_rows("S1")[0]["sharpe_posle_004"]
    assert abs(full - BAZY.loc["S1", "2018"]) < 1e-4
