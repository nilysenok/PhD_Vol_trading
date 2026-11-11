"""W4-B1, мутации признаков (W4_SPEC 9, W4-B): «признак дня t⁺», «бары после 18:40 дня d», «нормировка s7 по всей выборке», «пятидневное среднее с днём d+1» ловятся
боевыми проверками `check_pit` / `check_archive`. Подмены — только monkeypatch (класс № 43); реальные файлы только читаются."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from models_v2 import daily as D
from models_v2.s13 import v6_1_feat as FE
from models_v2.s13 import v6_1b_sig as SG
from models_v2.s16 import x1_w4_feat as F
from models_v2.s17.fwd_src import Src

ROOT = Path(F.__file__).resolve().parents[2]
REAL = [F.FEAT / "S_ext.parquet", F.FEAT / "SIGMA_ext.parquet", F.VOLCSV, ROOT / "docs/otchety/harj_x1/ALL_TRIALS.csv"]
DATES = ["2026-03-02", "2026-10-01"]


def real_state() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


@pytest.fixture(scope="module")
def src() -> Src:
    return Src()


def test_mutation_row_from_bars_of_day_t_plus_is_caught(src, monkeypatch) -> None:
    """Строка d берётся из баров дня t⁺ = следующего d (значения дня d+1 под меткой d)."""
    orig, nd = F.rows_at, list(F.new_dates())

    def mutant(cut, s, h):
        cut = pd.Timestamp(cut)
        S, G = orig(nd[nd.index(cut) + 1], s, h)
        return S.shift(-1).loc[:cut], G.shift(-1).loc[:cut]

    monkeypatch.setattr(F, "rows_at", mutant)
    res = F.check_pit(src, dates=DATES)
    assert not res["ok"] and res["n_bad"] == len(DATES) and res["max_abs_diff"] > 0


def test_mutation_bars_after_1840_of_day_d_are_caught(src, monkeypatch) -> None:
    """Окно основной сессии расширено до конца вечерней сессии: бары после 18:40 дня d входят в признаки дня d."""
    monkeypatch.setattr(D, "T_TO", "23:59")
    res = F.check_pit(src, dates=DATES)
    assert not res["ok"] and res["n_bad"] >= 1


def test_mutation_s7_normalisation_on_whole_sample_is_caught(src, monkeypatch) -> None:
    monkeypatch.setattr(FE, "vol_ratio", lambda vol, win=FE.WIN_VOL: vol / vol.mean())
    res = F.check_pit(src, dates=DATES)
    assert not res["ok"] and res["n_bad"] == len(DATES)
    assert not F.check_archive(src)["ok"]                                                 # и архив расходится с файлами V6
    h, d = F.load_h(), pd.Timestamp(DATES[0])                                             # прямое свойство: строка d зависит от cut
    a, _ = F.rows_at(d, src, h)
    b, _ = F.rows_at(F.NEW_LAST, src, h)
    assert a.loc[d, "s7"] != b.loc[d, "s7"]


def test_mutation_five_mean_with_day_d_plus_1_is_caught(src, monkeypatch) -> None:
    monkeypatch.setattr(SG, "five_mean", lambda x, win=SG.WIN5: x.rolling(win, min_periods=win).mean().shift(-1))
    res = F.check_pit(src, dates=DATES)
    assert not res["ok"] and res["n_bad"] >= 1


def test_originals_restored_after_mutation_tests() -> None:
    assert FE.vol_ratio.__code__.co_name == "vol_ratio" and SG.five_mean.__code__.co_name == "five_mean"
    assert F.rows_at.__code__.co_name == "rows_at" and D.T_TO == "18:40"
