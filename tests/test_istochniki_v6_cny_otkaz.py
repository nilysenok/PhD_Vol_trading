"""wfП-5 (Д-d): wf без файла CNYRUB_TOM — FileNotFoundError, а не None («только USD», пустая ось usdrub после 11.06.2024).

Синтетика: пустой теневой корень без файлов. train4/v2 (≤ 2021): CNY не нужен, поведение прежнее (None, как у любого отсутствующего индекса).
"""
import pandas as pd
import pytest

from fastbt import istochniki_v6 as iv

END = pd.Timestamp("2022-01-01")


def _otkaz(cls, tmp_path) -> bool:
    try:
        cls(END, tmp_path, "wf").indeks("cnyrub")
    except FileNotFoundError:
        return True
    return False


def test_wf_bez_cny_otkaz_s_prichinoj(tmp_path):
    with pytest.raises(FileNotFoundError, match="CNYRUB_TOM"):
        iv.Fajly(END, tmp_path, "wf").indeks("cnyrub")


def test_wf_otsutstvie_drugih_ryadov_prezhnee(tmp_path):
    assert iv.Fajly(END, tmp_path, "wf").indeks("rvi") is None


@pytest.mark.parametrize("period", ["train4", "v2", None])
def test_do_2022_cny_ne_nuzhen_none_kak_prezhde(tmp_path, period):
    f = iv.Fajly(END, tmp_path, period)
    assert f.indeks("cnyrub") is None and f.indeks("usdrub") is None


def test_mutatsiya_none_vmesto_otkaza_lovitsya(tmp_path):
    class Mut(iv.Fajly):
        def indeks(self, name):          # прежнее поведение: нет файла — None
            return None

    assert _otkaz(iv.Fajly, tmp_path) and not _otkaz(Mut, tmp_path)
