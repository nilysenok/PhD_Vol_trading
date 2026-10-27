"""Мутации к `test_sweep_val_kand.py`: подложенная ошибка -> детектор обязан упасть (False). Подложено в памяти, файлы не правятся."""
import random

import pytest

from fastbt import sweep_val as sv
from fastbt import sweep_val_kand as kd
from test_sweep_val_kand import (ok_dubli_svedeny, ok_otkaz_bez_da_do_zamka, ok_otkaz_posle_kommita, ok_spisok_sostav,
                                 ok_tablitsa_sborok, ok_vremya_iz_fayla, spisok)  # noqa: F401  (spisok — фикстура)


def test_mut_dubli_ne_svodyatsya(monkeypatch):
    monkeypatch.setattr(kd, "eff_key", lambda sid, p: repr(sorted(p.items(), key=str)))      # полный ключ: неактивное различает
    monkeypatch.setattr(kd, "_sig", lambda r: r.name)                                        # и подписи результата нет
    assert not ok_dubli_svedeny()


def test_mut_pravka_posle_kommita_ne_proveryaetsya(monkeypatch, tmp_path, spisok):
    monkeypatch.setattr(kd, "_commit_ok", lambda P: None)
    assert not ok_otkaz_posle_kommita(tmp_path, spisok)


def test_mut_da_ne_proveryaetsya(monkeypatch, tmp_path, spisok):
    monkeypatch.setattr(sv, "check_da", lambda spec, sha, P: None)
    assert not ok_otkaz_bez_da_do_zamka(tmp_path, monkeypatch, spisok)


def test_mut_zamorozka_iz_spec_a_ne_iz_fayla(monkeypatch, tmp_path):
    real = kd.load_zamorozka
    monkeypatch.setattr(kd, "load_zamorozka", lambda P: {**real(P), "zamorozheno_v": "2000-01-01 00:00"})   # время «из spec»
    assert not ok_vremya_iz_fayla(tmp_path)


def test_mut_fabrika_nedeterminirovana(monkeypatch):
    real = kd.kandidaty

    def noisy(sid, root=".", k=kd.TOP_K):
        d = real(sid, root, k)
        d["cands"][0]["plateau"] += random.random()
        return d
    monkeypatch.setattr(kd, "kandidaty", noisy)
    assert kd.build_spisok(".") != kd.build_spisok(".")


def test_mut_top2_sborok_po_kontrolyu_a_ne_po_tseli():
    import pandas as pd
    t = pd.read_csv("docs/tables/sweep_e1v2/A_sborki.csv", comment="#")
    s = t[t.kind == "sborka"].sort_values("ctrl_sharpe_c0.04", ascending=False)       # подложено: отбор по контролю 2019
    t["top2"] = 0
    t.loc[s.index[:2], "top2"] = [1, 2]
    assert not ok_tablitsa_sborok(t)


def test_mut_sostav_spiska_top2_sborki_odna(monkeypatch, spisok):
    monkeypatch.setattr(kd, "top2_sborki", lambda root=".": [{**kd.NET, "plateau_neighbors": 8}])
    with pytest.raises(Exception):
        assert ok_spisok_sostav(kd.build_spisok("."))                 # 18 + 3 ≠ 24: детектор падает на одной сборке
