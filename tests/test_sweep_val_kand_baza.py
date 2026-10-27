"""Базы сравнения validation A′ (Ж-039 п. 1): боевая A через s9.Engine.port и «A только Sk», а не центр сетки.
Данные не читаются: движок s9 — синтетический (`test_s9_shag0._world`)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_s9_shag0 import _world  # noqa: E402

from fastbt import s9, shag0  # noqa: E402
from fastbt import sweep_val as sv  # noqa: E402
from fastbt import sweep_val_baza as bz  # noqa: E402
from fastbt import sweep_val_kand as kd  # noqa: E402

COMM = 0.0004


@pytest.fixture(scope="module")
def spisok():
    return kd.build_spisok(".")


def test_bazy_spiska_A_i_Sk(spisok):
    sp, bazy, _ = spisok
    assert set(bazy) == {"A", *bz.SIDS}
    for m in sp:
        assert "orig_tochki" not in m and bz.baza_ok(m, bazy)
        assert m["baza"]["id"] == ("A" if m["kind"] == "A" else m["sid"])
        assert m["baza"]["konfig"]["semya"] == (None if m["kind"] == "A" else m["sid"])
    assert sum(m["kind"] == "A" for m in sp) == 6 and sum(m["kind"] == "S" for m in sp) == 18


def test_baza_A_v_zamorozke_ravna_s9_A():
    ns = _world()
    got = bz.S9Baza(bz.baza_ref("A"), eng=ns).frame()
    ref = s9.Engine.port(ns, ns.A_row, comm=COMM)
    assert np.array_equal(got["excess"].values, ref["excess"].values) and len(got) > 0


def test_baza_Sk_eto_A_tolko_Sk_a_ne_ves_A():
    ns = _world()
    for s in bz.SIDS:
        got = bz.S9Baza(bz.baza_ref(s), eng=ns).frame()
        ref = s9.Engine.port(ns, shag0.keep_sids(ns.A_row, shag0.sid_row(ns), [s]), comm=COMM)
        assert np.array_equal(got["excess"].values, ref["excess"].values)
        assert not np.array_equal(got["excess"].values, s9.Engine.port(ns, ns.A_row, comm=COMM)["excess"].values)


def test_validation_zakryta_do_otkrytiya():
    b = bz.S9Baza(bz.baza_ref("A"), eng=_world())
    with pytest.raises(PermissionError):
        b.evaluate({}, "validation")


def _zamorozka(tmp, sp, bazy):
    P = sv.Puti(koren=tmp, git_proverka=False)
    P.zamorozka.parent.mkdir(parents=True, exist_ok=True)
    P.zamorozka.write_text(json.dumps(dict(zamorozheno_v="2026-10-28 09:00", sha=sv.sha_kandidatov(sp), N_vsego={}, bazy=bazy,
                                           spisok=sp), sort_keys=True), "utf-8")
    return P


def test_mutatsiya_baza_tsentr_setki_lovitsya(tmp_path, spisok):
    sp, bazy, _ = spisok
    kd.load_zamorozka(_zamorozka(tmp_path / "ok", sp, bazy))                    # верная — читается
    fake = dict(id="A", funktsiya="fastbt.sweep_a.SkPanelSource", konfig={"tochki": "tsentr_setki"}, sha_konfiga="x")
    bad = [dict(m, baza=fake) if m["kind"] == "A" else m for m in sp]            # база A′ = центр сетки; sha пересчитан честно
    with pytest.raises(sv.Stop):
        kd.load_zamorozka(_zamorozka(tmp_path / "bad", bad, dict(bazy, A=fake)))
    swapped = [dict(m, baza=bazy["A"]) if m["kind"] == "S" else m for m in sp]    # база Sk = вся A
    with pytest.raises(sv.Stop):
        kd.load_zamorozka(_zamorozka(tmp_path / "bad2", swapped, bazy))


def test_sweep_val_dry_prohodit(capsys):
    assert sv.main(["--dry"]) == 0
    assert "proshel" in capsys.readouterr().out
