"""E1в: подключение блока `rm_sqrt_rv` к оценщику (sweep_setka.expand -> sweep_post.split -> SkEvaluator). Дым на реальном 2018, SBER."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fastbt import rm_sqrt_rv as rms
from fastbt import sweep_post as sp
from fastbt import sweep_setka as sk
from fastbt.sigprov import SIG_GRIDS

DESIGN = Path(__file__).resolve().parent.parent / "fastbt" / "designs" / "sweep"
SIDS = ["S1", "S2", "S3", "S4", "S5", "S6"]


@pytest.mark.parametrize("sid", SIDS)
def test_defaults_ravny_ishodnomu_v_setke(sid):
    s = yaml.safe_load((DESIGN / f"{sid}_setka_e1b.yaml").read_text("utf-8"))
    ish = {p["imya"]: p["ishodnoe"] for p in s["bloki"]["rm_sqrt_rv"]["parametry"]}
    d = rms.defaults(sid)
    assert all(d[k] == v for k, v in ish.items()), (ish, d)


@pytest.mark.parametrize("sid", SIDS)
def test_vse_klyuchi_sqrt_rv_ushli_iz_vne_v1_i_prinyaty(sid):
    e = sk.expand(DESIGN / f"{sid}_setka_e1b.yaml")
    assert not any("sqrt_rv" in s or "okno_rv" in s or "k_sl" in s or "rm_edinitsa" in s for s in e["vne_v1"]), e["vne_v1"]
    assert "rm_sqrt_rv.k_sl" in e["grid"] and ("rm_edinitsa" in e["grid"] or "rm_tip" in e["grid"])
    names = set(SIG_GRIDS[sid])
    n = 0
    for key, vals in e["grid"].items():
        for v in vals:
            if key.startswith("rm_sqrt_rv."):
                rtype, rm, _, _ = sp.split(sid, {**sk.oat_base(e, key), key: v}, names)
                assert rtype == "sqrt_rv" and rm[key.split(".")[1]] == v
                n += 1
    assert n == sum(len(v) for k, v in e["grid"].items() if k.startswith("rm_sqrt_rv.")) >= 20


def test_dym_s3_real_2018_sqrt_menyaet_pozitsii():
    from fastbt.sweep_sk import SkEvaluator
    ev = SkEvaluator("S3", ["SBER"])
    base = sk.expand(DESIGN / "S3_setka_e1b.yaml")["base"]
    sg = ev.signals({k: v for k, v in base.items() if k in ev.sig_names})
    end = pd.Timestamp("2019-01-01")
    a = ev.pnl(sg, base, 0.0004, end)
    b = ev.pnl(sg, {**base, "rm_edinitsa": "sqrt_rv"}, 0.0004, end)
    c = ev.pnl(sg, {**base, "rm_edinitsa": "sqrt_rv", "rm_sqrt_rv.k_sl": 1.0, "rm_sqrt_rv.k_trail": 1.0}, 0.0004, end)
    assert len(b) == len(a) and np.isfinite(b["excess"]).all()
    assert not np.array_equal(a["exposure"], b["exposure"]) and not np.array_equal(b["exposure"], c["exposure"])
