"""X1-пилот P0, G4 и сквозные: Йенсен в ln F на синтетике; мутация «смешивание в F вместо ln F» меняет вывод; P0 ничего не пишет; число в отчёте подменено — ловится."""
import hashlib
from pathlib import Path

import numpy as np

import models_v2.s16  # noqa: F401
from models_v2.s16 import run_x1 as R, x1_p0_mix as X

ROOT = Path(__file__).resolve().parents[1]


def synth(n=500, seed=0):
    rng = np.random.default_rng(seed)
    F = np.exp(rng.normal(0, 0.6, (n, 3)))
    p = rng.dirichlet(np.ones(3), n)
    return np.exp(rng.normal(0, 0.5, n)), F, p


def test_jensen_in_log_space_never_violated_known_answer():
    rv, F, p = synth()
    c = X.check_op(rv, F, p)
    assert (c["gap_log"] >= -1e-12).all()


def test_mutation_mixing_in_F_instead_of_lnF_changes_verdict():
    rv, F, p = synth()
    c = X.check_op(rv, F, p)
    assert (c["gap_ar"] < -1e-12).mean() > 0.05                      # в F нарушения есть
    assert (c["F_ar"] >= c["F_geo"] - 1e-12).all()                  # AM–GM
    assert not np.allclose(c["F_ar"], c["F_geo"])


def test_equal_experts_give_zero_gap_and_no_bias():
    rv = np.array([1.0, 2.0])
    F = np.array([[1.5, 1.5], [0.7, 0.7]])
    c = X.check_op(rv, F, np.full((2, 2), 0.5))
    assert np.allclose(c["gap_log"], 0) and np.allclose(c["F_geo"], c["F_ar"])


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if f.is_file():
                h.update(str(f).encode() + f.read_bytes())
    return h.hexdigest()


def test_p0_computations_write_nothing_to_v6_v7_de():
    before = digest(R.WATCH)
    X.mix_table()
    R.SC.scale_table()
    assert digest(R.WATCH) == before


def test_p0_modules_have_no_write_calls_to_forecast_dirs():
    for f in (ROOT / "models_v2/s16").glob("x1_p0_*.py"):
        src = f.read_text(encoding="utf-8")
        assert "to_parquet" not in src and "write_text" not in src and ".to_csv" not in src, f.name


def test_report_number_substitution_is_caught():
    scale, init, mix, dyn = (R.read_csv(n) for n in ("P0_SCALE.csv", "P0_INIT.csv", "P0_MIX.csv", "P0_DYN.csv"))
    good = R.build_report(scale, init, mix, dyn)
    assert good == (R.REP / "P0_REPORT.md").read_text(encoding="utf-8")
    bad = good.replace("0.99829", "0.91829", 1)
    assert bad != good
    mix2 = mix.copy()
    mix2.loc[(mix2["operator"] == "X1") & (mix2["year"].astype(str) == "all"), "bias_geo_over_arith"] = 0.91829
    assert R.build_report(scale, init, mix2, dyn) != good
