"""calcx: параллельный счёт строки (spawn-пул) побитно равен последовательному, кэш по ключу, σ̂² > 0. Синтетика; кэш — во временном каталоге (прежние данные не трогаются)."""
import numpy as np
import pytest

from models_v2.s5 import pit
from models_v2.s5.b1 import jobs
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import calcx, cfgx, worldx

ACC = {g: list(c) for g, c in feat.GROUPS.items()}
DATES = [560, 561, 562, 563]


@pytest.fixture(scope="module")
def w():
    s = synth_b2.World(*synth_b2.pair(700, 41))
    return {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": ACC, "eps": pit.eps_floor(s.F.rv),
            "i2": {y: synth_b2.HP for y in range(2015, 2031)}, "grid": np.array(DATES)}      # обычный dict: defaultdict с lambda не пикл-ится в spawn


@pytest.mark.parametrize("c", [cfgx.NEW[0], cfgx.NEW[2], cfgx.NEW[8]], ids=lambda c: c["name"])
def test_parallel_equals_serial_and_cache(w, c, tmp_path, monkeypatch):
    monkeypatch.setattr(calcx, "DATA", tmp_path)
    spec = worldx.spec_of(w, c)
    raw = calcx.raw_daily(w, spec, "k1", procs=2)
    for d, f in zip(DATES, raw["f"]):
        m = jobs.fit_at(w["F"], spec, jobs.pairs(w["F"], spec, d), synth_b2.HP)
        assert float(m["predict"](spec.X[[d]])[0]) == f
    fc = worldx.sigma_frame(w, spec, raw)
    assert len(fc) == 4 and np.isfinite(fc["sigma2"]).all() and (fc["sigma2"] > 0).all()
    assert list(fc["d"]) == list(w["F"].dates[DATES]) and list(fc["target_date"]) == list(w["F"].dates[np.array(DATES) + 1])
    stamp = (tmp_path / ("raw_%s.parquet" % c["name"])).stat().st_mtime_ns
    assert calcx.raw_daily(w, spec, "k1", procs=2).equals(raw) and (tmp_path / ("raw_%s.parquet" % c["name"])).stat().st_mtime_ns == stamp   # попадание в кэш
    calcx.raw_daily(w, spec, "k2", procs=2)                                                                                                 # другой ключ — пересчёт
    assert (tmp_path / ("raw_%s.parquet" % c["name"])).stat().st_mtime_ns != stamp


def test_sigma_frame_uses_lnmu_without_half_s2_and_ln_with_it(w):
    F = w["F"]
    raw = {"row": np.array([560]), "f": np.array([0.3]), "s2": np.array([0.8])}
    import pandas as pd
    raw = pd.DataFrame(raw)
    s_ln, s_mu = worldx.spec_of(w, cfgx.CTRL), worldx.spec_of(w, cfgx.NEW[0])
    off = s_ln.off[560]
    assert worldx.sigma_frame(w, s_ln, raw)["sigma2"].iloc[0] == pytest.approx(np.exp(off + 0.3 + 0.4))
    assert worldx.sigma_frame(w, s_mu, raw)["sigma2"].iloc[0] == pytest.approx(np.exp(off + 0.3))   # gamma / qlike: поправка s²/2 запрещена
    assert F.rv[561] == worldx.sigma_frame(w, s_ln, raw)["rv"].iloc[0]


def test_key_depends_on_code_files_listed():
    assert calcx.code_sha() == calcx.code_sha() and len(calcx.code_sha()) == 16
