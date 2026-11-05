"""G23–G27 на реальных строках: Г3–Г5 из models_v2/s3/chain.py на реальных прогнозах этапов 1, 2 и Б.
Главный тест: прогноз на d побитно не меняется при порче всего, что известно только после d; мутации m3, m5, m6, m7."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import run_p3  # noqa: E402
from models_v2.s3 import chain, post  # noqa: E402

OUT = run_p3.OUT
pytestmark = pytest.mark.skipif(not (OUT / "forecasts_pb.parquet").exists(), reason="нет прогнозов этапов 1–2 и Б")
KK = [300, 900, 1500, 2100]


@pytest.fixture(scope="module")
def real():
    p1, p2, pb_ = (pd.read_parquet(OUT / ("forecasts_%s.parquet" % s)) for s in ("p1", "p2", "pb"))
    ref, ln, *_ = run_p3.inputs(p1, p2, pb_)
    rv_w = pd.read_parquet(OUT / "daily_p1.parquet")["rv_w"].reindex(ref["d"]).to_numpy()
    return {"ref": ref, "ln": ln, "rv_w": rv_w, "rvh": run_p3.regime_high(ref)}


def make(real, ln=None, rv=None, rv_w=None, rvh=None):
    r = real["ref"]
    return chain.prep(r["d"].values, r["target_date"].values, r["rv"].values if rv is None else rv,
                      real["rv_w"] if rv_w is None else rv_w, real["rvh"] if rvh is None else rvh), (real["ln"] if ln is None else ln)


def damaged(real, kk):
    """Всё, что известно после d_kk: ln-прогнозы строк j > kk, RV_w и RVI в строках j > kk, цель строки kk и позже."""
    rng = np.random.default_rng(11)
    n = len(real["rv_w"])
    ln = {k: v.copy() for k, v in real["ln"].items()}
    for v in ln.values():
        v[kk + 1:] = rng.normal(2, 1, n - kk - 1)
    rv = real["ref"]["rv"].to_numpy().copy()
    rv[kk:] = np.exp(rng.normal(2, 1, n - kk))
    rv_w, rvh = real["rv_w"].copy(), real["rvh"].copy()
    rv_w[kk + 1:] = np.exp(rng.normal(2, 1, n - kk - 1))
    rvh[kk + 1:] = rng.integers(0, 2, n - kk - 1)
    return make(real, ln, rv, rv_w, rvh)


@pytest.fixture(scope="module")
def base(real):
    a, ln = make(real)
    sig, aux = chain.build(a, ln)
    return a, ln, sig, aux["choice"]


def diff_after_damage(real, choice, kk):
    """Модели, чей прогноз на строках <= kk изменился от порчи данных после d_kk (при активных подменах — мутациях)."""
    s1, _ = chain.build(*make(real), choice)
    s2, _ = chain.build(*damaged(real, kk), choice)
    return [n for n in s1 if s1[n][:kk + 1].tobytes() != s2[n][:kk + 1].tobytes()]


def test_saved_forecasts_equal_recomputed_chain(real, base):
    fc = pd.read_parquet(OUT / "forecasts_p3.parquet")
    for n, s in base[2].items():
        assert fc[fc.model == n].sort_values("d")["sigma2"].to_numpy().tobytes() == s.tobytes(), n


@pytest.mark.parametrize("kk", KK)
def test_main_g3_g4_g5_unchanged_by_data_after_d(real, base, kk):
    assert diff_after_damage(real, base[3], kk) == []


def test_mut_m3_closed_includes_d_is_caught(real, base, monkeypatch):
    orig = post.closed_count
    monkeypatch.setattr(post, "closed_count", lambda d, td: orig(d, td) + 1)
    caught = {n for kk in KK for n in diff_after_damage(real, base[3], kk)}
    assert "g3" in caught and any(n.startswith("g5_") for n in caught), caught


def test_q67_on_real_rows_excludes_d_and_mutation_m5_changes_stress(real, base):
    a, rw = base[0], real["rv_w"]
    for k in (260, 1000, 2000):
        assert a["q"][k] == np.quantile(rw[k - 252:k], 2 / 3)
    assert np.isnan(a["q"][:252]).all()
    mut = np.array([np.nan] * 252 + [np.quantile(rw[k - 251:k + 1], 2 / 3) for k in range(252, len(rw))])
    assert (mut[252:] != a["q"][252:]).mean() > 0.5  # м5: окно с днём d на реальном ряду даёт другой порог
    assert (post.stress(real["rvh"], rw, mut) != a["st"]).sum() > 0


def ewma_independent(rv, s2, m, k, h):
    r = rv[:m[k]] / s2[:m[k]]
    w = 0.5 ** ((m[k] - 1 - np.arange(m[k])) / h)
    return float(np.clip((w * r).sum() / w.sum(), 0.7, 1.5))


def test_ewma_on_real_rows_matches_independent_and_mutation_m6_caught(real, base, monkeypatch):
    a, ln, sig, ch = base
    s2 = np.exp(_base_ln(base))
    k = post.ewma_k(a["rv"], s2, a["m"], 20.0)
    for j in (300, 1200, 2200):
        assert k[j] == pytest.approx(ewma_independent(a["rv"], s2, a["m"], j, 20.0), abs=1e-12)
    rev = lambda rv, s, m, h, clip=post.CLIP: np.array([  # noqa: E731  м6: возраст 0 у старейшего
        np.clip(((0.5 ** (np.arange(m[j]) / h)) * (rv / s)[:m[j]]).sum() / (0.5 ** (np.arange(m[j]) / h)).sum(), *clip) if m[j] else 1.0
        for j in range(len(rv))])
    assert np.abs(rev(a["rv"], s2, a["m"], 20.0) - k).max() > 1e-3


def _base_ln(base):
    a, ln, sig, ch = base
    pool = dict(ln, **{n: np.log(s) for n, s in sig.items()})
    return pool[ch["base"]]


def test_clip_on_real_rows_and_mutation_m7_caught(real, base):
    a, ln, sig, ch = base
    s2 = np.exp(_base_ln(base))
    ratio = sig["g5_always_h10"] / s2
    assert ratio.min() >= 0.7 - 1e-12 and ratio.max() <= 1.5 + 1e-12 and ratio.max() > 1.0 and ratio.min() < 1.0
    r = a["rv"] / s2
    no_clip = np.array([((0.5 ** ((a["m"][j] - 1 - np.arange(a["m"][j])) / 10.0)) * r[:a["m"][j]]).sum()
                        / (0.5 ** ((a["m"][j] - 1 - np.arange(a["m"][j])) / 10.0)).sum() if a["m"][j] else 1.0 for j in range(2272)])
    assert (no_clip < 0.7).any() or (no_clip > 1.5).any()  # м7: без клипа k выходит за границы на реальных строках
