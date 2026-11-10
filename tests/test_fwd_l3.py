"""Л3 (склад, X_w5, z энкодера 2026, e_d, s_d) на РЕАЛЬНОМ архиве. Цен новых файлов тест не читает (только begin через Л1).
Гейты: архивные строки склада и X_w5 побитно с `store_raw.npz` / `X_w5.npz`; z 2026 = артефакт `tcn_w5_2026_s*` (bitwise), content_hash = CHECKPOINT V7;
e_d = колонка `s` x1.parquet; s_d = `x1_ya_feat.features`; выходы data/fwd/models/ = пересчёт. Мутации и обрезка — `test_fwd_l3_mut.py`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_ya_feat as YF
from models_v2.s17 import fwd_src as S, fwd_store as FS, fwd_z as FZ

R = S.ROOT
V7 = R / "data/regen/harj_hybrid_v7"
X1 = R / "data/regen/harj_deep/E1/x1.parquet"
HAVE = (V7 / "store_raw.npz").exists() and X1.exists() and S.Paths().new_sber.exists() and (V7 / "stageA/tcn_w5_2026_s4.npz").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива V7 или begin новых файлов")
SEEDS = FZ.SEEDS


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return S.Src(S.Paths(lock=tmp_path_factory.mktemp("l3") / "no_lock.json"))


@pytest.fixture(scope="module")
def built(src):
    store = FS.build_store(S.ARCHIVE_END, src)
    xw = FS.build_xw5(store)
    return dict(store=store, arr=FS.store_arrays(store), xw=xw, st=FS.build_states(S.ARCHIVE_END, src, store))


def test_sklad_arhiv_pobitno_store_raw(built):
    ref = np.load(V7 / "store_raw.npz")
    assert sorted(ref.files) == sorted(built["arr"])
    for k in ref.files:
        assert FS.same(ref[k], built["arr"][k]), k
    assert built["arr"]["raw"].dtype == np.float32 and len(ref["days"]) == 2944


def test_xw5_arhiv_pobitno(built):
    ref = np.load(V7 / "X_w5.npz")
    assert sorted(ref.files) == sorted(built["xw"])
    for k in ref.files:
        assert FS.same(ref[k], built["xw"][k]), k
    assert built["xw"]["X"].dtype == np.float32 and built["xw"]["X"].shape == (2944, 5, 53, 11)


def test_z_2026_ravno_artefaktu_pobitno_vse_seeds(built, src):
    for s in SEEDS:
        a = np.load(V7 / f"stageA/tcn_w5_2026_s{s}.npz")
        z = FZ.build_z(S.ARCHIVE_END, src, s, built["store"], built["xw"])
        assert np.array_equal(z["d"].to_numpy("datetime64[ns]").astype(np.int64), a["days"]), s
        assert np.array_equal(z.iloc[:, 1:].to_numpy(), a["z"]), s
        w = z[(z["d"] >= "2026-01-05") & (z["d"] <= "2026-02-27")]
        assert len(w) > 30 and w["d"].iloc[0] == pd.Timestamp("2026-01-05") and w["d"].iloc[-1] == S.ARCHIVE_END


def test_content_hash_artefakta_ravno_checkpoint_i_porcha_lovitsya(monkeypatch):
    for s in SEEDS:
        assert FZ.load_artifact(s)["pca_comps"].shape == (4, 16)
    orig = FZ.STA.load

    def spoiled(path):
        a = orig(path)
        a["w__head.bias"] = a["w__head.bias"] + np.float32(1e-6)
        return a
    monkeypatch.setattr(FZ.STA, "load", spoiled)
    with pytest.raises(ValueError, match="content_hash"):
        FZ.load_artifact(0)


def test_pca_ne_pereschityvaetsya_fit_pca_past_ne_vyzyvaetsya(built, src, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("PCA пересчитан")
    monkeypatch.setattr(FZ.REP, "fit_pca_past", boom)
    z = FZ.build_z(S.ARCHIVE_END, src, 1, built["store"], built["xw"])
    assert len(z) == 2925


def test_e_d_ravno_kolonke_s_x1(built):
    x1 = pd.read_parquet(X1)
    m = built["st"].set_index("d").loc[x1["d"]]
    assert np.array_equal(m["e"].to_numpy(), x1["s"].to_numpy()) and m["e"].sum() == x1["s"].sum() > 0
    assert (m["t_plus"].to_numpy("datetime64[ns]") == x1["t_plus"].to_numpy("datetime64[ns]")).all()


def test_s_d_ravno_x1_ya_feat_features(built):
    x1 = pd.read_parquet(X1)
    ref = YF.features(YF.load_store(), pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"]))
    m = built["st"].set_index("d").loc[x1["d"]]
    for k, c in (("s", "s"), ("n", "n"), ("nlast", "nlast"), ("zero", "zero"), ("rv_d", "rv_d"), ("delta", "delta")):
        assert np.array_equal(np.asarray(ref[k]), m[c].to_numpy()), k
    assert m["zero"].sum() >= 0 and np.isfinite(m["s"]).all()


def test_vyhody_data_fwd_models_ravny_pereschetu(built):
    for name in ("store.npz", "X_w5.npz", "state_es.parquet") + tuple(f"z_tcn2026_s{s}.npz" for s in SEEDS):
        assert (FS.OUT / name).exists(), name
    a, x = np.load(FS.OUT / "store.npz"), np.load(FS.OUT / "X_w5.npz")
    assert all(FS.same(a[k], built["arr"][k]) for k in built["arr"]) and all(FS.same(x[k], built["xw"][k]) for k in built["xw"])
    pd.testing.assert_frame_equal(pd.read_parquet(FS.OUT / "state_es.parquet"), built["st"])
    for s in SEEDS:
        f, ref = np.load(FS.OUT / f"z_tcn2026_s{s}.npz"), np.load(V7 / f"stageA/tcn_w5_2026_s{s}.npz")
        assert np.array_equal(f["z"], ref["z"]) and str(f["content_hash"]) == FZ.frozen_hash(s)


def test_arhivnyj_period_kalendar_dnej_sklada_ravna_dnyam_arhiva(built):
    days = pd.DatetimeIndex(np.load(V7 / "store_raw.npz")["days"])
    assert pd.DatetimeIndex(built["st"]["d"]).equals(days) and built["st"]["t_plus"].isna().sum() == 1
