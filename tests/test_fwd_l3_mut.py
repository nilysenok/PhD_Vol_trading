"""Л3: обрезка, AST-изоляция энкодера (класс № 141), замок до цен новых файлов, мутации «PCA пересчитан», «порог e_d с днём d»,
«объём в лотах», «бар дня d+1 в складе дня d», «день d в медиане слота», «обрезка по cut снята». Реальный архив; цен новых файлов не читает."""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_har as HH
from models_v2.s14 import v7_0_seq as SQ
from models_v2.s15 import de_e1_state as E1
from models_v2.s17 import fwd_src as S, fwd_store as FS, fwd_z as FZ

R = S.ROOT
V7 = R / "data/regen/harj_hybrid_v7"
HAVE = (V7 / "store_raw.npz").exists() and S.Paths().new_sber.exists() and (V7 / "stageA/tcn_w5_2026_s0.npz").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива V7 или begin новых файлов")
D0 = pd.Timestamp("2025-12-30")
BANNED = {"train_encoder", "fit_pca_past", "train_missing"}


def nolock(tmp, **kw) -> S.Src:
    return S.Src(S.Paths(lock=tmp / "no_lock.json", **kw))


def outputs(src, cut=S.ARCHIVE_END) -> dict:
    store = FS.build_store(cut, src)
    xw = FS.build_xw5(store)
    return dict(arr=FS.store_arrays(store), xw=xw, st=FS.build_states(cut, src, store), z=FZ.build_z(cut, src, 0, store, xw))


def prefix_equal(a: dict, b: dict) -> bool:
    """Строки <= D0 двух прогонов совпадают побитно: склад, X_w5, e_d/s_d, z."""
    n0 = int((pd.DatetimeIndex(a["arr"]["days"]) <= D0).sum())
    ok = all(FS.same(a["arr"][k][:n0], b["arr"][k][:n0]) for k in ("raw", "present", "days"))
    ok &= all(FS.same(a["xw"][k][:n0], b["xw"][k][:n0]) for k in ("X", "mask", "day_ok", "norm_ok", "days"))
    ok &= a["st"].iloc[:n0].equals(b["st"].iloc[:n0])
    za, zb = a["z"][a["z"]["d"] <= D0], b["z"][b["z"]["d"] <= D0]
    return bool(ok and za.equals(zb) and len(za) > 2000)


def bad_src(tmp, src) -> S.Src:
    """Бары SBER и IMOEX после d0 умножены на случайный множитель; новые файлы — не читаются (cut <= 2026-02-27)."""
    (tmp / "b").mkdir()
    HH.corrupt_after(pd.read_parquet(src.archive_path("SBER")), D0).to_parquet(tmp / "b/SBER.parquet")
    HH.corrupt_after(pd.read_parquet(src.archive_path("IMOEX")), D0).to_parquet(tmp / "IMOEX.parquet")
    return nolock(tmp, bars_dir=tmp / "b", imoex_archive=tmp / "IMOEX.parquet")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return nolock(tmp_path_factory.mktemp("l3m"))


@pytest.fixture(scope="module")
def clean(src):
    return outputs(src)


@pytest.fixture(scope="module")
def bad(tmp_path_factory, src):
    return bad_src(tmp_path_factory.mktemp("l3bad"), src)


def violations(text: str) -> list:
    """Запрещённые обращения: train_encoder / fit_pca_past / train_missing (имя, атрибут, импорт), `v7_1b_sta.compute` (атрибут STA.compute, импорт compute)."""
    out = []
    for n in ast.walk(ast.parse(text)):
        names = {a.name for a in getattr(n, "names", [])} if isinstance(n, (ast.Import, ast.ImportFrom)) else set()
        if isinstance(n, ast.Attribute):
            names.add(n.attr)
            if n.attr == "compute" and isinstance(n.value, ast.Name) and n.value.id in ("STA", "v7_1b_sta"):
                out.append("v7_1b_sta.compute")
        elif isinstance(n, ast.Name):
            names.add(n.id)
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("v7_1b_sta") and "compute" in names:
            out.append("v7_1b_sta.compute")
        out += sorted(names & BANNED)
    return out


def test_ast_fwd_moduli_ne_importiruyut_obuchenie_i_skaner_lovit_myutacii():
    files = sorted((R / "models_v2/s17").glob("fwd_*.py"))
    assert {f.name for f in files} >= {"fwd_store.py", "fwd_z.py", "fwd_src.py"}
    for f in files:
        assert violations(f.read_text(encoding="utf-8")) == [], f.name
    for bad_text in ("from models_v2.s14.v7_1e_pre import train_encoder", "REP.fit_pca_past(z, d, 2026)", "STA.compute('tcn', 2026, 0, p)",
                     "from models_v2.s14.v7_1b_sta import compute", "STA.train_missing()", "import x\ntrain_encoder(1)"):
        assert violations(bad_text), bad_text


def test_obuchenie_ne_vyzyvaetsya_vo_vremya_build_z(clean, src, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("обучение вызвано")
    for mod, name in ((FZ.STA, "compute"), (FZ.STA, "train_missing"), (FZ.REP, "fit_pca_past")):
        monkeypatch.setattr(mod, name, boom)
    import models_v2.s14.v7_1e_pre as PRE
    monkeypatch.setattr(PRE, "train_encoder", boom)
    assert len(FZ.build_z(S.ARCHIVE_END, src, 2)) == 2925


def test_obrezka_porcha_barov_posle_d0_ne_menyaet_stroki_do_d0(clean, bad, src):
    out = outputs(bad)
    assert prefix_equal(clean, out)
    assert not FS.same(out["arr"]["raw"][-20:], clean["arr"]["raw"][-20:])                   # порча видна после d0
    assert FS.same(FS.build_store(D0, bad)["raw"], FS.build_store(D0, src)["raw"])               # обрезка по cut: порченых баров нет в складе


def test_obrezka_po_cut_store_do_d0_ravnyj_prefiksu_polnogo(clean, src):
    cut = FS.build_store(D0, src)
    n = len(cut["days"])
    assert cut["days"][-1] == D0 and FS.same(cut["raw"].astype(np.float32), clean["arr"]["raw"][:n])
    assert FS.same(FS.build_xw5(cut)["X"], clean["xw"]["X"][:n])


def test_zamok_novye_ceny_ne_chitayutsya_a_begin_da(src, monkeypatch):
    seen, orig = [], pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **kw: (seen.append((str(p), kw.get("columns"))), orig(p, *a, **kw))[1])
    FS.archive_outputs(src)
    newp = [c for p, c in seen if "/data/fwd/" in p]
    assert newp and all(c == ["begin"] for c in newp) and not any(p.endswith("rv_fwd_SEALED.parquet") for p, _ in seen)
    for fn in (lambda: FS.build_store(pd.Timestamp("2026-03-02"), src), lambda: FZ.build_z(S.LAST_SESSION, src, 0)):
        with pytest.raises(S.SealedError):
            fn()


def test_myutaciya_pca_pereschitan_po_vsej_vyborke_lovitsya(clean, src, monkeypatch):
    ref = np.load(V7 / "stageA/tcn_w5_2026_s0.npz")["z"]
    days = pd.DatetimeIndex(clean["z"]["d"])
    tp, fit = FZ.REP.transform_pca, FZ.REP.fit_pca_past
    assert np.array_equal(clean["z"].iloc[:, 1:].to_numpy(), ref)
    monkeypatch.setattr(FZ.REP, "transform_pca", lambda p, raw: tp(fit(raw, days, 2027), raw))      # главные компоненты по z всех дат <= 2026
    z = FZ.build_z(S.ARCHIVE_END, src, 0).iloc[:, 1:].to_numpy()
    assert not np.array_equal(z, ref) and np.abs(z - ref).max() > 1e-6


def synth_store() -> tuple:
    """10 одинаковых дней прошлого (50 баров |r| = k/50 и 3 бара |r| = 1: 99-й перцентиль прошлого = 1, e_d = 0) и день d, все 53 бара которого = 1,5:
    честный e_d = 1. Если в порог входит сам день d (53 бара на вершине), перцентиль = 1,5 и e_d = 0."""
    days = pd.bdate_range("2020-01-06", periods=11)
    raw = np.zeros((11, 53, len(SQ.CH)))
    day = np.concatenate([np.arange(1, 51) / 50.0, [1.0, 1.0, 1.0]]) * np.where(np.arange(53) % 2 == 0, 1.0, -1.0)
    raw[:10, :, SQ.CH.index("r")] = day
    raw[10, :, SQ.CH.index("r")] = 1.5
    raw[:, :, SQ.CH.index("r2")] = raw[:, :, SQ.CH.index("r")] ** 2
    return dict(days=days, raw=raw, present=np.ones((11, 53), bool)), 1.5, 1.0


def test_myutaciya_porog_e_d_s_dnyom_d_lovitsya(src, monkeypatch):
    store, x, q = synth_store()
    assert x > q
    assert FS.build_states(S.ARCHIVE_END, src, store)["e"].tolist() == [0] * 10 + [1]           # известный ответ: порог по барам строго до d
    monkeypatch.setattr(E1, "past_end", lambda p: p + 1)
    assert FS.build_states(S.ARCHIVE_END, src, store)["e"].tolist() == [0] * 11                  # мутация: день d в пороге — ловится


def test_myutaciya_obyom_v_lotah_lovitsya(src):
    class Lots(S.Src):
        def bars(self, inst, cut):
            b = super().bars(inst, cut).copy()
            if inst == "SBER":
                b["volume"] = b["volume"].astype(float) / 10          # лот SBER = 10 акций
            return b
    arr = FS.store_arrays(FS.build_store(S.ARCHIVE_END, Lots(src.p)))
    assert not FS.same(arr["raw"], np.load(V7 / "store_raw.npz")["raw"])


def test_myutaciya_bar_dnya_d_plus_1_v_sklade_dnya_d_lovitsya(clean, bad, src, monkeypatch):
    ref = np.load(V7 / "X_w5.npz")["X"]
    monkeypatch.setattr(SQ, "stats_row", lambda store, i: min(i + 1, len(store["days"]) - 1))     # нормировка дня d видит сессию d+1
    a = FS.build_xw5(FS.build_store(S.ARCHIVE_END, src))
    b = FS.build_xw5(FS.build_store(S.ARCHIVE_END, bad))
    assert not FS.same(a["X"], ref)
    n0 = int((pd.DatetimeIndex(a["days"]) <= D0).sum())
    assert not FS.same(a["X"][:n0], b["X"][:n0])                                                 # порча после d0 меняет строки <= d0


def test_myutaciya_den_d_v_mediane_slota_lovitsya(src, monkeypatch):
    monkeypatch.setattr(SQ, "slot_median_before", lambda vol: pd.DataFrame(vol).expanding(min_periods=1).median().to_numpy())
    arr = FS.store_arrays(FS.build_store(S.ARCHIVE_END, src))
    assert not FS.same(arr["raw"], np.load(V7 / "store_raw.npz")["raw"])


def test_myutaciya_cut_ne_obrezaet_lovitsya(bad, src, tmp_path, monkeypatch):
    clean_cut = FS.build_store(D0, src)["raw"]
    assert FS.same(FS.build_store(D0, bad)["raw"], clean_cut)
    only_archive = nolock(tmp_path, bars_dir=bad.p.bars_dir, imoex_archive=bad.p.imoex_archive, new_sber=tmp_path / "net.parquet", new_mkt_dir=tmp_path)
    monkeypatch.setattr(S, "limit_of", lambda cut: pd.Timestamp.max)                             # мутация: обрезка по cut снята
    assert not FS.same(FS.build_store(D0, only_archive)["raw"], clean_cut)
