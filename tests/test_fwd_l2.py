"""Л2 (K2 XGBoost, K3 LightGBM на склейке «архив + новые») на РЕАЛЬНОМ архиве; цен новых файлов тест не читает.
Гейты: dz0 и F4 архивных строк побитно с архивным миром; K2/K3 на d 2025-12-30…2026-02-26 при данных, обрезанных по каждому d, = `k23_SBER`
побитно; `k23.parquet` = архив; И2 из метаданных, не пересчитывается (мутации «И2 пересчитан», «И2 подменён»)."""
from __future__ import annotations

import ast
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s5.b2 import feat as b2feat
from models_v2.s10 import hp_k23 as H23, hp_k23_world as KW
from models_v2.s17 import fwd_k23 as K, fwd_src as S

R = S.ROOT
PAN = R / "data/regen/harj_panel_hybrid"
HAVE = (PAN / "k23_SBER.parquet").exists() and S.Paths().new_sber.exists() and (R / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива или новых файлов")
D0 = pd.Timestamp("2025-12-30")
SIG = ["sigma2_k2", "sigma2_k3"]


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return S.Src(S.Paths(lock=tmp_path_factory.mktemp("l2") / "no_lock.json"))


@pytest.fixture(scope="module")
def arch():
    return pd.read_parquet(PAN / "k23_SBER.parquet")


@pytest.fixture(scope="module")
def w_arch():
    """Архивный мир тем же кодом, но из архивных файлов напрямую (`hp_k23.read_bars`, правило IMOEX «auto») — независимый эталон."""
    return KW.build_world(H23.read_bars("SBER"), H23.read_bars("IMOEX"), KW.pool_papers(H23.read_bars))


@pytest.fixture(scope="module")
def w_end(src):
    return K.build_world_at(S.ARCHIVE_END, src)


@pytest.fixture(scope="module")
def w_d0(src):
    return K.build_world_at(D0, src)


def test_dz0_i_f4_arhivnyh_strok_pobitno(w_end, w_arch):
    a, b = w_arch, w_end
    assert same(a["F"].dates, b["F"].dates) and same(a["F"].X, b["F"].X) and same(a["F"].rv, b["F"].rv) and same(a["grid"], b["grid"])
    assert same(a["dz0"].X, b["dz0"].X) and same(a["dz0"].y, b["dz0"].y) and same(a["dz0"].off, b["dz0"].off) and same(a["eps"], b["eps"])
    assert a["dz0"].X.shape[1] == 11 + 11 + 6 and np.isfinite(b["dz0"].X[-300:, -6:]).all()      # 11 базовых + BASE 11 + IDX 6 заполнены
    f4 = list(b2feat.GROUPS["F4"])
    assert f4 and all(same(a["feats"][c].to_numpy(), b["feats"][c].to_numpy()) for c in a["feats"].columns) and set(f4) <= set(b["feats"].columns)


def test_obrezannyj_mir_stroki_do_d0_ravny_arhivu(w_d0, w_arch):
    n = len(w_d0["F"].dates)
    assert pd.Timestamp(w_d0["F"].dates[-1]) == D0 and same(w_d0["F"].dates, w_arch["F"].dates[:n])
    assert same(w_d0["F"].X, w_arch["F"].X[:n]) and same(w_d0["dz0"].X, w_arch["dz0"].X[:n]) and same(w_d0["dz0"].off, w_arch["dz0"].off[:n])
    assert same(w_d0["dz0"].y[:n - 1], w_arch["dz0"].y[:n - 1]) and np.isnan(w_d0["dz0"].y[-1])      # цели строки d ещё нет
    assert all(same(w_d0["feats"][c].to_numpy(), w_arch["feats"][c].to_numpy()[:n]) for c in w_arch["feats"].columns)


def test_zaglushka_t_plus_posle_poslednej_stroki(src, w_d0):
    nxt = K.next_day(D0, src)
    assert nxt == pd.Timestamp("2026-01-05")
    p = K.pad_world(w_d0, nxt)
    n = len(w_d0["F"].dates)
    assert len(p["F"].dates) == n + 1 and pd.Timestamp(p["F"].dates[-1]) == nxt and np.isnan(p["F"].rv[-1]) and p["grid"][-1] == n - 1
    assert same(p["dz0"].X[:n], w_d0["dz0"].X) and same(p["feats"].to_numpy()[:n], w_d0["feats"].to_numpy()) and np.isnan(p["dz0"].X[-1]).all()
    assert K.next_day(S.LAST_SESSION, src) is None                                               # у последнего дня склейки t⁺ не определён


@pytest.mark.parametrize("pos", [0, -1])      # все 38 — скрипт сессии, L2_K23_CUTS38.csv; здесь крайние (лимит хука 300 с)
def test_k2_k3_na_obrezannyh_dannyh_ravny_arhivu_pobitno(src, arch, pos):
    ds = arch[(arch["date"] >= D0) & (arch["t_plus"] <= S.ARCHIVE_END)]["date"].reset_index(drop=True)
    d = ds.iloc[pos]
    fr = K.build_k23(d, src)
    a = arch[arch["date"] == d].reset_index(drop=True)
    assert len(fr) == 1 and fr["date"].iloc[0] == d and fr["t_plus"].iloc[0] == a["t_plus"].iloc[0] and np.isnan(fr["rv"].iloc[0])
    assert same(fr[SIG].to_numpy(), a[SIG].to_numpy()) and fr["hp_year"].iloc[0] == a["hp_year"].iloc[0] == 2026
    assert fr["hp_k2"].iloc[0] == a["hp_k2"].iloc[0] and fr["hp_k3"].iloc[0] == a["hp_k3"].iloc[0]


def test_k23_parquet_tolko_arhivnye_stroki_ravny_arhivu_pobitno(arch):
    out = pd.read_parquet(K.OUT)
    assert out["t_plus"].max() <= S.ARCHIVE_END and out["date"].max() == pd.Timestamp("2026-02-26")
    m = out.merge(arch, on=["date", "t_plus"], suffixes=("", "_a"))
    assert len(m) == len(out) == int(np.isfinite(arch["sigma2_k2"]).sum())
    assert same(m["sigma2_k2"], m["sigma2_k2_a"]) and same(m["sigma2_k3"], m["sigma2_k3_a"]) and same(m["rv"], m["rv_tplus"])
    assert (m["hp_k2"] == m["hp_k2_a"]).all() and (m["hp_k3"] == m["hp_k3_a"]).all() and (m["hp_year"] == m["hp_year_a"]).all()


def test_i2_iz_metadannyh_i_ne_pereschityvaetsya(src, arch, monkeypatch):
    i2 = K.load_i2()
    assert sorted(i2) == list(range(2017, 2027)) and KW.i2_dump(i2) == json.loads((PAN / "k23_meta_SBER.json").read_text())["i2"]

    def boom(*a, **k):
        raise AssertionError("И2 пересчитан")
    for nm in ("choose_i2", "loyo_scores", "first_year", "pairs_before"):
        monkeypatch.setattr(KW, nm, boom)
    d = arch[arch["t_plus"] <= S.ARCHIVE_END]["date"].iloc[-1]
    assert np.isfinite(K.build_k23(d, src)[SIG].to_numpy()).all()                                # весь путь без пересчёта И2


def test_ast_net_vyzova_vybora_i2():
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(ast.parse(open(K.__file__).read())) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not names & {"choose_i2", "loyo_scores", "run_loyo", "i2_dump"}


def test_mutaciya_i2_peresschitan_ili_podmenen_lovitsya(src, arch, monkeypatch):
    d = arch[arch["t_plus"] <= S.ARCHIVE_END]["date"].iloc[-1]
    good = arch[arch["date"] == d][SIG].to_numpy()
    real = K.load_i2()
    mut = dict(real)
    mut[2026] = dataclasses.replace(real[2026], depth=5)                                          # «пересчитанный» выбор года 2026 другой
    monkeypatch.setattr(K, "load_i2", lambda: mut)
    assert not same(K.build_k23(d, src)[SIG].to_numpy(), good)
