"""A_zashch = A (s9) (Ж-124 п. 1): помощник scripts/revers/r1_a_zashch.py.
Данные (кэш s9, Z2-вход) — из основной копии; нет среды FASTBT_Z2_* или кэша — тест пропускается, не проходит молча.
Подмножество (объявлено): test1 — все 17 тикеров × 6 Sk; train4 — позиции AFLT/S3 против независимого пересчёта ApproachA → turnover.apply.
Мутации (модульных подмен нет, mutations.md № 43): A_fix-подмена ансамбля на train4; перепутанные t0/t1; short вместо long; потеря знака."""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
sys.path.insert(0, str(ROOT / "tests"))
import r1_a_zashch as Z  # noqa: E402
from fastbt import s9  # noqa: E402

TOL = 1e-12
A_CD = ROOT / "docs" / "tables" / "revers_r1" / "a_cd_test1.csv"


def _kesh(period: str) -> dict:
    """Кэш s9 по ключу (код + входы + env + период), как scripts/revers/r1_a_zashch.py::kesh_s9. Не по указателю cache_latest_*:
    он смотрит на кэш, который собирали последним (после прогона NET — без дивидендов; Sharpe −0,8205 вместо −0,9146)."""
    with s9.period_ctx(period):
        p = s9.OUT / f"cache_{s9.cache_key()}.pkl"
    if not p.exists():
        pytest.skip(f"нет {p.name}: кэш s9 {period} с текущим ключом не собран")
    return pickle.loads(p.read_bytes())


def test_kesh_beretsya_po_klyuchu_a_ne_po_ukazatelyu(tmp_path, monkeypatch):
    """Мутация: указатель cache_latest_test1.txt смотрит на кэш с другими дивидендами — тест берёт кэш по ключу."""
    monkeypatch.setattr(s9, "OUT", tmp_path)
    monkeypatch.setattr(s9, "cache_key", lambda: "klyuch")
    for name, div in (("cache_klyuch.pkl", "tablica"), ("cache_chuzhoj.pkl", "NET")):
        (tmp_path / name).write_bytes(pickle.dumps({"div": div}))
    (tmp_path / "cache_latest_test1.txt").write_text("cache_chuzhoj.pkl")            # указатель — на чужой кэш
    assert _kesh("test1") == {"div": "tablica"}
    (tmp_path / "cache_klyuch.pkl").unlink()
    with pytest.raises(pytest.skip.Exception):                                         # нет кэша с ключом — пропуск, не чужой кэш
        _kesh("test1")


def _test1():
    if not (os.environ.get("FASTBT_Z2_R2FINAL") and os.environ.get("FASTBT_Z2_DAILY")):
        pytest.skip("нет среды Z2 (FASTBT_Z2_*)")
    return _kesh("test1")


def _ryad(variant, kesh):
    with s9.period_ctx("test1"):
        return Z.ryad_i_pozicii("test1", None, variant, kesh)


def test_test1_baza_ravna_stroke_a_cd():
    p, poz = _ryad("baza", _test1())
    o = pd.read_csv(A_CD, comment="#")
    o = o[o.portfel.str.startswith("A (s9") & (o.stroka == "(а) полные")].iloc[0]
    x = p["excess"].to_numpy(float)
    from fastbt import metriki_e as me
    assert len(x) == int(o.dni) == 744
    assert abs(me.sr_dnevnoy(x) * np.sqrt(252) - float(o.sharpe)) <= TOL
    assert abs(me.msr(x) - float(o.msr)) <= TOL
    assert len(poz) == 17 * 6 and all(sorted(v) == [2022, 2023, 2024] for v in poz.values())
    assert abs(float(o.sharpe)) > 0.5                                    # мутация «ряд обнулён/пуст» не пройдёт сравнение


def test_varianty_invarianty_i_ne_podmeneny_znak():
    K = _test1()
    b, l, s = (_ryad(v, K) for v in ("baza", "long", "short"))
    assert np.abs(l[0]["gross"] + s[0]["gross"] - b[0]["gross"]).max() <= TOL           # срез знака: лонг + шорт = всё (валовая доходность)
    assert l[0]["gross"].sum() != s[0]["gross"].sum() and (l[0]["baza"] == 0).all() and (s[0]["baza"] > 0).any()   # шорт — не лонг; лонг не платит заём
    assert (b[0]["baza"] > 0).any()
    assert np.array_equal(b[0].index.to_numpy(), l[0].index.to_numpy())


def test_fdr_t0_ravno_baze_t1_ne_vyshe():
    K = _test1()
    b, t0, t1 = (_ryad(v, K)[0]["excess"].to_numpy(float) for v in ("baza", "fdr_t0", "fdr_t1"))
    assert np.abs(t0 - b).max() <= TOL                        # s9: позиция держится ночью целиком, exposure = noch
    assert (t1 - b).max() <= TOL and (t1 - b).min() < -1e-9   # T+1 занимает капитал дольше: ставка на свободный — строго ниже хотя бы в день
    assert not np.allclose(t0, t1)                            # перепутанные t0/t1 не равны (мутация ловится)


def test_comm0_vyshe_i_div0_menyaet_tolko_tsenu():
    K = _test1()
    (b, poz_b), (c, _), (d, poz_d) = (_ryad(v, K) for v in ("baza", "comm0", "div0"))
    assert c["excess"].sum() > b["excess"].sum()
    assert all(np.array_equal(poz_b[k][y], poz_d[k][y]) for k in poz_b for y in poz_b[k])    # позиции те же
    dx = np.abs(d["excess"].to_numpy() - b["excess"].to_numpy())
    assert dx.max() > 1e-6 and (dx > 1e-12).mean() < 0.1       # различие — только в днях дивидендных гэпов
    assert d["excess"].sum() < b["excess"].sum()               # дивиденд добавляет доходность лонгам


def test_lr_bez_dividendov_sovpadaet_vne_gepov():
    K = _test1()
    with s9.period_ctx("test1"):
        t = "SBER"
        lr = Z.lr_bez_dividendov(t, K[t]["dates"])
    izm = ~np.isclose(lr, K[t]["lr"], rtol=0, atol=1e-12)
    assert 0 < izm.sum() <= 15 and (K[t]["lr"][izm] > lr[izm]).all()      # дивиденд прибавлен ровно в немногих днях гэпа, со знаком «плюс»


def test_div_ne_sovpal_so_sredoy_otkaz():
    with pytest.raises(SystemExit):
        Z.ryad_i_pozicii("test1", "NET" if not Z.zona.div_net() else "tablica", "baza", {})


def test_variant_neizvesten_i_srez_znaka():
    with pytest.raises(ValueError):
        Z.ryad_i_pozicii("test1", None, "xyz", {})
    p = np.array([0.5, -0.5, 0.0, 1.0])
    assert Z.pozicii_variant(p, "long").tolist() == [0.5, 0.0, 0.0, 1.0]
    assert Z.pozicii_variant(p, "short").tolist() == [0.0, -0.5, 0.0, 0.0]
    assert Z.pozicii_variant(p, "baza") is p


def _ansambl_train4(fiksirovat: bool):
    from test_yakor_ansambl_s9_train4 import _ansambl
    return _ansambl(fiksirovat)


def test_train4_pozicii_ravny_kesh_i_nezavisimomu_pereschetu():
    K = _kesh("train4")
    with s9.period_ctx("train4"):
        p, poz = Z.ryad_i_pozicii("train4", None, "baza", K)
    eta = K["AFLT"]["A"]["S3"]
    got = poz[("AFLT", "S3")]
    assert sorted(got) == [2016, 2017, 2018, 2019] and all(np.abs(got[y] - eta[y]).max() <= TOL for y in eta)   # = кэш s9
    ans = _ansambl_train4(False)                                                     # и = независимый ApproachA → turnover.apply
    assert max(float(np.abs(got[y] - ans[y]).max()) for y in eta) <= TOL
    mut = _ansambl_train4(True)                                                      # мутация A_fix: тест её отличает
    assert max(float(np.abs(got[y] - mut[y]).max()) for y in eta) > TOL
    assert len(p) > 900
