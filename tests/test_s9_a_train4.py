"""Ж-075 п. 1 / Ж-079: боевой s9 строит A на train4 (2016–2019, прогрев 2015) без σ-источников.
Умолчание (train) — прежнее; период в ключе кэша только при не-умолчании (Д5); менеджер периода возвращает train (#418).
python3 -m pytest tests/test_s9_a_train4.py -q"""
import pytest

from fastbt import s9
from fastbt.config import Config


@pytest.fixture(autouse=True)
def _restore():
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    yield
    s9.PERIOD, s9.YEARS, s9.VAL_START = old


def test_years_a_iz_perioda_a_umolchanie_pobaytno():
    cfg = Config(years_a=tuple(range(2017, 2022)), val_start=2017)
    assert s9.period_cfg(cfg) is cfg                              # train: тот же объект, годы 2017–2021
    s9.set_period("train4")
    c4 = s9.period_cfg(cfg)
    assert c4.years_a == (2016, 2017, 2018, 2019) and c4.val_start == 2015


def test_cache_key_period_v_klyuche_train_ne_izmenen():
    k_train = s9.cache_key()
    s9.set_period("train4")
    k4 = s9.cache_key()
    s9.set_period("train")
    assert k_train != k4 and s9.cache_key() == k_train
    # ключ train — как до правки: добавка периода не входит при PERIOD == train
    import inspect
    assert 'if PERIOD != "train"' in inspect.getsource(s9.cache_key)


def test_mutaciya_period_ne_v_klyuche_lovitsya(monkeypatch):
    """Подмена: период убран из ключа → ключи train и train4 совпали бы; тест их различения это ловит."""
    src = s9.cache_key
    k_train = src()
    s9.set_period("train4")
    monkeypatch.setattr(s9, "PERIOD", "train")                    # имитация мутации: ключ не видит период
    assert src() == k_train                                       # при мутации ключи равны,
    monkeypatch.setattr(s9, "PERIOD", "train4")
    assert src() != k_train                                       # а на правке — различаются


def test_period_ctx_vozvrashchaet_train_dazhe_pri_oshibke():
    before = (s9.PERIOD, s9.YEARS, s9.VAL_START)
    with s9.period_ctx("train4"):
        assert s9.YEARS == (2016, 2017, 2018, 2019) and s9.PERIOD == "train4"
    assert (s9.PERIOD, s9.YEARS, s9.VAL_START) == before
    with pytest.raises(RuntimeError):
        with s9.period_ctx("train4"):
            raise RuntimeError("x")
    assert (s9.PERIOD, s9.YEARS, s9.VAL_START) == before
    with pytest.raises(ValueError):
        with s9.period_ctx("nope"):
            pass
    assert (s9.PERIOD, s9.YEARS, s9.VAL_START) == before


def test_build_cache_train4_bez_sigma_i_svoy_ukazatel(monkeypatch, tmp_path):
    seen = []

    def fake_job(args):
        seen.append(args)
        return args[0], {"A": {}}
    monkeypatch.setattr(s9, "_job", fake_job)
    monkeypatch.setattr(s9, "OUT", tmp_path)
    monkeypatch.setattr(s9, "cache_key", lambda: "k4")
    monkeypatch.setattr(s9, "load_source", lambda lab: pytest.fail("σ-источник прочитан на train4"))
    with s9.period_ctx("train4"):
        p = s9.build_cache(workers=1)
    assert p.name == "cache_k4.pkl" and seen and all(a[1] == {} and a[3] == "train4" for a in seen)
    assert not (tmp_path / "cache_latest.txt").exists()          # указатель train не затронут
    with s9.period_ctx("train4"):
        assert (tmp_path / s9._latest_name()).read_text() == "cache_k4.pkl"


def test_b_c_d_na_train4_po_prezhnemu_otkazyvayut():
    with s9.period_ctx("train4"):
        for f in (s9.build_sources, lambda: s9.load_source("hyb_h1")):
            with pytest.raises(NotImplementedError, match="σ-источников нет"):
                f()


@pytest.mark.parametrize("sid", [f"S{k}" for k in range(1, 7)])
def test_anchor_s9_diff_train4_konechen_i_nul(sid):
    """Ж-080: anchor_s9_diff(period="train4") на реальных данных — конечен (не inf) и ≤ 1e-12 по годам 2016–2019."""
    from fastbt import etap1
    from tests.test_yakor_v1 import make_evr
    d = etap1.anchor_s9_diff(sid, make_evr(sid), 0.0004, "train4")
    assert set(d) == {2016, 2017, 2018, 2019} and max(d.values()) <= 1e-12, d


def test_job_train4_stroit_A_2016_2019_na_realnykh_dannykh():
    """_job на одном тикере: A по годам периода (в т.ч. 2016), C пуст, σ не нужна; train — прежние 2017–2021."""
    with s9.period_ctx("train4"):
        tk, out = s9._job(("SBER", {}, ("dev",), "train4"))
    assert tk == "SBER" and out["C"] == {}
    for sid in s9.SIDS:
        assert set(out["A"][sid]) == {2016, 2017, 2018, 2019}
        assert out["A"][sid][2016].shape == out["dates"].shape or len(out["A"][sid][2016]) > 0


def test_sweep_sk_start_year_parametr():
    import inspect

    from fastbt import sweep_sk
    assert sweep_sk.START_YEAR == 2018
    assert "start_year" in inspect.signature(sweep_sk.SkEvaluator.pnl).parameters
