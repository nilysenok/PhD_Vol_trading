"""wf_sigma_cd.py: --out внутри test_locked/ — отказ до чтения входов; рядом — нормально. Синтетика, сеть не нужна.
Мутация «проверка снята» (proverit_out — пустая) ловится: запись в test_locked/ проходит, тест падает.
python3 -m pytest tests/test_wf_sigma_cd_out.py -q"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("wsc_out", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_cd.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)
_tspec = importlib.util.spec_from_file_location("twsc_out", Path(__file__).resolve().parent / "test_wf_sigma_cd.py")
T = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(T)


@pytest.fixture
def bez_zamka(monkeypatch):
    monkeypatch.setattr(W.period_v2, "proverit_chtenie", lambda period, put=None, **k: None)


def _nelzya_chitat(monkeypatch):
    def nelzya(*a, **k):
        raise AssertionError("входы прочитаны до проверки --out")
    monkeypatch.setattr(W.pd, "read_parquet", nelzya)


def test_out_v_test_locked_otkaz_do_chteniya(tmp_path, monkeypatch, bez_zamka):
    main, spec, _ = T.make_main(tmp_path)
    out = main / "data/regen/test_locked/wf_sigma"
    _nelzya_chitat(monkeypatch)
    with pytest.raises(SystemExit, match="test_locked"):
        W.run(main, out, spec)
    assert not out.exists()


def test_out_cherez_simlink_v_test_locked_otkaz(tmp_path, monkeypatch, bez_zamka):
    """Путь без «test_locked» в тексте, но симлинк ведёт внутрь test_locked/ — проверяется разрешённый путь."""
    main, spec, _ = T.make_main(tmp_path)
    tl = main / "data/regen/test_locked"
    tl.mkdir(parents=True, exist_ok=True)
    (tmp_path / "vyhod").symlink_to(tl)
    _nelzya_chitat(monkeypatch)
    with pytest.raises(SystemExit, match="test_locked"):
        W.run(main, tmp_path / "vyhod" / "wf_sigma", spec)


def test_out_ryadom_normalno(tmp_path, bez_zamka):
    main, spec, _ = T.make_main(tmp_path)
    out = main / "data/regen/wf/ch4_sigma"
    W.run(main, out, spec)
    assert (out / "sigma_hyb_h5_dev.parquet").exists() and (out / "wf_sigma_run.json").exists()
