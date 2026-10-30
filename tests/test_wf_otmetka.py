"""Отметка снятого замка wf годна только как непустой JSON с preregistraciya_v2, sha256 (64 hex), data_snyatiya.
Одна функция `period_v2.otmetka_godna`; stdlib-копия в manifest_proverka.py согласована таблицей. python3 -m pytest tests/test_wf_otmetka.py -q"""
import importlib.util
import json
from pathlib import Path

import pytest

from fastbt import period_v2, zona

OK = {"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "ab" * 32, "data_snyatiya": "2026-10-30"}


def _mp():
    spec = importlib.util.spec_from_file_location("mp", Path(__file__).resolve().parents[1] / "scripts/server/manifest_proverka.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _f(tmp_path, text):
    p = tmp_path / "o.json"
    p.write_text(text)
    return p


SLUCHAI = [("", False), ("not json", False), ("[1]", False), ("{}", False), (json.dumps(OK), True),
           (json.dumps({k: v for k, v in OK.items() if k != "sha256"}), False),
           (json.dumps({k: v for k, v in OK.items() if k != "preregistraciya_v2"}), False),
           (json.dumps({k: v for k, v in OK.items() if k != "data_snyatiya"}), False),
           (json.dumps({**OK, "sha256": "ab" * 31}), False), (json.dumps({**OK, "sha256": "AB" * 32}), False),
           (json.dumps({**OK, "sha256": 5}), False)]


@pytest.mark.parametrize("text,ok", SLUCHAI)
def test_otmetka_godna_tablica(tmp_path, text, ok):
    p = _f(tmp_path, text)
    assert period_v2.otmetka_godna(p) is ok
    assert _mp().otmetka_godna(p) is ok            # stdlib-копия даёт то же


def test_net_fajla_otkaz(tmp_path):
    assert not period_v2.otmetka_godna(tmp_path / "net.json")


@pytest.mark.parametrize("text", ["", "not json", json.dumps({"sha256": "ab" * 32})])
def test_wf_s_negodnoj_otmetkoj_otkaz(tmp_path, monkeypatch, text):
    monkeypatch.setattr(period_v2, "OTMETKA", _f(tmp_path, text))
    with pytest.raises(period_v2.ZamokVladeltsa):
        zona.set_zone("wf")
    assert zona.name() == "dev"


def test_wf_s_godnoj_otmetkoj_prinyata(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "OTMETKA", _f(tmp_path, json.dumps(OK)))
    with zona.zone_ctx("wf"):
        assert zona.name() == "wf"
    assert zona.name() == "dev"


def test_granica_wf_manifest_trebuet_godnuyu(tmp_path):
    mp = _mp()
    with pytest.raises(mp.Otkaz):
        mp.granica_wf("2026-02-28", _f(tmp_path, "x"))
    assert mp.granica_wf("2026-02-28", _f(tmp_path, json.dumps(OK))) == "2026-02-28"


def test_mutaciya_exists_vmesto_proverki_lovitsya(tmp_path, monkeypatch):
    p = _f(tmp_path, "")
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    with pytest.raises(period_v2.ZamokVladeltsa):
        period_v2.proverit_chtenie("wf", otmetka=p)
    monkeypatch.setattr(period_v2, "otmetka_godna", lambda o: Path(o).exists())      # мутация: старое exists()
    period_v2.proverit_chtenie("wf", otmetka=p)                                     # пустой файл принят — тесты выше упали бы
