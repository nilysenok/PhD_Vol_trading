"""К-5: обвязка scripts/dannye/wf_sigma_910_progon.py (теневой корень). Синтетика в tmp, сеть и данные проекта не нужны.
Мутации: теневой корень указывает на старую базу — ловится сверкой хэшей входов; выход прогона ведёт в основную копию — ловится;
FASTBT_MAIN не задан / указывает не на теневой корень / на основную копию — отказ.
python3 -m pytest tests/test_sigma_wf_k5.py -q"""
import importlib.util
import os
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("wk5", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_910_progon.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def _tree(root, tag):
    r = root / "data" / "regen"
    for sub in ("D13", "vyr", "test_locked", "R2/cache", "vyr/vyr1", "test_locked/vyr", "R2/cache/VYR"):
        (r / sub).mkdir(parents=True, exist_ok=True)
    for e in W.NOVYE:
        (r / e).write_text(f"{tag}:{e}")
    (r / "D22_h1_grid.csv").write_text("g")
    (root / "data" / "calendar").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def kit(tmp_path):
    main, novye = (_tree(tmp_path / n, t) for n, t in (("main", "old"), ("novye", "new")))
    return main, novye, tmp_path / "shadow", tmp_path / "out_dir"


def _build(kit):
    main, novye, shadow, out = kit
    W.build_shadow(main, novye, shadow, out)
    return W.proverit_ten(main, novye, shadow, out)


def test_ten_vhody_novye_vyhody_v_out(kit):
    main, novye, shadow, out = kit
    assert _build(kit) == []
    r = shadow / "data" / "regen"
    for e in W.NOVYE:
        assert os.path.realpath(r / e) == str((novye / "data" / "regen" / e).resolve())
    assert (r / "D22_h1_grid.csv").is_symlink() and os.path.realpath(r / "D22_h1_grid.csv").startswith(str(main.resolve()))
    for rel, k in W.VYHODY.items():
        assert os.path.realpath(shadow / "data" / rel) == str((out / k).resolve())
    (shadow / "data" / "regen" / "vyr" / "vyr1" / "t.txt").write_text("zapis")
    assert (out / "vyr1" / "t.txt").exists() and not (main / "data" / "regen" / "vyr" / "vyr1" / "t.txt").exists()


def test_mutaciya_ten_na_staruyu_bazu_lovitsya(kit):
    main, novye, shadow, out = kit
    _build(kit)
    p = shadow / "data" / "regen" / "features_pit_v3_rrv.parquet"
    p.unlink()
    p.symlink_to(main / "data" / "regen" / "features_pit_v3_rrv.parquet")
    err = W.proverit_ten(main, novye, shadow, out)
    assert any("features_pit_v3_rrv" in e and "старая база" in e for e in err)
    assert any("features_pit_v3_rrv" in e and "хэш" in e for e in err)


def test_mutaciya_d13_na_staruyu_bazu_lovitsya(kit):
    main, novye, shadow, out = kit
    _build(kit)
    p = shadow / "data" / "regen" / "D13" / "intraday1850.parquet"
    p.unlink()
    p.symlink_to(main / "data" / "regen" / "D13" / "intraday1850.parquet")
    assert any("D13/intraday1850" in e for e in W.proverit_ten(main, novye, shadow, out))


def test_mutaciya_vyhod_v_osnovnoj_kopii_lovitsya(kit):
    main, novye, shadow, out = kit
    _build(kit)
    p = shadow / "data" / "regen" / "R2" / "cache" / "VYR"
    p.unlink()
    p.symlink_to(main / "data" / "regen" / "R2" / "cache" / "VYR")
    err = W.proverit_ten(main, novye, shadow, out)
    assert any("вне out" in e for e in err) and any("основной копии" in e for e in err)


def test_ten_vnutri_osnovnoj_kopii_zapreshchen(kit):
    main, novye, _, out = kit
    with pytest.raises(SystemExit):
        W.build_shadow(main, novye, main / "data" / "shadow", out)


def test_mjagkij_kontrol_ne_ostanavlivaet_no_ne_glotaet_drugoe():
    def stop(*a):
        raise SystemExit("контроль не пройден")

    def boom(*a):
        raise ValueError("не SystemExit")
    assert W._mjagkij(stop)(1) is None
    with pytest.raises(ValueError):
        W._mjagkij(boom)(1)


def test_mutaciya_staraya_s_novymi_matricami_ne_obkhoditsya(kit, tmp_path, monkeypatch):
    """--staraya разрешает старые входы только когда --novye = основная копия; иначе сверка хэшей остаётся в силе."""
    main, novye, shadow, out = kit
    monkeypatch.setenv("FASTBT_MAIN", str(shadow))
    W.build_shadow(main, novye, shadow, out)
    p = shadow / "data" / "regen" / "features_pit_v3.parquet"
    p.unlink()
    p.symlink_to(main / "data" / "regen" / "features_pit_v3.parquet")
    monkeypatch.setattr("sys.argv", ["x", "run", "--staraya", "--osnova", str(main), "--novye", str(novye), "--shadow", str(shadow), "--out", str(out)])
    with pytest.raises(SystemExit, match="старая база"):
        W.main()


def test_otkaz_bez_fastbt_main(kit, monkeypatch):
    main, _, shadow, _ = kit
    monkeypatch.delenv("FASTBT_MAIN", raising=False)
    with pytest.raises(SystemExit, match="FASTBT_MAIN не задан"):
        W.korenj_progona(shadow, main)


def test_otkaz_fastbt_main_ne_na_ten_ili_na_osnovnuyu(kit, monkeypatch):
    main, _, shadow, _ = kit
    monkeypatch.setenv("FASTBT_MAIN", str(main))
    with pytest.raises(SystemExit):
        W.korenj_progona(shadow, main)
    with pytest.raises(SystemExit):
        W.korenj_progona(main, main)
    monkeypatch.setenv("FASTBT_MAIN", str(shadow))
    assert W.korenj_progona(shadow, main) == shadow.resolve()
