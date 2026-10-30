"""К-1 / К-4: мутации, не пойманные tests/test_sigma_wf_k1.py (тестировщик PR #797).

Не пойманы были: RVI10 и CANDLES в l2_lib смотрят мимо FASTBT_MAIN; константа GRANICA сдвинута в 2024;
умолчание --konec ≥ 2026-02-04; main() строит свечи с пустым набором ДСВД (ночь от воскресенья)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/dannye"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import wf_baza_bez_dsvd as K  # noqa: E402


@pytest.mark.parametrize("attr", ["RVI10", "CANDLES", "DATA"])
def test_l2_lib_vse_puti_ot_peremennoj(attr, tmp_path):
    kod = "import sys; sys.path.insert(0, %r); import l2_lib as m; print(m.%s)" % (str(ROOT / "scripts/analysis"), attr)
    env = {**os.environ, "FASTBT_MAIN": str(tmp_path), "NUMBA_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, "-c", kod], env=env, capture_output=True, text=True, cwd=tmp_path)
    if r.returncode and "ModuleNotFoundError" in r.stderr:
        pytest.skip(r.stderr.strip().splitlines()[-1])
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip().splitlines()[-1].startswith(str(tmp_path))


def test_granitsa_pereschyota_1_yanvarya_2025():
    assert K.GRANICA == pd.Timestamp("2025-01-01")  # мутация 2024: пересчёт заходит в строки, которые обязаны быть прежними


def test_main_provody_umolchaniya_konec_i_dsvd(tmp_path, monkeypatch):
    """main(): умолчание конца ряда 2026-02-03; в калькулятор свечей уходят даты ДСВД из календаря (не пустой набор)."""
    seen = {}
    base = pd.DataFrame({"date": [pd.Timestamp("2025-03-14")], "ticker": ["AA"]})

    class Lib:
        OWN = ["rv"]

    monkeypatch.setattr(K, "MAIN", tmp_path)
    monkeypatch.setattr(K.pd, "read_parquet", lambda *a, **k: base)
    monkeypatch.setattr(K, "_lib", lambda: Lib)
    monkeypatch.setattr(K, "svechi_bez_dsvd", lambda d, dsvd: seen.update(dsvd=dsvd))
    monkeypatch.setattr(K, "own_bez_dsvd", lambda calc, tk: None)
    monkeypatch.setattr(K, "sobrat", lambda b, o, dsvd, konec, cols: seen.update(konec=konec, dsvd2=dsvd) or base)
    monkeypatch.setattr(K.shutil, "copy", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "o")])
    K.main()
    assert seen["konec"] == pd.Timestamp("2026-02-03")
    assert seen["dsvd"] == K.daty_dsvd() and len(seen["dsvd"]) == 76 and seen["dsvd2"] == seen["dsvd"]
