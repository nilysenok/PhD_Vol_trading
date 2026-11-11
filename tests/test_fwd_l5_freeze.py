"""Л5: заморозка (fwd_freeze, fwd_frz_parts). Модуль вне `code_sha256` ловится; FREEZE.json не перезаписывается; n_new = длина календаря; строка «да» есть в журнале;
`FWD.freeze` вливается без потери чужих ключей и не пишется дважды. Мутации: файл замыкания выброшен / sha изменён / нет «да» / репетиция не «0 расхождений» / день календаря выброшен.
Цен новых файлов не читает. Сети нет."""
from __future__ import annotations

import hashlib
import json
import sys
import types

import pandas as pd
import pytest

import torch  # noqa: F401  (первым: общий OpenMP)
from models_v2.s17 import fwd_cal as FC, fwd_freeze as FZ, fwd_frz_parts as P, fwd_power as PW, fwd_src as S

HAVE = FZ.OUT.exists()


@pytest.fixture(scope="module")
def loaded() -> dict:
    P.import_chain()
    return P.loaded_files()


def test_zamykanie_polno_i_modul_vne_code_sha256_lovitsya(loaded):
    assert "models_v2/s17/fwd_rehearsal.py" in loaded and "models_v2/s17/fwd_k23.py" in loaded and "models_v2/s14/v7_3_cfg.py" in loaded
    assert P.closure_problems(dict(loaded), loaded) == {"not_in_freeze": [], "sha_differs": []}
    cut = {k: v for k, v in loaded.items() if k != "models_v2/s17/fwd_cfg8.py"}                    # мутация: файл замыкания выброшен
    assert P.closure_problems(cut, loaded)["not_in_freeze"] == ["models_v2/s17/fwd_cfg8.py"]
    mut = {**loaded, "models_v2/s17/fwd_x1.py": "0" * 64}                                          # мутация: sha изменён
    assert P.closure_problems(mut, loaded)["sha_differs"] == ["models_v2/s17/fwd_x1.py"]


def test_novyj_modul_v_sys_modules_popadaet_v_zamykanie(loaded, monkeypatch):
    m = types.ModuleType("models_v2.fwd_l5_fake")
    m.__file__ = str(S.ROOT / "models_v2/metrics.py")                                              # существующий файл models_v2 вне замораживаемого списка
    monkeypatch.setitem(sys.modules, "models_v2.fwd_l5_fake", m)
    frozen = {k: v for k, v in loaded.items() if k != "models_v2/metrics.py"}
    assert "models_v2/metrics.py" in P.closure_problems(frozen, P.loaded_files())["not_in_freeze"]


def test_freeze_ne_perezapisyvaetsya(tmp_path, monkeypatch):
    f = tmp_path / "FREEZE.json"
    f.write_text('{"v": 0}', encoding="utf-8")
    before = hashlib.sha256(f.read_bytes()).hexdigest()
    with pytest.raises(FZ.Refusal, match="не перезаписывается"):
        FZ.preconditions(f)
    with pytest.raises(FZ.Refusal):
        FZ.write(path=f)
    assert hashlib.sha256(f.read_bytes()).hexdigest() == before
    if HAVE:
        with pytest.raises(FZ.Refusal, match="не перезаписывается"):
            FZ.preconditions()


def test_otkaz_bez_da_vladeltsa_i_bez_nulya_rashozhdenij(tmp_path, monkeypatch):
    j = tmp_path / "resheniya.md"
    j.write_text("Ж-248 без разрешения", encoding="utf-8")
    monkeypatch.setattr(FZ, "JOURNAL", j)
    assert not FZ.owner_in_journal(FZ.OWNER, j)
    with pytest.raises(FZ.Refusal, match="«да» владельца"):
        FZ.preconditions(tmp_path / "new.json")
    monkeypatch.undo()
    assert FZ.owner_in_journal() and FZ.OWNER == {**FZ.OWNER, "zh": "Ж-248", "date": "2026-11-11", "punkt": 4}
    rh = tmp_path / "REH.csv"
    monkeypatch.setattr(FZ.RH, "OUT", rh)
    with pytest.raises(FZ.Refusal, match="не записан"):                                           # репетиции нет
        FZ.preconditions(tmp_path / "new2.json")
    rh.write_text("d,node\n", encoding="utf-8")
    monkeypatch.setattr(FZ.RH, "problems", lambda df: ["расхождение"])                            # репетиция не «0 расхождений»
    with pytest.raises(FZ.Refusal, match="репетиция"):
        FZ.preconditions(tmp_path / "new3.json")


def test_n_new_ravno_dline_kalendarya():
    cal = pd.read_parquet(FC.OUT)
    pairs = FC.new_pairs(cal)
    assert len(pairs) == PW.n_new_of_calendar() == 180 and pairs["t_plus"].iloc[-1] == S.LAST_SESSION and pairs["date"].iloc[0] == S.ARCHIVE_END
    cut = cal[cal["date"] != pd.Timestamp("2026-03-10")].reset_index(drop=True)                    # мутация: день календаря выброшен
    cut["t_plus"] = cut["date"].shift(-1)
    assert len(FC.new_pairs(cut.assign(new_pair=cut["t_plus"].notna() & (cut["t_plus"] > S.ARCHIVE_END)))) == 179


@pytest.mark.skipif(not HAVE, reason="FREEZE.json ещё не записан")
def test_freeze_json_razdely_i_n_new():
    z = json.loads(FZ.OUT.read_text(encoding="utf-8"))
    need = {"v", "written_at", "git", "code_sha256", "packages", "inputs_archive_sha256", "inputs_new_sha256", "params", "initial_state", "calendar", "dates", "candidates", "controls", "family", "outputs", "rehearsal", "power", "owner_yes"}
    assert need <= set(z) and z["v"] == 1 and z["candidates"] == ["X1", "ZB"] and z["controls"] == ["K1", "K2", "K3"] and z["owner_yes"] == FZ.OWNER
    c = z["calendar"]
    assert c["n_new"] == len(c["pairs"]) == len(PW.pd.read_parquet(FC.OUT).pipe(FC.new_pairs)) == z["power"]["n_new"] == 180
    assert z["dates"]["last_session"] == "2026-11-10" and z["dates"]["last_t_plus"] == "2026-11-10" and z["family"]["holm_m"] == 7
    assert z["rehearsal"]["result"] == "0 расхождений" and z["rehearsal"]["sha256"] == hashlib.sha256(FZ.RH.OUT.read_bytes()).hexdigest()
    assert z["power"]["sha256"] == hashlib.sha256(PW.OUT.read_bytes()).hexdigest()
    assert c["refit_cfg8"] == ["2026-03-02", "2026-04-01", "2026-05-04", "2026-06-01", "2026-07-01", "2026-08-03", "2026-09-01", "2026-10-01", "2026-11-02"]
    assert P.sha(S.ROOT / "docs/otchety/harj_x1/ZE_CRIT.csv") == z["not_admitted"]["ZE"]["sha256"]


def test_merge_cp_sohranyaet_chuzhie_klyuchi_i_ne_pishet_dvazhdy(tmp_path, monkeypatch):
    cp = tmp_path / "CHECKPOINT.json"
    cp.write_text(json.dumps({"P0": {"x": 1}, "FWD": {"other": 2}}), encoding="utf-8")
    out = tmp_path / "FREEZE.json"
    out.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(FZ, "OUT", out)
    frz = dict(v=1, written_at="t", calendar=dict(n_new=180), rehearsal=dict(sha256="a"), power=dict(sha256="b"))
    FZ.merge_cp(frz, cp)
    d = json.loads(cp.read_text(encoding="utf-8"))
    assert d["P0"] == {"x": 1} and d["FWD"]["other"] == 2 and d["FWD"]["freeze"]["n_new"] == 180
    with pytest.raises(FZ.Refusal):
        FZ.merge_cp(frz, cp)


@pytest.mark.skipif(not HAVE, reason="FREEZE.json ещё не записан")
def test_zamykanie_realnoj_zamorozki_polno(loaded):
    z = json.loads(FZ.OUT.read_text(encoding="utf-8"))
    p = P.closure_problems(z["code_sha256"], loaded)
    assert p == {"not_in_freeze": [], "sha_differs": []}, p
