"""Мутации PR 3 (validation A′, Ж-037): подложенная ошибка -> детектор из `test_sweep_val.py` обязан упасть."""
import re

from fastbt import sweep as sweep_mod
from fastbt import sweep_val as sv
from fastbt import sweep_val_stat as vs
from fastbt.s9_stats import lw_sharpe_diff
from test_sweep_val import (ok_delta_raznost_tablichnyh, ok_k2_lag0_ishodnyy_ryad, ok_otkaz_bez_da, ok_otkaz_sha_ne_tot,
                            ok_povtor_N_x2, ok_reestr_pustoy_do_vzglyada_i_dve_stroki_posle, ok_sboy_udalyaet_i_pishet_preryvanie,
                            ok_zamok_do_chteniya)


def test_mut_1_zamok_posle_chteniya(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "acquire_lock", lambda P, spec, sha: 1)       # замок не ставится до чтения
    assert not ok_zamok_do_chteniya(tmp_path, monkeypatch)


def test_mut_2_da_bez_sverki_sha(monkeypatch, tmp_path):
    real = sv.check_da
    own = lambda spec: re.search(r"[0-9a-f]{64}", spec["da_vladeltsa"]["stroka"]).group(0)      # sha берётся из самой строки «да»
    monkeypatch.setattr(sv, "check_da", lambda spec, sha, P: real(spec, own(spec), P))
    assert not ok_otkaz_sha_ne_tot(tmp_path)


def test_mut_3_da_ne_proveryaetsya_voobsche(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "check_da", lambda spec, sha, P: None)
    assert not ok_otkaz_bez_da(tmp_path)


def test_mut_4_reestr_pishetsya_do_da(monkeypatch, tmp_path):
    real = sv.check_da

    def early(spec, sha, P):
        sv.reestr(P, sha, "Ж-099", "начато")
        real(spec, sha, P)
    monkeypatch.setattr(sv, "check_da", early)
    assert not ok_otkaz_bez_da(tmp_path)


def test_mut_5_sboy_ne_udalyaet_fayly(monkeypatch, tmp_path):
    monkeypatch.setattr(sv.shutil, "rmtree", lambda *a, **k: None)
    assert not ok_sboy_udalyaet_i_pishet_preryvanie(tmp_path)


def test_mut_6_povtor_ne_udvaivaet_N(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep_mod.Sweep, "n_mult", lambda self: 1)
    assert not ok_povtor_N_x2(tmp_path)


def test_mut_7_k2_sdvig_lag0_ne_ishodnyy(monkeypatch):
    real = vs.shift_panel
    monkeypatch.setattr(vs, "shift_panel", lambda pn, lags: real(pn, {t: l + 1 for t, l in lags.items()}))
    assert not ok_k2_lag0_ishodnyy_ryad()


def test_mut_8_delta_iz_lw_a_ne_raznost_tablichnyh(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "delta_sharpe", lambda rc, ro: float(lw_sharpe_diff(rc["excess_d"], ro["excess_d"], 20, 5, 1)[0]))
    assert not ok_delta_raznost_tablichnyh(tmp_path)


def test_mut_9_reestr_pustoy_ne_pishet_vtoruyu_strokku(monkeypatch, tmp_path):
    real = sv.reestr
    monkeypatch.setattr(sv, "reestr", lambda P, sha, zh, status: None if status == "просмотр состоялся" else real(P, sha, zh, status))
    assert not ok_reestr_pustoy_do_vzglyada_i_dve_stroki_posle(tmp_path)
