"""Р-4 / Ф-ранг: реальные данные train4 (2016–2019), 50 точек Соболя S3 и S4 с рангом (m_sily = kvadr, rang_tip = obshchiy).
(а) быстрый слой = эталон побайтно; (б) доля точек с ненулевым excess до/после правки (до = rang_starta → тождество).
python3 -m pytest tests/test_rang_trenda_real.py -q -s"""
import os

import numpy as np
import pytest

from fastbt import eval_v5
from test_stop_nach_trail_net import real

_S: dict = {}
IZM = dict(m_sily="kvadr", rang_tip="obshchiy")


def _sreda(sid):
    if sid not in _S:
        from fastbt import canon_v5, progon_sobol as ps
        rz = ps.Rezhim("v5", "train4", "seed", False, canon_v5.load_kub(sid)["fdr"])
        _S[sid] = (ps, ps.fabrika_real(sid, rz), rz, dict(ps.tochki_v5(ps.load_kub(sid, rz), 0, 70, rz)))
    return _S[sid]


def _ryad(sid, i, fast):
    ps, sr, rz, P = _sreda(sid)
    row, x, _ = ps.schitat_tochku(sr, sid, i, {**P[i], **IZM}, "r4", False, fast, rz=rz)
    return row, (np.asarray(x) if row["status"] == "ok" else None)


def _tochki(sid):
    return sorted(_sreda(sid)[3])[:50]


@real
@pytest.mark.skipif(not os.environ.get("FASTBT_RANG_REAL"), reason="~3 мин на данных train4: задать FASTBT_RANG_REAL=1")
@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_bystryj_sloj_raven_etalonu_i_dolya_nenulevykh(sid, monkeypatch):
    ok, nz, nz_do, ravno = 0, 0, 0, 0
    for i in _tochki(sid):
        rs, xs = _ryad(sid, i, False)
        rf, xf = _ryad(sid, i, True)
        assert rs["status"] == rf["status"]
        if xs is None:
            continue
        ok += 1
        ravno += int(np.array_equal(xs, xf))
        nz += int(np.any(xs != 0))
    with monkeypatch.context() as m:
        m.setattr(eval_v5, "rang_starta", lambda sig, rg: rg)           # «как было»
        for i in _tochki(sid):
            _, x = _ryad(sid, i, False)
            nz_do += int(x is not None and np.any(x != 0))
    print(f"\nRES {sid}: точек ok {ok}; быстрый = эталон {ravno}; ненулевой excess до {nz_do}, после {nz}")
    assert ok > 0 and ravno == ok and nz >= nz_do
