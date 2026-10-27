"""Э1-д (Ж-031, Ж-037): validation A′ — замок, «да», реестр, повтор, сбой; к2 и Δ. Синтетика, временный каталог.
Реальных данных не читает; `docs/otbor/validation_reestr.csv` основной копии не создаётся (пути — во `tmp`).
Детекторы `ok_*` возвращают True, если свойство выполнено; мутации — `test_sweep_val_mut.py`."""
import json

import numpy as np
import pandas as pd
import pytest

from fastbt import sweep_val as sv
from fastbt import sweep_val_stat as vs
from fastbt.sweep_a import SynthPanelSource, make_sweep
from fastbt.sweep_a_core import assemble
from fastbt.sweep_val_dry import NET, make_factory, make_meta, make_spec

REPS = dict(reps_lw=100, reps_k2=5)


def refused(tmp, spec_edit=None, meta_edit=None) -> bool:
    """Отказ «да»: Stop, и ни замка, ни реестра, ни каталога, ни чтения данных."""
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    reads = []
    if spec_edit:
        spec_edit(spec, P)
    try:
        sv.run(P, spec, meta_edit(meta) if meta_edit else meta, make_factory(tmp, lambda: reads.append(1)), **REPS)
    except sv.Stop:
        return bool(not reads and not P.lock.exists() and not P.reestr.exists() and not P.out.exists())
    return False


def ok_otkaz_bez_da(tmp):
    return refused(tmp, lambda s, P: s["da_vladeltsa"].update(stroka=""))


def ok_otkaz_sha_ne_tot(tmp):
    return refused(tmp, meta_edit=lambda m: m[:-1])           # список кандидатов изменён после «да»


def ok_otkaz_da_ranshe_zamorozki(tmp):
    def late(s, P):                                     # время заморозки — в файле, не в spec
        z = json.loads(P.zamorozka.read_text("utf-8"))
        P.zamorozka.write_text(json.dumps(dict(z, zamorozheno_v="2026-10-28 10:00")), "utf-8")
    return refused(tmp, late)


def ok_zamok_do_chteniya(tmp, monkeypatch) -> bool:
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    seen = []

    def rec(name):
        return lambda *a, **k: seen.append((name, P.lock.exists(), P.reestr.exists())) or None

    import pyarrow.parquet as pq
    for obj, n in ((pd, "read_parquet"), (pq, "read_table"), (pq, "read_metadata"), (np, "fromfile")):
        monkeypatch.setattr(obj, n, rec(n))

    def build():
        for f in (pd.read_parquet, pq.read_table, pq.read_metadata, np.fromfile):
            f("2020.bin")
    sv.run(P, spec, meta, make_factory(tmp, build), **REPS)
    return bool(len(seen) == 4 and all(lock and reestr for _, lock, reestr in seen))


def ok_reestr_pustoy_do_vzglyada_i_dve_stroki_posle(tmp) -> bool:
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    pre = not P.reestr.exists() and not P.out.exists()
    sv.run(P, spec, meta, make_factory(tmp), **REPS)
    rows = P.reestr.read_text("utf-8").splitlines()
    return bool(pre and len(rows) == 3 and rows[1].endswith("начато") and rows[2].endswith("просмотр состоялся")
                and sv.sha_kandidatov(meta) in rows[1] and "Ж-099" in rows[1] and P.out.exists())


def ok_sboy_udalyaet_i_pishet_preryvanie(tmp) -> bool:
    meta = make_meta()
    P, spec = make_spec(tmp, meta)

    def boom():
        P.out.mkdir(parents=True)
        (P.out / "chast.csv").write_text("x")
        raise OSError("сбой посередине")
    try:
        sv.run(P, spec, meta, make_factory(tmp, boom), **REPS)
    except RuntimeError as e:
        return bool("прервано до просмотра" in str(e) and not P.out.exists()
                    and P.reestr.read_text("utf-8").splitlines()[-1].endswith("прервано до просмотра"))
    return False


def ok_povtor_N_x2(tmp) -> bool:
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    n1 = sv.run(P, spec, meta, make_factory(tmp), **REPS)["rows"][0]["N"]
    try:
        sv.run(P, spec, meta, make_factory(tmp), **REPS)
        return False                                           # повтор без записанного изменения должен быть отказом
    except sv.Stop:
        pass
    P2, spec2 = make_spec(tmp, meta, note="заранее записанное изменение")
    n2 = sv.run(P2, spec2, meta, make_factory(tmp), **REPS)["rows"][0]["N"]
    try:
        sv.run(P2, spec2, meta, make_factory(tmp), **REPS)
        return False                                           # третьего взгляда нет
    except sv.Stop:
        return bool(n2 == 2 * n1)


def src():
    return SynthPanelSource(3, ("S1", "S2"), seed=4)


def ok_k2_lag0_ishodnyy_ryad() -> bool:
    s = src()
    pn = s.panel(vs.END)
    base = assemble(pn, NET, 0.0004, {})["excess"].values
    zero = assemble(vs.shift_panel(pn, {t: 0 for t in range(3)}), NET, 0.0004, {})["excess"].values
    moved = assemble(vs.shift_panel(pn, {t: 40 for t in range(3)}), NET, 0.0004, {})["excess"].values
    sh0 = vs.k2_sharpes(s, NET, 1, lags=[{t: 0 for t in range(3)}])[0]
    import fastbt.sweep_stat as ss
    from fastbt import metriki_e as me
    return bool(np.array_equal(base, zero) and not np.allclose(base, moved)
                and sh0 == float(me.sharpe(ss.year_slice(assemble(pn, NET, 0.0004, {}), vs.YEARS)["excess"].values)))


def ok_delta_raznost_tablichnyh(tmp) -> bool:
    meta = make_meta()
    c = make_factory(tmp)(meta, None)[3]
    for sw in (c.sw, c.orig_sw):
        sw.open_validation()
    r = vs.evaluate_candidate(c, 100, 5)
    return bool(r["delta"] == c.sw.evaluate(c.params, "validation", 0.04)["sharpe"]
                - c.orig_sw.evaluate(c.orig_params, "validation", 0.04)["sharpe"] and r["delta"] != 0)


def test_otkaz_bez_da(tmp_path):
    assert ok_otkaz_bez_da(tmp_path)


def test_otkaz_sha_spiska_ne_tot(tmp_path):
    assert ok_otkaz_sha_ne_tot(tmp_path)


def test_otkaz_da_ne_pozzhe_zamorozki(tmp_path):
    assert ok_otkaz_da_ranshe_zamorozki(tmp_path)


def test_zamok_do_chteniya_dannyh(tmp_path, monkeypatch):
    assert ok_zamok_do_chteniya(tmp_path, monkeypatch)


def test_reestr_ne_sozdayotsya_do_vzglyada(tmp_path):
    assert ok_reestr_pustoy_do_vzglyada_i_dve_stroki_posle(tmp_path)


def test_sboy_udalyaet_i_pishet_preryvanie(tmp_path):
    assert ok_sboy_udalyaet_i_pishet_preryvanie(tmp_path)


def test_povtor_otkaz_ili_N_x2_a_tretiy_net(tmp_path):
    assert ok_povtor_N_x2(tmp_path)


def test_k2_lag0_daet_ishodnyy_ryad_mut9():
    assert ok_k2_lag0_ishodnyy_ryad()


def test_delta_raznost_tablichnyh_sharpov_mut13(tmp_path):
    assert ok_delta_raznost_tablichnyh(tmp_path)


def test_vorota_plato_menshe_3_ne_prohodit(tmp_path):
    c = make_factory(tmp_path)(make_meta(), None)[0]
    c.neighbors = 2
    for sw in (c.sw, c.orig_sw):
        sw.open_validation()
    r = vs.evaluate_candidate(c, 100, 5)
    assert not r["ok_plato"] and not r["proshel"]


def test_reshenie_ne_proshel_ostayotsya_ishodnaya():
    rows = [dict(id="S1:1", proshel=False, delta=1.0), dict(id="S1:2", proshel=True, delta=0.2),
            dict(id="S1:3", proshel=True, delta=0.5), dict(id="A:1", proshel=False, delta=2.0)]
    assert sv.reshenie(rows) == {"S1": "S1:3", "A": "ishodnaya"}


def test_dry_cherez_main(capsys):
    assert sv.main(["--dry"]) == 0
    out = capsys.readouterr().out
    assert "S1:1" in out and "A:1" in out and "'S1':" in out


def test_realnyy_progon_bez_dry_otkaz():
    with pytest.raises(sv.Stop):
        sv.main([])
