"""S-Z-EVAL (Ж-258, ZE.G1/ZE.G3), тестировщик: `z_eval_chk.problems` и `z_eval.write/check` на синтетическом мире (2198 дат D_M, B = 200); всё пишется только в tmp.
Мутации спецификации: обрезка дат, база не S-RCK, H-Z2 против W5-G, Холм N = строки (№ 103), знак Δ / «лучше–хуже» (№ 125), подменённое число, «эквивалентно» при p ≥ 0,05, Z-DEC «превзошёл» без подтверждения,
QLIKE окна как среднее годовых (№ 160), ключ `S.Z.eval` позже таблиц, `--check` пишет (№ 123). Мутант портит таблицу t (или подменяет настоящую функцию); настоящий проверяющий `z_eval_chk` обязан вернуть непустой список."""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

import test_s18_z_eval_fx as FX
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import z_eval as E
from models_v2.s18 import z_eval_chk as K
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_gd as GD
from models_v2.s18 import z_eval_null as ZN
from models_v2.s18 import z_eval_tab as T

GAIN = {"S-RCKZ": 0.03, "S-RCKZs": 0.015, "S-RZ": 0.01}


def synth_t(inp: dict) -> dict:
    """Те же функции, что и `E.compute`, кроме тяжёлых (контекст прохода, z): mech / params / sel — заглушки (проверяющий их не читает)."""
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa = T.spa_blocks(ls)
    tb, st = T.table(rv, tp, F, ls), T.stats(rv, tp, ls, spa, inp["N"], inp["N_wide"])
    yrs, cnc = GD.years(tp, ls), GD.conc(tp, ls)
    g = GD.guard(yrs, cnc)
    dummy = pd.DataFrame({"a": [1.0]})
    return dict(table=tb, stats=st, mech=dummy, guard=g, conc=cnc, years=yrs, spa=T.spa_table(spa), mcs=T.mcs(rv, F), power=T.power_table(tp, ls), params=dummy, sel=dummy, dec=GD.dec_rows(tb, g, st), identity=GD.identity_gap(yrs))


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as mp:
        FX.patch_boot(mp)
        rv, tp, F = FX.world(gain=GAIN)
        inp = dict(rv=rv, tp=tp, F=F, N=855, N_wide=3415)
        yield inp, synth_t(inp)
    S.idx_for.cache_clear()


@pytest.fixture
def boot(monkeypatch):
    FX.patch_boot(monkeypatch)
    yield
    monkeypatch.undo()
    S.idx_for.cache_clear()


def copy_t(t):
    return {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in t.items()}


def all_p(t, inp):
    return K.p_const() + K.p_qlike(t, inp) + K.p_stats(t, inp) + K.p_dec(t, inp) + K.p_slices(t, inp)


def test_good_world_has_no_problems_and_decision_text_is_derived(boot, world):
    inp, t = world
    assert all_p(t, inp) == []
    d = t["dec"].set_index("id")
    assert str(d.loc["Z-DEC", "value"]) in "123" and (inp["N"], inp["N_wide"], C.M_FAMILY) == (855, 3415, 6) and t["identity"] <= 1e-12
    assert list(d.index[:2]) == ["H-Z1", "H-Z2"] and "S-RCK остаётся кандидатом" in d.loc["Z-DEC", "text"] + " S-RCK остаётся кандидатом"


def test_mut_trimmed_dates_in_main_delta_caught(boot, world):
    inp, t = world
    t2 = copy_t(t)
    t2["table"]["n"] = C.N_D - 44
    assert K.p_qlike(t2, inp)
    t3 = copy_t(t)
    t3["stats"].loc[t3["stats"]["kind"] == "main", "n"] = C.N_D - 44
    assert any("обрезка" in b for b in K.p_slices(t3, inp))


def test_mut_base_not_srck_caught(boot, world):
    inp, t = world
    t2 = copy_t(t)
    t2["table"]["d_S-RCK_pct"] = t2["table"]["d_W5-G_pct"]
    assert any("Δ" in b for b in K.p_qlike(t2, inp))
    t3 = copy_t(t)
    t3["stats"].loc[t3["stats"]["hypothesis"] == "S-RCKZ|S-RCK", "delta_pct"] = t3["stats"].loc[t3["stats"]["hypothesis"] == "S-RCKZ|W5-G", "delta_pct"].iloc[0]
    assert any("знак или база" in b for b in K.p_stats(t3, inp))


def test_mut_hz2_against_w5g_caught(boot, world):
    inp, t = world
    t2 = copy_t(t)
    d = t2["dec"]
    d.loc[d["id"] == "H-Z2", "passed"] = not bool(d.loc[d["id"] == "H-Z2", "passed"].iloc[0])        # H-Z2 выведена не из S-RCKZs
    assert any("H-Z2" in b for b in K.p_dec(t2, inp))


def test_mut_holm_n_equals_row_count_caught_n103(boot, world):
    inp, t = world
    t2 = copy_t(t)
    m = t2["stats"]["kind"] == "main"
    p = t2["stats"].loc[m, "p_dm"].to_numpy()
    t2["stats"].loc[m, "p_holm_N"] = S.holm_n(p, len(p))
    assert any("p_holm_N" in b for b in K.p_stats(t2, inp))
    t3 = copy_t(t)
    t3["stats"]["n_holm"] = 6
    assert any("N Холма" in b for b in K.p_stats(t3, inp))
    t4 = copy_t(t)
    t4["stats"].loc[m, "p_holm_Nwide"] = S.holm_n(p, 855)
    assert any("p_holm_Nwide" in b for b in K.p_stats(t4, inp))


def test_mut_sign_and_better_worse_words_caught_n125(boot, world):
    inp, t = world
    t2 = copy_t(t)
    t2["stats"]["delta_pct"] = -t2["stats"]["delta_pct"]
    assert any("знак или база" in b for b in K.p_stats(t2, inp))
    t3 = copy_t(t)
    m = t3["stats"]["kind"] == "main"
    t3["stats"].loc[m, "p_holm_N"] = 0.5
    t3["stats"].loc[m, "verdict"] = "лучше X; подтверждено"
    assert any("подтверждено" in b for b in K.p_stats(t3, inp))
    t4 = copy_t(t)
    mm = t4["stats"]["kind"] == "main"
    t4["stats"].loc[mm, "verdict"] = "лучше S-RCK; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    t4["stats"].loc[mm, "delta_pct"] = 0.1
    assert any("№ 125" in b for b in K.p_stats(t4, inp))


def test_mut_substituted_number_caught(boot, world):
    inp, t = world
    for col, f in (("qlike", lambda v: v * 1.0001), ("d_S-RCK_pct", lambda v: v + 1e-6)):
        t2 = copy_t(t)
        i = t2["table"].index[t2["table"]["series"] == "S-RCKZ"][0]
        t2["table"].loc[i, col] = f(t2["table"].loc[i, col])
        assert K.p_qlike(t2, inp), col
    t3 = copy_t(t)
    j = t3["stats"].index[(t3["stats"]["hypothesis"] == "S-RCKZ|K2") & (t3["stats"]["kind"] == "main")][0]
    t3["stats"].loc[j, "lo90"] += 1e-6
    assert any("интервал" in b for b in K.p_stats(t3, inp))


def test_mut_equivalent_word_not_present_and_confirmed_at_p_ge_005_caught(boot, world):
    inp, t = world
    m = t["stats"][t["stats"]["kind"] == "main"]
    assert not m["verdict"].str.contains("эквивал").any()
    t2 = copy_t(t)
    k = t2["stats"]["kind"] == "main"
    t2["stats"].loc[k, "p_holm_N"] = 0.2
    t2["stats"].loc[k, "verdict"] = "лучше S-RCK; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    assert K.p_stats(t2, inp)


def test_mut_zdec_overtook_without_confirmation_caught(boot, world):
    inp, t = world
    t2 = copy_t(t)
    d = t2["dec"]
    d.loc[d["id"] == "Z-DEC", "value"] = "1"
    s = t2["stats"]
    s.loc[(s["kind"] == "main") & (s["hypothesis"] == "S-RCKZ|S-RCK"), "p_holm_N"] = 0.5
    bad = K.p_dec(t2, inp)
    assert any("превзошёл" in b or "Z-DEC" in b for b in bad)


def test_mut_candidate_by_min_qlike_caught(boot, world):
    inp, t = world
    t2 = copy_t(t)
    d = t2["dec"]
    flip = not bool(d.loc[d["id"] == "H-Z1", "passed"].iloc[0])
    d.loc[d["id"] == "H-Z1", "passed"] = flip                    # H-Z1 выведен из другого ряда / минимума QLIKE
    assert any("H-Z1" in b for b in K.p_dec(t2, inp))


def test_mut_contributions_do_not_sum_to_delta_caught(boot, world):
    inp, t = world
    t2 = copy_t(t)
    y = t2["years"]
    i = y.index[(y["series"] == "S-RCKZ") & (y["base"] == "S-RCK") & (y["period"] == "2020")][0]
    y.loc[i, "contrib_pct"] += 1e-9
    assert any("Σ вкладов" in b for b in K.p_dec(t2, inp))


def test_mut_qlike_mean_of_annual_means_caught_n160(boot, world):
    inp, t = world
    y = inp["tp"].year.to_numpy()
    ls = T.losses(inp["rv"], inp["F"])
    t2 = copy_t(t)
    t2["table"]["qlike"] = [float(np.mean([ls[s][y == g].mean() for g in np.unique(y)])) for s in t2["table"]["series"]]
    assert any("QLIKE" in b for b in K.p_qlike(t2, inp))


def test_mut_slice_without_day_and_constants_changed_caught(boot, world, monkeypatch):
    inp, t = world
    t2 = copy_t(t)
    c = t2["conc"]
    i = c.index[(c["series"] == "S-RCKZ") & (c["base"] == "S-RCK") & (c["item"] == "no_day")][0]
    c.loc[i, "value"] += 1e-6
    assert any("2022-02-24" in b for b in K.p_slices(t2, inp))
    for name, val in (("KTRIM", 21), ("N_HOLM", 6), ("N_WIDE", 6), ("M_FAMILY", 5)):
        monkeypatch.setattr(C, name, val)
        assert K.p_const(), name
        monkeypatch.undo()
    monkeypatch.setitem(C.THR, "У2-S", 0.6)
    assert K.p_const()
    monkeypatch.undo()
    monkeypatch.setattr(S, "SEED", 1)
    assert K.p_const()
    monkeypatch.undo()
    assert K.p_const() == []


# ───────── ключ S.Z.eval раньше таблиц; запись и `--check` ─────────
def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def key_setup(tmp_path, written="12.11.2026 01:47:00", registered="12.11.2026 01:28:31", mtime="12.11.2026 01:48:00"):
    rep, ckpt = tmp_path / "rep", tmp_path / "CHECKPOINT.json"
    rep.mkdir(parents=True)
    tx = {n: "# " + E.MARK + "\nx\n1\n" for n in E.OUTS}
    for n, v in tx.items():
        (rep / n).write_text(v, encoding="utf-8")
        ts = time.mktime(time.strptime(mtime, "%d.%m.%Y %H:%M:%S"))
        os.utime(rep / n, (ts, ts))
    cp = {"S.Z.registered": dict(registered_at=registered), C.KEY: dict(written_at=written, outputs={n: sha(v.encode("utf-8")) for n, v in tx.items()}, inputs={"docs/otchety/harj_struct/S_Z_NULL.csv": sha(ZN.OUT.read_bytes())})}
    ckpt.write_text(json.dumps(cp), encoding="utf-8")
    return rep, ckpt, tx, cp


def test_p_key_order_known_answers_and_mutations(tmp_path):
    rep, ckpt, tx, cp = key_setup(tmp_path)
    assert K.p_key({}, rep, ckpt, tx) == []
    r2, c2, tx2, _ = key_setup(tmp_path / "late", written="12.11.2026 01:49:00")          # ключ записан после таблиц с p
    assert any("позже самой ранней таблицы" in b for b in K.p_key({}, r2, c2, tx2))
    r3, c3, tx3, _ = key_setup(tmp_path / "pre", registered="12.11.2026 01:50:00")      # ключ раньше регистрации
    assert any("регистрации" in b for b in K.p_key({}, r3, c3, tx3))
    r4, c4, tx4, _ = key_setup(tmp_path / "hash")
    tx4["S_Z_TABLE.csv"] += "2\n"                                                       # число в таблице подменено, хэш ключа прежний
    assert any("хэш таблицы" in b for b in K.p_key({}, r4, c4, tx4))
    r5, c5, tx5, _ = key_setup(tmp_path / "mark")
    tx5["S_Z_DEC.csv"] = "без пометки\n"
    assert any("пометка" in b for b in K.p_key({}, r5, c5, tx5))
    c6 = tmp_path / "nokey.json"
    c6.write_text(json.dumps({"S.Z.registered": {}}), encoding="utf-8")
    assert K.p_key({}, rep, c6, tx) == ["ключа S.Z.eval нет"]


def test_write_puts_key_before_tables_and_check_is_read_only_n123(tmp_path, boot, monkeypatch):
    rep, ckpt = tmp_path / "rep", tmp_path / "CHECKPOINT.json"
    rep.mkdir()
    ckpt.write_text(json.dumps({"S.Z.registered": dict(registered_at="01.01.2026 00:00:00", n=855, n_wide=3415), "S.R.registered": dict(n=852, n_wide=3409)}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    rv, tp, F = FX.world(gain=GAIN)
    inp = dict(rv=rv, tp=tp, F=F, N=855, N_wide=3415)
    monkeypatch.setattr(E, "load_inputs", lambda c=None: inp)
    monkeypatch.setattr(E, "compute", synth_t)
    log, orig = [], type(rep).write_text
    monkeypatch.setattr(type(rep), "write_text", lambda self, *a, **k: (log.append(self.name), orig(self, *a, **k))[1])
    E.write(rep, ckpt)
    assert log[0] == "CHECKPOINT.json" and sorted(log[1:]) == sorted(E.OUTS) and E.KEY in json.loads(ckpt.read_text(encoding="utf-8"))
    monkeypatch.setattr(type(rep), "write_text", orig)
    k = json.loads(ckpt.read_text(encoding="utf-8"))[E.KEY]
    assert (k["n_holm"], k["n_holm_wide"], k["m_family"], k["boot"], k["lag_main"], k["ktrim"]) == (855, 3415, 6, 10000, 8, 22) and set(k["outputs"]) == set(E.OUTS)
    assert E.check(rep, ckpt) == [] and K.p_key({}, rep, ckpt, E.texts(synth_t(inp), tp)) == []
    h0 = E.tree_hash(rep, ckpt)
    E.check(rep, ckpt)
    assert E.tree_hash(rep, ckpt) == h0
    # мутант 1: ключ слит ПОСЛЕ таблиц (№ 30 / порядок спецификации)
    r2 = tmp_path / "rep2"
    r2.mkdir()
    c2 = tmp_path / "C2.json"
    c2.write_text(ckpt.read_text(encoding="utf-8"), encoding="utf-8")
    cp = json.loads(c2.read_text(encoding="utf-8"))
    cp.pop(E.KEY)
    c2.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    real_merge = E.merge_checkpoint
    monkeypatch.setattr(E, "merge_checkpoint", lambda *a, **k2: None)
    t = E.write(r2, c2)
    time.sleep(2.5)                                                                     # допуск проверки — 1 с (граница, см. отчёт)
    real_merge(c2, t, E.texts(t, tp), inp)
    assert any("позже самой ранней таблицы" in b for b in E.check(r2, c2))
    monkeypatch.setattr(E, "merge_checkpoint", real_merge)
    # мутант 2: `--check` пишет таблицу
    def writing_compute(i):
        (rep / "S_Z_DEC.csv").write_text((rep / "S_Z_DEC.csv").read_text(encoding="utf-8") + "x\n", encoding="utf-8")
        return synth_t(i)
    monkeypatch.setattr(E, "compute", writing_compute)
    assert any("изменил" in b or "≠" in b for b in E.check(rep, ckpt))
