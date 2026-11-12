"""S-RCD-EVAL (Ж-259, RE.G1/RE.G3), тестировщик: ключ `S.RCD.eval` раньше таблиц, `rd_eval.write/check` (только tmp), `--check` не пишет (№ 123), `registered_n`, формат слияния CHECKPOINT.
Боевые выходы, реестр и CHECKPOINT в tests не пишутся: `write` идёт в tmp-копию, `load_inputs` / `compute` подменены на синтетику."""
import hashlib
import json
import os
import time

import pytest

import test_s18_rd_eval_fx as FX
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import rd_eval as E
from models_v2.s18 import rd_eval_chk as K
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_null as RN

GAIN = {FX.R: 0.10, FX.CDF: 0.04, FX.RCK: 0.12}


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
    cp = {"S.RCD.registered": dict(registered_at=registered), C.KEY: dict(written_at=written, outputs={n: sha(v.encode("utf-8")) for n, v in tx.items()}, inputs={"docs/otchety/harj_struct/S_RCD_NULL.csv": sha(RN.OUT.read_bytes())})}
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
    tx4["S_RCD_TABLE.csv"] += "2\n"                                                     # число в таблице подменено, хэш ключа прежний
    assert any("хэш таблицы" in b for b in K.p_key({}, r4, c4, tx4))
    r5, c5, tx5, _ = key_setup(tmp_path / "mark")
    tx5["S_RCD_DEC.csv"] = "без пометки\n"
    assert any("пометка" in b for b in K.p_key({}, r5, c5, tx5))
    r7, c7, tx7, cp7 = key_setup(tmp_path / "nullsha")
    cp7[C.KEY]["inputs"]["docs/otchety/harj_struct/S_RCD_NULL.csv"] = "0" * 64         # нулевая диагностика подменена после записи ключа
    c7.write_text(json.dumps(cp7), encoding="utf-8")
    assert any("S_RCD_NULL.csv" in b for b in K.p_key({}, r7, c7, tx7))
    c6 = tmp_path / "nokey.json"
    c6.write_text(json.dumps({"S.RCD.registered": {}}), encoding="utf-8")
    assert K.p_key({}, rep, c6, tx) == ["ключа S.RCD.eval нет"]


def test_registered_n_known_answers_and_refusals(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"S.RCD.registered": dict(n=858, n_wide=3421), "S.R.registered": dict(n=852, n_wide=3409)}), encoding="utf-8")
    assert C.registered_n(p) == (858, 3421)
    p.write_text(json.dumps({"S.R.registered": dict(n=852, n_wide=3409)}), encoding="utf-8")
    with pytest.raises(SystemExit):                                                     # без S.RCD.registered — оценка отказана
        C.registered_n(p)
    p.write_text(json.dumps({"S.RCD.registered": dict(n=858)}), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.registered_n(p)


def test_merge_refuses_unsorted_checkpoint(tmp_path, monkeypatch):
    FX.patch_boot(monkeypatch)
    inp = FX.inputs(GAIN)
    t = FX.synth_t(inp)
    ckpt = tmp_path / "C.json"
    ckpt.write_text(json.dumps({"b": 1, "a": 2}), encoding="utf-8")                      # не indent 1 / sorted — слияние сломало бы чужие ключи
    with pytest.raises(SystemExit):
        E.merge_checkpoint(ckpt, t, E.texts(t, inp["tp"]), inp)
    S.idx_for.cache_clear()


def test_write_puts_key_before_tables_and_check_is_read_only_n123(tmp_path, monkeypatch):
    FX.patch_boot(monkeypatch)
    rep, ckpt = tmp_path / "rep", tmp_path / "CHECKPOINT.json"
    rep.mkdir()
    ckpt.write_text(json.dumps({"S.RCD.registered": dict(registered_at="01.01.2026 00:00:00", n=858, n_wide=3421), "S.R.registered": dict(n=852, n_wide=3409)}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    inp = FX.inputs(GAIN)
    monkeypatch.setattr(E, "load_inputs", lambda c=None: inp)
    monkeypatch.setattr(E, "compute", FX.synth_t)
    monkeypatch.setattr(K, "p_saved", lambda t, rep_: [])                                # сверка с таблицами прежних этапов читает боевые файлы — в tmp их нет
    log, orig = [], type(rep).write_text
    monkeypatch.setattr(type(rep), "write_text", lambda self, *a, **k: (log.append(self.name), orig(self, *a, **k))[1])
    E.write(rep, ckpt)
    assert log[0] == "CHECKPOINT.json" and sorted(log[1:]) == sorted(E.OUTS) and E.KEY in json.loads(ckpt.read_text(encoding="utf-8"))
    monkeypatch.setattr(type(rep), "write_text", orig)
    k = json.loads(ckpt.read_text(encoding="utf-8"))[E.KEY]
    assert (k["n_holm"], k["n_holm_wide"], k["m_family"], k["boot"], k["boot_block"], k["lag_main"], k["ktrim"], k["n"]) == (858, 3421, 6, 10000, 20, 8, 22, 2198)
    assert set(k["outputs"]) == set(E.OUTS) and k["decision"] in "1234" and isinstance(k["h_rcd1"], bool)
    assert E.check(rep, ckpt) == [] and K.p_key({}, rep, ckpt, E.texts(FX.synth_t(inp), inp["tp"])) == []
    h0 = E.tree_hash(rep, ckpt)
    E.check(rep, ckpt)
    assert E.tree_hash(rep, ckpt) == h0
    # мутант 1: ключ слит ПОСЛЕ таблиц (№ 30 / порядок спецификации)
    r2, c2 = tmp_path / "rep2", tmp_path / "C2.json"
    r2.mkdir()
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    cp.pop(E.KEY)
    c2.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    real_merge = E.merge_checkpoint
    monkeypatch.setattr(E, "merge_checkpoint", lambda *a, **k2: None)
    t = E.write(r2, c2)
    time.sleep(2.5)                                                                     # допуск проверки — 1 с
    real_merge(c2, t, E.texts(t, inp["tp"]), inp)
    assert any("позже самой ранней таблицы" in b for b in E.check(r2, c2))
    monkeypatch.setattr(E, "merge_checkpoint", real_merge)
    # мутант 2: `--check` пишет таблицу
    def writing_compute(i):
        (rep / "S_RCD_DEC.csv").write_text((rep / "S_RCD_DEC.csv").read_text(encoding="utf-8") + "x\n", encoding="utf-8")
        return FX.synth_t(i)
    monkeypatch.setattr(E, "compute", writing_compute)
    assert any("изменил" in b or "≠" in b for b in E.check(rep, ckpt))
    S.idx_for.cache_clear()


def test_check_detects_table_changed_after_key_and_stale_hash(tmp_path, monkeypatch):
    FX.patch_boot(monkeypatch)
    rep, ckpt = tmp_path / "rep", tmp_path / "CHECKPOINT.json"
    rep.mkdir()
    ckpt.write_text(json.dumps({"S.RCD.registered": dict(registered_at="01.01.2026 00:00:00", n=858, n_wide=3421)}, indent=1, sort_keys=True), encoding="utf-8")
    inp = FX.inputs(GAIN)
    monkeypatch.setattr(E, "load_inputs", lambda c=None: inp)
    monkeypatch.setattr(E, "compute", FX.synth_t)
    monkeypatch.setattr(K, "p_saved", lambda t, rep_: [])
    E.write(rep, ckpt)
    assert E.check(rep, ckpt) == []
    f = rep / "S_RCD_STATS.csv"
    txt = f.read_text(encoding="utf-8")
    f.write_text(txt + "extra\n", encoding="utf-8")                                         # строка в таблице добавлена после записи ключа
    assert "таблицы ≠ пересчёту" in E.check(rep, ckpt)
    f.write_text(txt, encoding="utf-8")
    assert E.check(rep, ckpt) == []
    S.idx_for.cache_clear()
