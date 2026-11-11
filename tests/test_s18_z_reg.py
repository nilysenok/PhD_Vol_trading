"""S-Z-IMPL-2 (Ж-258), тестировщик: регистратор `z_reg` — три строки одной записью, N из файла (+3 / +6), реестр до прогноза (№ 30), прежние строки побайтно, проверка по id, а не по позиции (№ 164/166).
Всё через tmp: боевой реестр и CHECKPOINT не пишутся; боевые читаются только `check()`."""
import hashlib
import json
import os
import time

import pytest

from models_v2.s18 import z_reg as G

HEAD = "# реестр\nid,name,formula,params,status,registered_at\nY1,имя,формула,пар,зарегистрирована,10.11.2026 04:21:03\n"
NOW = "12.11.2026 01:28:31"


def setup(tmp_path, n=852, nw=3409):
    t, c, s, d = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "SPEC.md", tmp_path / "Z"
    t.write_text(HEAD, encoding="utf-8")
    s.write_text("спека", encoding="utf-8")
    c.write_text(json.dumps({"S.R.registered": dict(n=n, n_wide=nw), "S.A.registered": dict(n=831, n_wide=3367), "x": 1}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return t, c, s, d


def reg(tmp_path, **kw):
    t, c, s, d = setup(tmp_path)
    G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s, **kw)
    return t, c, s, d


def dump(c, k):
    c.write_text(json.dumps(k, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def test_registers_three_rows_one_time_n_plus_3_and_6_from_file(tmp_path):
    t, c, s, d = reg(tmp_path)
    assert G.check(t, d, c) == [] and G.IDS == ("S-RCKZ", "S-RZ", "S-RCKZs") and (G.ADD, G.ADD_WIDE) == (3, 6)
    k = json.loads(c.read_text(encoding="utf-8"))
    assert k[G.KEY]["n"] == 855 and k[G.KEY]["n_wide"] == 3415 and k[G.KEY]["rows"] == 3 and k["x"] == 1 and k["S.R.registered"]["n"] == 852 and set(k) == {"S.R.registered", "S.A.registered", "x", G.KEY}
    txt = t.read_text(encoding="utf-8")
    assert txt.startswith(HEAD) and txt.count(NOW) == 3 and [ln.split(",")[0] for ln in txt.splitlines()[-3:]] == list(G.IDS)


def test_n_follows_current_value_in_file_and_ignores_own_old_key(tmp_path):
    t, c, s, d = setup(tmp_path, n=900, nw=3500)
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY] = dict(n=999, n_wide=9999)
    dump(c, k)
    G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s)
    assert json.loads(c.read_text(encoding="utf-8"))[G.KEY]["n"] == 903 and json.loads(c.read_text(encoding="utf-8"))[G.KEY]["n_wide"] == 3506


def test_prior_rows_stay_bytewise_and_hash_is_recorded(tmp_path):
    t, c, s, d = reg(tmp_path)
    k = json.loads(c.read_text(encoding="utf-8"))[G.KEY]
    assert t.read_bytes()[: k["prior_bytes"]] == HEAD.encode("utf-8") and k["prior_sha256"] == hashlib.sha256(HEAD.encode("utf-8")).hexdigest()
    t.write_text(t.read_text(encoding="utf-8").replace("формула,пар", "формула,ПАР", 1), encoding="utf-8")      # мутация: прежняя строка изменена
    assert any("прежняя часть" in b for b in G.check(t, d, c))


def test_prior_without_final_newline_is_kept_bytewise(tmp_path):
    t, c, s, d = setup(tmp_path)
    t.write_text(HEAD.rstrip("\n"), encoding="utf-8")
    G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s)
    assert t.read_bytes().startswith(HEAD.rstrip("\n").encode("utf-8")) and G.check(t, d, c) == []


@pytest.mark.parametrize("name", ["rckz.parquet", "zmaps.npz"])
def test_mut_registry_after_forecast_is_refused_and_nothing_written(tmp_path, name):
    t, c, s, d = setup(tmp_path)
    d.mkdir()
    (d / name).write_bytes(b"x")
    before = c.read_bytes()
    with pytest.raises(SystemExit, match="позже прогноза"):
        G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD and c.read_bytes() == before


def test_repeat_and_partial_registration_are_refused(tmp_path):
    t, c, s, d = reg(tmp_path)
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        G.register(now="12.11.2026 01:29:00", trials=t, z_dir=d, ckpt=c, spec=s)
    u, c2, s2, d2 = setup(tmp_path / "b") if (tmp_path / "b").mkdir() is None else None
    u.write_text(HEAD + "S-RZ,a,b,c,d,10.11.2026 05:00:00\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="S-RZ"):
        G.register(now=NOW, trials=u, z_dir=d2, ckpt=c2, spec=s2)
    assert u.read_text(encoding="utf-8").count("S-RCKZ") == 0


def test_foreign_checkpoint_format_is_refused_before_any_write(tmp_path):
    t, c, s, d = setup(tmp_path)
    c.write_text(json.dumps({"S.R.registered": dict(n=852, n_wide=3409)}, indent=4), encoding="utf-8")
    with pytest.raises(SystemExit, match="формат"):
        G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD


# Был xfail(strict): дефект z_reg.register (запись реестра до проверки CHECKPOINT) найден тестировщиком 12.11, исправлен автором; метку снял дирижёр 01:5x.
def test_checkpoint_without_s_keys_is_refused_before_any_write(tmp_path):
    t, c, s, d = setup(tmp_path)
    dump(c, {"x": 1})
    with pytest.raises(SystemExit):
        G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD


def test_check_by_id_not_by_position(tmp_path):
    t, c, s, d = reg(tmp_path)
    ok = t.read_text(encoding="utf-8")
    with t.open("a", encoding="utf-8") as fh:
        fh.write("Z9,другой лист,ф,п,зарегистрирована,12.11.2026 02:00:00\n")
    assert G.check(t, d, c) == []                                          # чужая строка после наших трёх
    head, rows = ok.split("Y1,")[0], ok.splitlines()[-3:]
    t.write_text(HEAD + "\n".join([rows[2], "Z9,другой,ф,п,зарегистрирована,12.11.2026 02:00:00", rows[0], rows[1]]) + "\n", encoding="utf-8")
    bad = G.check(t, d, c)
    assert not any("строк S-Z" in b or "registered_at не одна" in b or "ключ" in b for b in bad)    # порядок и вставка чужой строки между ними не ломают проверку по id


def test_mut_row_missing_two_times_wrong_n_or_rows_are_caught(tmp_path):
    t, c, s, d = reg(tmp_path)
    ok, k0 = t.read_text(encoding="utf-8"), json.loads(c.read_text(encoding="utf-8"))
    t.write_text("\n".join(ln for ln in ok.splitlines() if not ln.startswith("S-RZ,")) + "\n", encoding="utf-8")
    assert any("строк S-Z" in b for b in G.check(t, d, c))
    t.write_text(ok.replace(NOW, "12.11.2026 01:28:32", 1), encoding="utf-8")
    assert any("registered_at не одна" in b for b in G.check(t, d, c))
    t.write_text(ok + ok.splitlines()[-1] + "\n", encoding="utf-8")
    assert any("строк S-Z" in b for b in G.check(t, d, c))                 # дубль id
    t.write_text(ok, encoding="utf-8")
    for f, v in (("n", 856), ("n_wide", 3416), ("rows", 2), ("registered_at", "12.11.2026 01:00:00")):
        k = json.loads(json.dumps(k0))
        k[G.KEY][f] = v
        dump(c, k)
        assert any("ключ" in b for b in G.check(t, d, c)), f
    dump(c, k0)
    assert G.check(t, d, c) == []


def test_mut_output_older_than_registration_or_eval_key_is_caught(tmp_path):
    t, c, s, d = reg(tmp_path)
    d.mkdir()
    p = d / "rz.parquet"
    p.write_bytes(b"x")
    assert G.check(t, d, c) == []                                          # свежий файл позже registered_at (01:28:31 12.11.2026 < now)
    old = time.mktime((2026, 11, 12, 1, 0, 0, 0, 0, -1))
    os.utime(p, (old, old))
    assert any("не раньше mtime" in b for b in G.check(t, d, c))
    new = time.time()
    os.utime(p, (new, new))
    k = json.loads(c.read_text(encoding="utf-8"))
    k["S.Z.eval"] = dict(written_at="12.11.2026 01:00:00")
    dump(c, k)
    assert any("S.Z.eval" in b for b in G.check(t, d, c))


def test_require_refuses_unknown_id_and_accepts_registered(tmp_path):
    t, c, s, d = setup(tmp_path)
    with pytest.raises(SystemExit, match="нет в реестре"):
        G.require("S-RCKZ", t)
    G.register(now=NOW, trials=t, z_dir=d, ckpt=c, spec=s)
    for i in G.IDS:
        G.require(i, t)
    with pytest.raises(SystemExit, match="нет в реестре"):
        G.require("S-RCKZx", t)


def test_merge_checkpoint_changes_only_own_key_with_other_keys_verbatim(tmp_path):
    t, c, s, d = setup(tmp_path)
    before = json.loads(c.read_text(encoding="utf-8"))
    key = G.merge_checkpoint(c, NOW, s, b"prior")
    after = json.loads(c.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != G.KEY} == before and after[G.KEY] == key and key["prior_sha256"] == hashlib.sha256(b"prior").hexdigest() and key["prior_bytes"] == 5


def test_real_registry_is_consistent_and_check_does_not_write():
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    h0 = (h(G.TRIALS), h(G.CKPT))
    assert G.check() == []
    k = json.loads(G.CKPT.read_text(encoding="utf-8"))[G.KEY]
    assert k["n"] == k["n_base"] + 3 and k["n_wide"] == k["n_wide_base"] + 6 and k["rows"] == 3 and (h(G.TRIALS), h(G.CKPT)) == h0
