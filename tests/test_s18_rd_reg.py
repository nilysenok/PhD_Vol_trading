"""S-RCD-IMPL-2 (Ж-259), тестировщик: регистратор `rd_reg` — 3 строки одной записью, N из файла (+3 / +6), реестр до прогноза (№ 30), прежние строки побайтно, проверка по id, а не по позиции (№ 164 / 166), `check` только читает (№ 123).
Всё через tmp: боевые реестр, CHECKPOINT и RD/ не пишутся. Мутации — `test_s18_rd_reg_mut`."""
import datetime as dt
import hashlib
import json
import os

import pytest

from models_v2.s18 import rd_reg as G

HEAD = "# реестр\nid,name,formula,params,status,registered_at\nY1,имя,формула,пар,зарегистрирована,10.11.2026 04:21:03\n"
NOW = "12.11.2026 01:28:31"


def dump(c, k):
    c.write_text(json.dumps(k, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def setup(tmp_path, n=852, nw=3409):
    t, c, s, d = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "SPEC.md", tmp_path / "RD"
    t.write_text(HEAD, encoding="utf-8")
    s.write_text("спека", encoding="utf-8")
    dump(c, {"S.R.registered": dict(n=n, n_wide=nw), "S.A.registered": dict(n=831, n_wide=3367), "x": 1})
    return t, c, s, d


def reg(tmp_path, **kw):
    t, c, s, d = setup(tmp_path)
    G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s, **kw)
    return t, c, s, d


def test_registers_three_rows_one_time_n_plus_3_and_6_from_file(tmp_path, capsys):
    t, c, s, d = reg(tmp_path)
    assert G.check(t, d, c) == [] and G.IDS == ("S-RCD", "S-CDF", "S-CKF") and (G.ADD, G.ADD_WIDE, G.KEY) == (3, 6, "S.RCD.registered")
    k = json.loads(c.read_text(encoding="utf-8"))
    assert k[G.KEY]["n"] == 855 and k[G.KEY]["n_wide"] == 3415 and k[G.KEY]["rows"] == 3 and k[G.KEY]["n_base"] == 852 and k["x"] == 1 and set(k) == {"S.R.registered", "S.A.registered", "x", G.KEY}
    txt = t.read_text(encoding="utf-8")
    assert txt.startswith(HEAD) and txt.count(NOW) == 3 and [ln.split(",")[0] for ln in txt.splitlines()[-3:]] == list(G.IDS)
    assert "N = 855" in capsys.readouterr().out and not list(tmp_path.glob("*.tmp"))


def test_n_follows_current_file_value_and_ignores_own_old_key(tmp_path):
    t, c, s, d = setup(tmp_path, n=900, nw=3500)
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY] = dict(n=999, n_wide=9999)
    assert G.base_n(k) == (900, 3500)
    k.pop(G.KEY)
    dump(c, k)
    G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    assert (json.loads(c.read_text(encoding="utf-8"))[G.KEY]["n"], json.loads(c.read_text(encoding="utf-8"))[G.KEY]["n_wide"]) == (903, 3506)


def test_base_n_refuses_without_s_keys_and_takes_maximum():
    with pytest.raises(SystemExit):
        G.base_n({"x": 1, "S.A.registered": "не словарь"})
    assert G.base_n({"S.A.registered": dict(n=5, n_wide=50), "S.B.registered": dict(n=7, n_wide=40)}) == (7, 50)


def test_prior_rows_stay_bytewise_and_hash_is_recorded(tmp_path):
    t, c, s, d = reg(tmp_path)
    k = json.loads(c.read_text(encoding="utf-8"))[G.KEY]
    assert t.read_bytes()[: k["prior_bytes"]] == HEAD.encode("utf-8") and k["prior_sha256"] == hashlib.sha256(HEAD.encode("utf-8")).hexdigest()
    t.write_text(t.read_text(encoding="utf-8").replace("формула,пар", "формула,ПАР", 1), encoding="utf-8")
    assert any("прежняя часть" in b for b in G.check(t, d, c))


def test_prior_without_final_newline_is_kept_bytewise(tmp_path):
    t, c, s, d = setup(tmp_path)
    t.write_text(HEAD.rstrip("\n"), encoding="utf-8")
    G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    assert t.read_bytes().startswith(HEAD.rstrip("\n").encode("utf-8")) and G.check(t, d, c) == []


@pytest.mark.parametrize("name", ["rcd.parquet", "cdf.parquet", "ckf.parquet", "other.parquet"])
def test_registry_after_forecast_is_refused_and_nothing_written(tmp_path, name):     # № 30
    t, c, s, d = setup(tmp_path)
    d.mkdir()
    (d / name).write_bytes(b"x")
    before = c.read_bytes()
    with pytest.raises(SystemExit, match="позже прогноза"):
        G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD and c.read_bytes() == before


def test_repeat_and_partial_registration_are_refused(tmp_path):
    t, c, s, d = reg(tmp_path)
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        G.register(now="12.11.2026 01:29:00", trials=t, rd_dir=d, ckpt=c, spec=s)
    (tmp_path / "b").mkdir()
    u, c2, s2, d2 = setup(tmp_path / "b")
    u.write_text(HEAD + "S-CDF,a,b,c,d,10.11.2026 05:00:00\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="S-CDF"):
        G.register(now=NOW, trials=u, rd_dir=d2, ckpt=c2, spec=s2)
    assert u.read_text(encoding="utf-8").count("S-RCD") == 0


def test_foreign_checkpoint_format_is_refused_before_any_write(tmp_path):
    t, c, s, d = setup(tmp_path)
    c.write_text(json.dumps({"S.R.registered": dict(n=852, n_wide=3409)}, indent=4), encoding="utf-8")
    with pytest.raises(SystemExit, match="формат"):
        G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD


def test_checkpoint_without_s_keys_is_refused_before_any_write(tmp_path):
    t, c, s, d = setup(tmp_path)
    dump(c, {"x": 1})
    before = c.read_bytes()
    with pytest.raises(SystemExit):
        G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD and c.read_bytes() == before


def test_existing_key_in_checkpoint_is_refused_before_any_write(tmp_path):
    t, c, s, d = setup(tmp_path)
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY] = dict(n=1, n_wide=1)
    dump(c, k)
    with pytest.raises(SystemExit, match="ключ"):
        G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD


def test_check_by_id_not_by_position(tmp_path):
    t, c, s, d = reg(tmp_path)
    ok = t.read_text(encoding="utf-8")
    with t.open("a", encoding="utf-8") as fh:
        fh.write("Z9,другой лист,ф,п,зарегистрирована,12.11.2026 02:00:00\n")
    assert G.check(t, d, c) == []
    rows = ok.splitlines()[-3:]
    t.write_text(ok.replace(rows[1] + "\n", "").replace(rows[0] + "\n", rows[0] + "\nZ8,чужая,ф,п,з,12.11.2026 02:00:00\n"), encoding="utf-8")
    assert any("строк S-RCD" in b for b in G.check(t, d, c))     # одной нашей строки нет — замечание, чужая строка между ними не мешает счёту по id


def test_check_flags_missing_duplicate_and_unequal_time_rows(tmp_path):
    t, c, s, d = reg(tmp_path)
    ok = t.read_text(encoding="utf-8")
    cdf = [ln for ln in ok.splitlines() if ln.startswith("S-CDF")][0]
    t.write_text(ok + cdf + "\n", encoding="utf-8")
    assert any("строк S-RCD" in b for b in G.check(t, d, c))
    t.write_text(ok.replace(cdf, cdf.replace(NOW, "12.11.2026 01:28:32")), encoding="utf-8")
    assert any("registered_at не одна" in b for b in G.check(t, d, c))


@pytest.mark.parametrize("field,delta", [("n", 1), ("n_wide", -1), ("rows", 1)])
def test_check_flags_wrong_n_in_key(tmp_path, field, delta):
    t, c, s, d = reg(tmp_path)
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY][field] += delta
    dump(c, k)
    assert any("ключ" in b for b in G.check(t, d, c))


def test_check_registered_at_before_earliest_parquet_and_before_eval_key(tmp_path):
    t, c, s, d = reg(tmp_path)
    d.mkdir()
    p = d / "rcd.parquet"
    p.write_bytes(b"x")
    ts = lambda h, m: dt.datetime(2026, 11, 12, h, m, 0).timestamp()
    os.utime(p, (ts(1, 29), ts(1, 29)))
    assert G.check(t, d, c) == []
    os.utime(p, (ts(1, 28), ts(1, 28)))     # parquet старше регистрации
    assert any("не раньше mtime" in b for b in G.check(t, d, c))
    os.utime(p, (ts(1, 29), ts(1, 29)))
    k = json.loads(c.read_text(encoding="utf-8"))
    k["S.RCD.eval"] = dict(written_at="12.11.2026 01:00:00")
    dump(c, k)
    assert any("S.RCD.eval" in b for b in G.check(t, d, c))
    k["S.RCD.eval"] = dict(written_at="12.11.2026 02:00:00")
    dump(c, k)
    assert G.check(t, d, c) == []


def test_require_refuses_unregistered_id_and_accepts_registered(tmp_path):
    t, c, s, d = setup(tmp_path)
    with pytest.raises(SystemExit, match="реестр до прогноза"):
        G.require("S-RCD", trials=t)
    G.register(now=NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    for i in G.IDS:
        G.require(i, trials=t)
    with pytest.raises(SystemExit):
        G.require("S-RCD-НЕТ", trials=t)


def test_check_changes_no_file_in_tmp(tmp_path):     # № 123
    t, c, s, d = reg(tmp_path)
    d.mkdir()
    (d / "rcd.parquet").write_bytes(b"x")
    snap = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    before = snap()
    G.check(t, d, c)
    assert snap() == before
