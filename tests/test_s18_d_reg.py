"""S-D-IMPL-2: регистратор `d_reg` — 2 строки одной записью, N из файла (+2 / +4), реестр до прогноза, прежние строки побайтно (D2.G3; мутации № 30, 123, 164/166, N-в-поправке)."""
import hashlib
import json
import os
import time

import pytest

from models_v2.s18 import d_reg as G

HEAD = "# реестр\nid,name,formula,params,status,registered_at\nY1,имя,формула,пар,зарегистрирована,10.11.2026 04:21:03\n"
NOW = "11.11.2026 20:40:00"


def _setup(tmp_path, r=846, rw=3397):
    t, c, s, d = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "SPEC.md", tmp_path / "D"
    t.write_text(HEAD, encoding="utf-8")
    s.write_text("спека", encoding="utf-8")
    c.write_text(json.dumps({"S.R.registered": dict(n=r, n_wide=rw), "S.C.registered": dict(n=836, n_wide=3377), "x": 1}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return t, c, s, d


def _reg(tmp_path, **kw):
    t, c, s, d = _setup(tmp_path, **kw)
    G.register(now=NOW, trials=t, d_dir=d, ckpt=c, spec=s)
    return t, c, s, d


def test_registers_2_rows_with_one_time_n_848_3401(tmp_path):
    t, c, s, d = _reg(tmp_path)
    assert G.check(t, d, c) == []
    k = json.loads(c.read_text(encoding="utf-8"))
    assert k[G.KEY]["n"] == 848 and k[G.KEY]["n_wide"] == 3401 and k[G.KEY]["rows"] == 2 and k["x"] == 1 and k["S.R.registered"]["n"] == 846
    txt = t.read_text(encoding="utf-8")
    assert txt.startswith(HEAD) and txt.count(NOW) == 2 and "S-RCKm," in txt and "S-RCKf," in txt


def test_n_follows_current_maximum_in_file_not_a_constant(tmp_path):
    t, c, s, d = _reg(tmp_path, r=850, rw=3400)
    k = json.loads(c.read_text(encoding="utf-8"))[G.KEY]
    assert k["n"] == 852 and k["n_wide"] == 3404 and k["n_base"] == 850


def test_mut_n_equals_row_count_or_ignores_other_registrations_is_caught(tmp_path):
    t, c, s, d = _reg(tmp_path)
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY]["n"] = 2
    c.write_text(json.dumps(k, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("ключ" in b for b in G.check(t, d, c))
    cp = {"S.R.registered": dict(n=846, n_wide=3397), "S.C.registered": dict(n=836, n_wide=3377), "S.D.registered": dict(n=999, n_wide=9999), "S.A.eval": dict(n=5000)}
    assert G._base(cp, "n") == 846 and G._base(cp, "n_wide") == 3397    # максимум по чужим S.*.registered; свой ключ и eval не считаются


def test_mut_registry_after_forecast_is_refused(tmp_path):
    t, c, s, d = _setup(tmp_path)
    d.mkdir()
    (d / "rckm.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="позже прогноза"):
        G.register(now=NOW, trials=t, d_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD and G.KEY not in json.loads(c.read_text(encoding="utf-8"))


def test_repeat_registration_is_refused(tmp_path):
    t, c, s, d = _reg(tmp_path)
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        G.register(now="11.11.2026 20:41:00", trials=t, d_dir=d, ckpt=c, spec=s)


def test_check_by_id_not_by_position(tmp_path):
    t, c, s, d = _reg(tmp_path)
    with t.open("a", encoding="utf-8") as fh:
        fh.write("Z9,другой лист,ф,п,зарегистрирована,11.11.2026 21:00:00\n")
    assert G.check(t, d, c) == []
    ok = t.read_text(encoding="utf-8")
    t.write_text("\n".join(ln for ln in ok.splitlines() if not ln.startswith("S-RCKf,")) + "\n", encoding="utf-8")
    assert G.check(t, d, c) != []


def test_mut_two_times_and_prior_rows_changed_are_caught(tmp_path):
    t, c, s, d = _reg(tmp_path)
    ok = t.read_text(encoding="utf-8")
    t.write_text(ok.replace("20:40:00", "20:40:01", 1), encoding="utf-8")
    assert any("registered_at не одна" in b for b in G.check(t, d, c))
    t.write_text(ok.replace("формула,пар", "формула,ПАР", 1), encoding="utf-8")
    assert any("прежняя часть" in b for b in G.check(t, d, c))


def test_mut_parquet_older_than_registration_is_caught(tmp_path):
    t, c, s, d = _reg(tmp_path)
    d.mkdir()
    p = d / "rckm.parquet"
    p.write_bytes(b"x")
    old = time.mktime((2026, 11, 11, 18, 0, 0, 0, 0, -1))
    os.utime(p, (old, old))
    assert any("не раньше mtime" in b for b in G.check(t, d, c))


def test_check_only_reads_files(tmp_path):
    t, c, s, d = _reg(tmp_path)
    h = lambda: [hashlib.sha256(p.read_bytes()).hexdigest() for p in (t, c, s)]
    h0 = h()
    G.check(t, d, c)
    assert h0 == h()


def test_rows_are_ablations_not_candidates_and_cost_two_in_n():
    assert G.ADD == 2 and G.ADD_WIDE == 4 and G.IDS == ("S-RCKm", "S-RCKf") and all(r[4].startswith("абляция") and "не кандидат" in r[4] for r in G.ROWS)
    assert "π := 1" in G.ROWS[1][2] and "m_T := 1" in G.ROWS[0][2]


def test_ckpt_without_s_keys_is_refused(tmp_path):
    t, c, s, d = _setup(tmp_path)
    c.write_text(json.dumps({"x": 1}, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(SystemExit):
        G.register(now=NOW, trials=t, d_dir=d, ckpt=c, spec=s)


def test_real_registry_row_ids_present_and_before_forecast_check_is_by_id():
    ids = [ln.split(",", 1)[0] for ln in G.TRIALS.read_text(encoding="utf-8").splitlines() if ln.startswith(("S-RCKm,", "S-RCKf,"))]
    assert ids in ([], ["S-RCKm", "S-RCKf"])
