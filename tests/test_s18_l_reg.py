"""S-L-IMPL-2: регистратор `l_reg` — 3 строки одной записью, N из файла (+3 / +6), реестр до прогноза, прежние строки побайтно (L2.G4; мутации № 30, 123, 164/166, N-в-поправке)."""
import hashlib
import json
import os
import time

import pytest

from models_v2.s18 import l_reg as G

HEAD = "# реестр\nid,name,formula,params,status,registered_at\nY1,имя,формула,пар,зарегистрирована,10.11.2026 04:21:03\n"
NOW = "11.11.2026 21:20:00"


def _setup(tmp_path, d=848, dw=3401):
    t, c, s, p = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "SPEC.md", tmp_path / "L"
    t.write_text(HEAD, encoding="utf-8")
    s.write_text("спека", encoding="utf-8")
    c.write_text(json.dumps({"S.D.registered": dict(n=d, n_wide=dw), "S.R.registered": dict(n=846, n_wide=3397), "x": 1}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return t, c, s, p


def _reg(tmp_path, **kw):
    t, c, s, p = _setup(tmp_path, **kw)
    G.register(now=NOW, trials=t, l_dir=p, ckpt=c, spec=s)
    return t, c, s, p


def test_registers_3_rows_with_one_time_n_851_3407(tmp_path):
    t, c, s, p = _reg(tmp_path)
    assert G.check(t, p, c) == []
    k = json.loads(c.read_text(encoding="utf-8"))
    assert k[G.KEY]["n"] == 851 and k[G.KEY]["n_wide"] == 3407 and k[G.KEY]["rows"] == 3 and k["x"] == 1 and k["S.D.registered"]["n"] == 848
    txt = t.read_text(encoding="utf-8")
    assert txt.startswith(HEAD) and txt.count(NOW) == 3 and all(("%s," % i) in txt for i in ("S-RL1", "S-RL1a", "S-RL1b"))


def test_n_follows_current_maximum_in_file_not_a_constant(tmp_path):
    t, c, s, p = _reg(tmp_path, d=860, dw=3420)
    k = json.loads(c.read_text(encoding="utf-8"))[G.KEY]
    assert k["n"] == 863 and k["n_wide"] == 3426 and k["n_base"] == 860


def test_mut_n_equals_row_count_or_ignores_other_registrations_is_caught(tmp_path):
    t, c, s, p = _reg(tmp_path)
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY]["n"] = 3
    c.write_text(json.dumps(k, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("ключ" in b for b in G.check(t, p, c))
    cp = {"S.R.registered": dict(n=846, n_wide=3397), "S.D.registered": dict(n=848, n_wide=3401), "S.L.registered": dict(n=999, n_wide=9999), "S.A.eval": dict(n=5000)}
    assert G._base(cp, "n") == 848 and G._base(cp, "n_wide") == 3401    # максимум по чужим S.*.registered; свой ключ и eval не считаются


def test_mut_registry_after_forecast_is_refused(tmp_path):
    t, c, s, p = _setup(tmp_path)
    p.mkdir()
    (p / "rl1.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="позже прогноза"):
        G.register(now=NOW, trials=t, l_dir=p, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD and G.KEY not in json.loads(c.read_text(encoding="utf-8"))


def test_repeat_registration_is_refused(tmp_path):
    t, c, s, p = _reg(tmp_path)
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        G.register(now="11.11.2026 21:21:00", trials=t, l_dir=p, ckpt=c, spec=s)


def test_check_by_id_not_by_position(tmp_path):
    t, c, s, p = _reg(tmp_path)
    with t.open("a", encoding="utf-8") as fh:
        fh.write("Z9,другой лист,ф,п,зарегистрирована,11.11.2026 21:30:00\n")
    assert G.check(t, p, c) == []
    ok = t.read_text(encoding="utf-8")
    t.write_text("\n".join(ln for ln in ok.splitlines() if not ln.startswith("S-RL1b,")) + "\n", encoding="utf-8")
    assert G.check(t, p, c) != []


def test_mut_two_times_and_prior_rows_changed_are_caught(tmp_path):
    t, c, s, p = _reg(tmp_path)
    ok = t.read_text(encoding="utf-8")
    t.write_text(ok.replace("21:20:00", "21:20:01", 1), encoding="utf-8")
    assert any("registered_at не одна" in b for b in G.check(t, p, c))
    t.write_text(ok.replace("формула,пар", "формула,ПАР", 1), encoding="utf-8")
    assert any("прежняя часть" in b for b in G.check(t, p, c))


def test_mut_parquet_older_than_registration_is_caught(tmp_path):
    t, c, s, p = _reg(tmp_path)
    p.mkdir()
    f = p / "rl1.parquet"
    f.write_bytes(b"x")
    old = time.mktime((2026, 11, 11, 18, 0, 0, 0, 0, -1))
    os.utime(f, (old, old))
    assert any("не раньше mtime" in b for b in G.check(t, p, c))


def test_check_only_reads_files(tmp_path):
    t, c, s, p = _reg(tmp_path)
    h = lambda: [hashlib.sha256(q.read_bytes()).hexdigest() for q in (t, c, s)]
    h0 = h()
    G.check(t, p, c)
    assert h0 == h()


def test_rows_are_candidate_and_two_controls_and_cost_three_in_n():
    assert G.ADD == 3 and G.ADD_WIDE == 6 and G.IDS == ("S-RL1", "S-RL1a", "S-RL1b")
    assert G.ROWS[0][4].startswith("кандидат") and all(r[4].startswith("контроль") and "не кандидат" in r[4] for r in G.ROWS[1:])
    assert "π̄" in G.ROWS[1][1] and "252" in G.ROWS[2][1] and "a ∈ [−0,5; 0,5]" in G.ROWS[0][2]


def test_ckpt_without_s_keys_is_refused(tmp_path):
    t, c, s, p = _setup(tmp_path)
    c.write_text(json.dumps({"x": 1}, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(SystemExit):
        G.register(now=NOW, trials=t, l_dir=p, ckpt=c, spec=s)


def test_real_registry_row_ids_present_and_before_forecast_check_is_by_id():
    ids = [ln.split(",", 1)[0] for ln in G.TRIALS.read_text(encoding="utf-8").splitlines() if ln.startswith(("S-RL1,", "S-RL1a,", "S-RL1b,"))]
    assert ids in ([], ["S-RL1", "S-RL1a", "S-RL1b"])
