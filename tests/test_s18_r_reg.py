"""S-R-IMPL-2: регистратор `r_reg` — 12 строк одной записью, N из файла, реестр до прогноза, прежние строки побайтно (W.G1; мутации № 30, 164/166)."""
import json
import os
import time

import pytest

from models_v2.s18 import r_reg as G

HEAD = "# реестр\nid,name,formula,params,status,registered_at\nY1,имя,формула,пар,зарегистрирована,10.11.2026 04:21:03\n"


def _setup(tmp_path, n=836, nw=3377):
    t, c, s, d = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "SPEC.md", tmp_path / "R"
    t.write_text(HEAD, encoding="utf-8")
    s.write_text("спека", encoding="utf-8")
    c.write_text(json.dumps({"S.C.registered": dict(n=n, n_wide=nw), "S.A.registered": dict(n=831, n_wide=3367), "x": 1}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return t, c, s, d


def _reg(tmp_path, **kw):
    t, c, s, d = _setup(tmp_path)
    G.register(now="11.11.2026 19:00:00", trials=t, r_dir=d, ckpt=c, spec=s, **kw)
    return t, c, s, d


def test_registers_12_rows_with_one_time_and_n_from_file(tmp_path):
    t, c, s, d = _reg(tmp_path)
    assert G.check(t, d, c) == []
    k = json.loads(c.read_text(encoding="utf-8"))
    assert k[G.KEY]["n"] == 846 and k[G.KEY]["n_wide"] == 3397 and k[G.KEY]["rows"] == 12 and k["x"] == 1 and k["S.C.registered"]["n"] == 836
    assert t.read_text(encoding="utf-8").startswith(HEAD)
    assert t.read_text(encoding="utf-8").count("11.11.2026 19:00:00") == 12


def test_n_follows_current_value_in_file_not_the_spec(tmp_path):
    t, c, s, d = _setup(tmp_path, n=850, nw=3400)
    G.register(now="11.11.2026 19:00:00", trials=t, r_dir=d, ckpt=c, spec=s)
    assert json.loads(c.read_text(encoding="utf-8"))[G.KEY]["n"] == 860


def test_mut_registry_after_forecast_is_refused(tmp_path):
    t, c, s, d = _setup(tmp_path)
    d.mkdir()
    (d / "ra2.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="позже прогноза"):
        G.register(now="11.11.2026 19:00:00", trials=t, r_dir=d, ckpt=c, spec=s)
    assert t.read_text(encoding="utf-8") == HEAD


def test_repeat_registration_is_refused(tmp_path):
    t, c, s, d = _reg(tmp_path)
    with pytest.raises(SystemExit, match="уже зарегистрированы"):
        G.register(now="11.11.2026 19:01:00", trials=t, r_dir=d, ckpt=c, spec=s)


def test_check_by_id_not_by_position(tmp_path):
    t, c, s, d = _reg(tmp_path)
    with t.open("a", encoding="utf-8") as fh:
        fh.write("Z9,другой лист,ф,п,зарегистрирована,11.11.2026 20:00:00\n")
    assert G.check(t, d, c) == []


def test_mut_row_missing_or_two_times_or_wrong_n_are_caught(tmp_path):
    t, c, s, d = _reg(tmp_path)
    ok = t.read_text(encoding="utf-8")
    t.write_text("\n".join(ln for ln in ok.splitlines() if not ln.startswith("S-B3,")) + "\n", encoding="utf-8")
    assert G.check(t, d, c) != []
    t.write_text(ok.replace("S-B4,", "S-B4,", 1).replace("19:00:00", "19:00:01", 1), encoding="utf-8")
    assert any("registered_at не одна" in b for b in G.check(t, d, c))
    t.write_text(ok, encoding="utf-8")
    k = json.loads(c.read_text(encoding="utf-8"))
    k[G.KEY]["n"] = k[G.KEY]["n_base"] + 12
    c.write_text(json.dumps(k, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("ключ" in b for b in G.check(t, d, c))


def test_mut_prior_rows_changed_is_caught(tmp_path):
    t, c, s, d = _reg(tmp_path)
    t.write_text(t.read_text(encoding="utf-8").replace("формула,пар", "формула,ПАР", 1), encoding="utf-8")
    assert any("прежняя часть" in b for b in G.check(t, d, c))


def test_mut_parquet_older_than_registration_is_caught(tmp_path):
    t, c, s, d = _reg(tmp_path)
    d.mkdir()
    p = d / "ra2.parquet"
    p.write_bytes(b"x")
    old = time.mktime((2026, 11, 11, 18, 0, 0, 0, 0, -1))
    os.utime(p, (old, old))
    assert any("не раньше mtime" in b for b in G.check(t, d, c))


def test_ids_and_not_run_rows_do_not_count_in_n():
    assert G.ADD == 10 and G.ADD_WIDE == 20 and sum(r[4] == G.NOT_RUN for r in G.ROWS) == 2 and sum(r[0].startswith("S-B") for r in G.ROWS) == 4


def test_ckpt_without_s_keys_is_refused(tmp_path):
    t, c, s, d = _setup(tmp_path)
    c.write_text(json.dumps({"x": 1}, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(SystemExit):
        G.register(now="11.11.2026 19:00:00", trials=t, r_dir=d, ckpt=c, spec=s)
