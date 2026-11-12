"""S-RCD-IMPL-2 (Ж-259), тестировщик: мутации реестра и порядка записи (mutations.md № 30, 103, 123, 164 / 166). Подложенные реализации — копии порядка действий; детектор обязан быть непустым.
Плюс порядок `rd_walk.run`: реестр до прогноза, перезапись иного выхода отказана. Всё через tmp."""
import datetime as dt
import hashlib
import json
import os

import pandas as pd
import pytest

from models_v2.s18 import rd_reg as G
from models_v2.s18 import rd_walk as RD
import test_s18_rd_reg as T

ORIG_REGISTER, ORIG_CHECK, ORIG_READ = G.register, G.check, G._read


def refused_before_write(fn, tmp_path) -> list:
    """Сценарий «в CHECKPOINT нет ключей S.*.registered»: отказ обязан быть ДО любой записи в реестр."""
    t, c, s, d = T.setup(tmp_path)
    T.dump(c, {"x": 1})
    try:
        fn(now=T.NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    except SystemExit:
        pass
    return ["реестр изменён до отказа"] * (t.read_text(encoding="utf-8") != T.HEAD)


def late_registry_problems(fn, tmp_path) -> list:
    """Реестр позже прогноза: есть RD/*.parquet — регистрация обязана быть отказана, реестр не тронут (№ 30)."""
    t, c, s, d = T.setup(tmp_path)
    d.mkdir()
    (d / "rcd.parquet").write_bytes(b"x")
    try:
        fn(now=T.NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    except SystemExit:
        return ["реестр изменён"] * (t.read_text(encoding="utf-8") != T.HEAD)
    return ["регистрация допущена при готовом parquet"]


def n_problems(tmp_path) -> list:
    t, c, s, d = T.setup(tmp_path)
    G.register(now=T.NOW, trials=t, rd_dir=d, ckpt=c, spec=s)
    k = json.loads(c.read_text(encoding="utf-8"))[G.KEY]
    return ["N = %s, N_wide = %s вместо 855 / 3415" % (k["n"], k["n_wide"])] * (k["n"] != 855 or k["n_wide"] != 3415)


def position_problems(check_fn, tmp_path) -> list:
    t, c, s, d = T.reg(tmp_path)
    with t.open("a", encoding="utf-8") as fh:
        fh.write("Z9,другой лист,ф,п,зарегистрирована,12.11.2026 02:00:00\n")
    return ["чужая строка после наших трёх сломала проверку: %s" % b for b in check_fn(t, d, c)]


def mtime_problems(check_fn, tmp_path) -> list:
    t, c, s, d = T.reg(tmp_path)
    d.mkdir()
    p = d / "cdf.parquet"
    p.write_bytes(b"x")
    ts = dt.datetime(2026, 11, 12, 1, 0, 0).timestamp()
    os.utime(p, (ts, ts))     # parquet старше регистрации (01:28:31)
    return ["registered_at позже parquet не замечено"] * (check_fn(t, d, c) == [])


def readonly_problems(check_fn, tmp_path) -> list:
    t, c, s, d = T.reg(tmp_path)
    d.mkdir()
    (d / "rcd.parquet").write_bytes(b"x")
    snap = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    before = snap()
    check_fn(t, d, c)
    return ["check изменил файлы: %s" % sorted(set(snap().items()) ^ set(before.items()))] * (snap() != before)


def test_honest_detectors_are_clean(tmp_path):
    for i, f in enumerate((lambda p: refused_before_write(G.register, p), lambda p: late_registry_problems(G.register, p), n_problems, lambda p: position_problems(G.check, p), lambda p: mtime_problems(G.check, p), lambda p: readonly_problems(G.check, p))):
        (tmp_path / str(i)).mkdir()
        assert f(tmp_path / str(i)) == []


def test_mut_registry_written_before_checkpoint_check_is_caught(tmp_path):     # образец test_s18_z_reg
    def register(now=None, trials=G.TRIALS, rd_dir=G.RD_DIR, ckpt=G.CKPT, spec=G.SPEC):
        with trials.open("a", encoding="utf-8", newline="") as fh:
            fh.write(pd.DataFrame([dict(zip(G.COLS, (*r, now))) for r in G.ROWS])[G.COLS].to_csv(index=False, header=False, lineterminator="\n"))
        return ORIG_REGISTER(now=now, trials=trials, rd_dir=rd_dir, ckpt=ckpt, spec=spec)
    assert refused_before_write(register, tmp_path) != []


def test_mut_registry_after_forecast_is_not_refused_is_caught(tmp_path):     # № 30
    assert late_registry_problems(lambda **kw: ORIG_REGISTER(**{**kw, "rd_dir": tmp_path / "пусто"}), tmp_path) != []


def test_mut_registry_written_even_if_parquet_exists_is_caught(tmp_path):     # № 30
    def register(**kw):
        with kw["trials"].open("a", encoding="utf-8") as fh:
            fh.write("S-RCD,x,y,z,w,%s\n" % kw["now"])
        return ORIG_REGISTER(**kw)
    assert late_registry_problems(register, tmp_path) != []


def test_mut_wrong_add_to_n_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "ADD", 1)
    assert n_problems(tmp_path) != []


def test_mut_wrong_add_wide_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "ADD_WIDE", 3)
    assert n_problems(tmp_path) != []


def test_mut_check_by_position_of_last_three_rows_is_caught(tmp_path):     # № 164 / 166
    def check(trials=G.TRIALS, rd_dir=G.RD_DIR, ckpt=G.CKPT):
        cur = G._read(trials)
        return [] if list(cur["id"].tail(3)) == list(G.IDS) else ["строки реестра: хвост ≠ нашим трём"]
    assert position_problems(check, tmp_path) != []


def test_mut_check_ignores_parquet_mtime_is_caught(tmp_path):     # № 30
    assert mtime_problems(lambda t, d, c: ORIG_CHECK(t, d / "нет", c), tmp_path) != []


def test_mut_check_that_writes_is_caught(tmp_path):     # № 123
    def check(t, d, c):
        (tmp_path / "check.log").write_text("ok", encoding="utf-8")
        return ORIG_CHECK(t, d, c)
    assert readonly_problems(check, tmp_path) != []


def test_mut_check_rewrites_trials_is_caught(tmp_path):     # № 123
    def check(t, d, c):
        t.write_text(t.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return ORIG_CHECK(t, d, c)
    assert readonly_problems(check, tmp_path) != []


def test_run_refuses_before_any_compute_when_registry_not_ready(monkeypatch, tmp_path):     # № 30
    ev = []
    monkeypatch.setattr(RD.RG, "require", lambda cfg: ev.append("require"))
    monkeypatch.setattr(RD.RG, "check", lambda: ev.append("check") or ["registered_at позже"])
    monkeypatch.setattr(RD, "make_ctx", lambda a: ev.append("ctx") or (_ for _ in ()).throw(AssertionError("счёт до реестра")))
    with pytest.raises(SystemExit, match="реестр S-RCD не готов"):
        RD.run("rcd", write=False, out=tmp_path)
    assert ev == ["require", "check"]


def test_run_refuses_unregistered_id_before_ctx(monkeypatch, tmp_path):
    monkeypatch.setattr(RD, "make_ctx", lambda a: (_ for _ in ()).throw(AssertionError("счёт до реестра")))
    monkeypatch.setattr(RD.RG, "require", lambda cfg: (_ for _ in ()).throw(SystemExit("нет в реестре")))
    with pytest.raises(SystemExit, match="нет в реестре"):
        RD.run("cdf", write=False, out=tmp_path)


def stub_run(monkeypatch, df):
    monkeypatch.setattr(RD.RG, "require", lambda cfg: None)
    monkeypatch.setattr(RD.RG, "check", lambda: [])
    monkeypatch.setattr(RD, "make_ctx", lambda a: object())
    monkeypatch.setattr(RD, "walk", lambda *a, **k: [])
    monkeypatch.setattr(RD, "assemble", lambda *a, **k: df)


def test_run_writes_via_tmp_refuses_different_existing_and_accepts_same(monkeypatch, tmp_path):
    df = pd.DataFrame(dict(d=[1, 2], C=[1.0, 2.0]))
    stub_run(monkeypatch, df)
    RD.run("rcd", write=False, out=tmp_path)
    assert not list(tmp_path.glob("*.parquet"))
    RD.run("rcd", out=tmp_path)
    assert (tmp_path / "rcd.parquet").exists() and not list(tmp_path.glob("*.tmp")) and pd.read_parquet(tmp_path / "rcd.parquet").equals(df)
    RD.run("rcd", out=tmp_path)     # тот же результат — не отказ
    stub_run(monkeypatch, df.assign(C=[1.0, 2.5]))
    with pytest.raises(SystemExit, match="перезапись отказана"):
        RD.run("rcd", out=tmp_path)
    assert pd.read_parquet(tmp_path / "rcd.parquet").equals(df)
