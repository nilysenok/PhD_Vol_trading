"""S-C-IMPL-3 (тестировщик), регистратор: 9 строк S-C разом с одной меткой, реестр только дополняется, отказ при parquet в CM (реестр позже прогноза, № 30),
проверка по id, а не по позиции (№ 164/166), `--check` только читает (№ 123). Работа — на копиях во tmp; реальные файлы охраняет autouse-проверка.
Ожидаемое API — по образцу `a_reg`: ROWS, IDS, KEY, N_PROGRAM, N_WIDE, CM_DIR, TRIALS, CKPT, register(now, trials, cm_dir, ckpt, spec), check(trials, cm_dir, ckpt), require, _parquet, _read."""
import hashlib
import json
import os
import shutil
import time

import pandas as pd
import pytest

from models_v2.s18 import cm_reg as R
from s18_cm_guard import CM_DIR, GUARDED, ROOT, guard_real_files  # noqa: F401  (autouse)

NOW = "11.11.2026 19:30:00"
RUN = ("S-CD", "S-CK", "S-CKA", "S-C7", "S-CKL")
NOT_RUN = ("S-C2f", "S-C7i", "S-CDx", "S-CDw")


@pytest.fixture
def env(tmp_path):
    t, c, s = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "S_C_SPEC.md"
    lines = GUARDED[0].read_text(encoding="utf-8").splitlines(keepends=True)
    t.write_text("".join(ln for ln in lines if not ln.startswith("S-C")), encoding="utf-8")
    cp = json.loads(GUARDED[1].read_text(encoding="utf-8"))
    cp.pop(R.KEY, None)
    c.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    shutil.copy(ROOT / "docs/otchety/harj_struct/S_C_SPEC.md", s)
    (tmp_path / "CM").mkdir()
    return dict(trials=t, ckpt=c, spec=s, cm_dir=tmp_path / "CM", base_n=base("n", cp), base_w=base("n_wide", cp))


def base(field, cp):
    """N регистратора = максимум по ключам S.*.registered (кроме своего) — как в API cm_reg."""
    return max(v[field] for k, v in cp.items() if k.startswith("S.") and k.endswith(".registered") and k != R.KEY)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snap(env):
    return {k: sha(env[k]) for k in ("trials", "ckpt")} | {p.name: sha(p) for p in env["cm_dir"].glob("*")}


def reg(env, now=NOW):
    return R.register(now, trials=env["trials"], cm_dir=env["cm_dir"], ckpt=env["ckpt"], spec=env["spec"])


def chk(env):
    return R.check(trials=env["trials"], cm_dir=env["cm_dir"], ckpt=env["ckpt"])


def add_foreign_row(env):
    df = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)
    extra = df.iloc[[0]].copy()
    extra["id"] = "Z-OTHER"
    with env["trials"].open("a", encoding="utf-8", newline="") as fh:
        fh.write(extra.to_csv(index=False, header=False, lineterminator="\n"))


def test_constants_rows_and_statuses_match_spec_section_6():
    assert R.IDS == RUN + NOT_RUN and len(R.ROWS) == 9 and R.KEY == "S.C.registered"
    assert R.N_PROGRAM == 831 + 5 == 836 and R.N_WIDE == 3367 + 2 * 5 == 3377
    st = {r[0]: r[4] for r in R.ROWS}
    assert st["S-CK"] == st["S-CKA"] == "зарегистрирована"
    assert st["S-CD"] == st["S-C7"] == st["S-CKL"] == "контроль"
    assert all(st[i].startswith("не запускается") for i in NOT_RUN)


def test_register_appends_nine_rows_one_timestamp_and_only_key_in_checkpoint(env):
    t0, c0 = env["trials"].read_text(encoding="utf-8"), json.loads(env["ckpt"].read_text(encoding="utf-8"))
    assert chk(env) != []
    assert reg(env) == 9
    t1 = env["trials"].read_text(encoding="utf-8")
    assert t1.startswith(t0)                                   # прочие строки побайтно, только дописано
    new = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False).tail(9)
    assert list(new["id"]) == list(R.IDS) and set(new["registered_at"]) == {NOW}
    c1 = json.loads(env["ckpt"].read_text(encoding="utf-8"))
    k = c1.pop(R.KEY)
    assert c1 == c0 and k["rows"] == 9 and k["registered_at"] == NOW and k["spec_sha256"] == sha(env["spec"])
    assert k["n"] == env["base_n"] + 5 and k["n_wide"] == env["base_w"] + 10   # N берётся из файла: текущее + 5 / + 10
    assert chk(env) == []


def test_second_registration_refused_without_writing(env):
    reg(env)
    before = snap(env)
    with pytest.raises(SystemExit):
        reg(env, "11.11.2026 20:00:00")
    assert snap(env) == before


def test_registry_after_forecast_refused_without_writing(env):
    """Мутация № 30: регистратор не отказывает, когда в CM уже есть parquet."""
    (env["cm_dir"] / "ck.parquet").write_bytes(b"x")
    before = snap(env)
    with pytest.raises(SystemExit):
        reg(env)
    assert snap(env) == before


def test_mutation_register_ignores_existing_parquet_caught(env, monkeypatch):
    p = env["cm_dir"] / "ck.parquet"
    p.write_bytes(b"x")
    os.utime(p, (1.7e9, 1.7e9))
    monkeypatch.setattr(R, "_parquet", lambda cm_dir: [])
    assert reg(env) == 9                                        # мутант регистрирует поздно, без отказа
    monkeypatch.undo()
    assert any("mtime" in b for b in chk(env))                  # честный сторож видит позднюю запись


def test_check_registered_after_parquet_mtime_is_flagged(env):
    reg(env)
    p = env["cm_dir"] / "cd.parquet"
    p.write_bytes(b"x")
    os.utime(p, (time.time() - 10 ** 7, time.time() - 10 ** 7))
    assert any("mtime" in b for b in chk(env))
    os.utime(p, (time.time() + 10 ** 6, time.time() + 10 ** 6))
    assert chk(env) == []


def test_check_is_by_id_not_by_position(env):
    reg(env)
    add_foreign_row(env)
    assert chk(env) == []
    ids = list(pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)["id"])
    assert ids[-1] == "Z-OTHER" and ids[-10:-1] == list(R.IDS)


def test_mutation_check_by_position_caught_by_appended_foreign_row(env):
    def by_position(trials):
        return [] if list(R._read(trials).tail(9)["id"]) == list(R.IDS) else ["не последние 9 строк"]
    reg(env)
    assert by_position(env["trials"]) == []
    add_foreign_row(env)
    assert by_position(env["trials"]) != [] and chk(env) == []


def test_check_flags_missing_duplicate_two_stamps_and_wrong_n(env):
    reg(env)
    df = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)
    df[df["id"] != "S-CKL"].to_csv(env["trials"], index=False, lineterminator="\n")
    assert chk(env) != []                                        # 8 из 9
    pd.concat([df, df[df["id"] == "S-CK"]]).to_csv(env["trials"], index=False, lineterminator="\n")
    assert chk(env) != []                                        # дубль
    d3 = df.copy()
    d3.loc[d3["id"] == "S-CD", "registered_at"] = "11.11.2026 19:31:00"
    d3.to_csv(env["trials"], index=False, lineterminator="\n")
    assert chk(env) != []                                        # две метки
    df.to_csv(env["trials"], index=False, lineterminator="\n")
    c = json.loads(env["ckpt"].read_text(encoding="utf-8"))
    for field, bad in (("n", c[R.KEY]["n"] - 5), ("n_wide", c[R.KEY]["n_wide"] - 10), ("rows", 5)):
        c2 = json.loads(json.dumps(c))
        c2[R.KEY][field] = bad
        env["ckpt"].write_text(json.dumps(c2, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        assert chk(env) != [], field


def test_check_reads_only_hashes_before_after_and_writing_check_is_caught(env, monkeypatch):
    reg(env)
    p = env["cm_dir"] / "ck.parquet"
    p.write_bytes(b"x")
    os.utime(p, (time.time() + 10 ** 6, time.time() + 10 ** 6))
    before = snap(env)
    assert chk(env) == [] and snap(env) == before
    orig = R.check

    def writing_check(trials=R.TRIALS, cm_dir=R.CM_DIR, ckpt=R.CKPT):
        out = orig(trials, cm_dir, ckpt)
        trials.write_text(trials.read_text(encoding="utf-8") + "\n", encoding="utf-8")       # мутант № 123
        return out
    monkeypatch.setattr(R, "check", writing_check)
    chk(env)
    assert snap(env) != before


def test_require_refuses_unknown_id_and_accepts_registered(env):
    with pytest.raises(SystemExit):
        R.require("S-CK", trials=env["trials"])
    reg(env)
    for i in RUN:
        R.require(i, trials=env["trials"])
    with pytest.raises(SystemExit):
        R.require("S-C9", trials=env["trials"])


def test_checkpoint_with_foreign_format_refused_before_any_write(env):
    c = json.loads(env["ckpt"].read_text(encoding="utf-8"))
    env["ckpt"].write_text(json.dumps(c, indent=4), encoding="utf-8")
    before = snap(env)
    with pytest.raises(SystemExit):
        reg(env)
    assert snap(env) == before


def test_real_files_untouched_by_check():
    before = {str(p): sha(p) for p in GUARDED}
    R.check()
    assert {str(p): sha(p) for p in GUARDED} == before
    assert CM_DIR == R.CM_DIR
