"""S-A-IMPL-3 (тестировщик), регистратор: 7 строк разом с одной меткой, реестр только дополняется, отказ при parquet в A (реестр позже прогноза, № 30),
проверка по id а не по позиции (№ 164), `--check` только читает (№ 123). Работа — на копиях во tmp; реальные файлы охраняет autouse-проверка."""
import hashlib
import json
import os
import shutil
import time

import pandas as pd
import pytest

from models_v2.s18 import a_reg as R
from s18_a_guard import A_DIR, GUARDED, ROOT, guard_real_files  # noqa: F401  (autouse)

NOW = "11.11.2026 17:30:00"


@pytest.fixture
def env(tmp_path):
    t, c, s = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "S_A_SPEC.md"
    # копия реестра «до регистрации»: реальные строки S-A (если уже записаны) и ключ S.A.registered отброшены; остальное побайтно
    lines = GUARDED[0].read_text(encoding="utf-8").splitlines(keepends=True)
    t.write_text("".join(ln for ln in lines if not ln.startswith("S-A")), encoding="utf-8")
    cp = json.loads(GUARDED[1].read_text(encoding="utf-8"))
    cp.pop(R.KEY, None)
    cp.pop("S.A.eval", None)   # ключ стадии позже регистрации не должен влиять на копию «до прогноза»
    c.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    shutil.copy(ROOT / "docs/otchety/harj_struct/S_A_SPEC.md", s)
    (tmp_path / "A").mkdir()
    return dict(trials=t, ckpt=c, spec=s, a_dir=tmp_path / "A")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snap(env):
    return {k: sha(env[k]) for k in ("trials", "ckpt")} | {p.name: sha(p) for p in env["a_dir"].glob("*")}


def reg(env, now=NOW):
    return R.register(now, trials=env["trials"], a_dir=env["a_dir"], ckpt=env["ckpt"], spec=env["spec"])


def chk(env):
    return R.check(trials=env["trials"], a_dir=env["a_dir"], ckpt=env["ckpt"])


def test_constants_rows_and_statuses_match_spec_section_7():
    assert R.IDS == ("S-A0", "S-A1", "S-A2", "S-A2p", "S-A3", "S-A2s", "S-A2g") and (R.N_PROGRAM, R.N_WIDE) == (831, 3367)
    st = {r[0]: r[4] for r in R.ROWS}
    assert st["S-A0"] == st["S-A1"] == st["S-A2"] == "зарегистрирована" and st["S-A2p"] == "контроль"
    assert all(st[i].startswith("не запускается в пилоте") for i in ("S-A3", "S-A2s", "S-A2g"))
    assert R.KEY == "S.A.registered" and R.N_PROGRAM == 827 + 4 and R.N_WIDE == 3359 + 2 * 4


def test_register_appends_seven_rows_one_timestamp_and_only_key_in_checkpoint(env):
    t0, c0 = env["trials"].read_text(encoding="utf-8"), json.loads(env["ckpt"].read_text(encoding="utf-8"))
    assert chk(env) != []                                      # до регистрации — замечания
    assert reg(env) == 7
    t1 = env["trials"].read_text(encoding="utf-8")
    assert t1.startswith(t0)                                   # старые строки побайтно, только дописано
    new = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False).tail(7)
    assert list(new["id"]) == list(R.IDS) and set(new["registered_at"]) == {NOW}
    c1 = json.loads(env["ckpt"].read_text(encoding="utf-8"))
    k = c1.pop(R.KEY)
    assert c1 == c0 and k["n"] == 831 and k["n_wide"] == 3367 and k["rows"] == 7 and k["registered_at"] == NOW
    assert k["spec_sha256"] == sha(env["spec"])
    assert chk(env) == []


def test_second_registration_refused_without_writing(env):
    reg(env)
    before = snap(env)
    with pytest.raises(SystemExit):
        reg(env, "11.11.2026 18:00:00")
    assert snap(env) == before


def test_registry_after_forecast_refused_without_writing(env):
    """Мутация № 30: регистратор не отказывает, когда в A уже есть parquet."""
    (env["a_dir"] / "a0.parquet").write_bytes(b"x")
    before = snap(env)
    with pytest.raises(SystemExit):
        reg(env)
    assert snap(env) == before


def test_mutation_register_ignores_existing_parquet_caught(env, monkeypatch):
    p = env["a_dir"] / "a2.parquet"
    p.write_bytes(b"x")
    os.utime(p, (1.7e9, 1.7e9))                                 # 2023: заведомо раньше registered_at
    monkeypatch.setattr(R, "_parquet", lambda a_dir: [])
    assert reg(env) == 7                                        # мутант регистрирует поздно, без отказа
    monkeypatch.undo()
    assert any("mtime" in b for b in chk(env))                  # честный сторож видит позднюю запись


def test_check_registered_after_parquet_mtime_is_flagged(env):
    reg(env)
    p = env["a_dir"] / "a0.parquet"
    p.write_bytes(b"x")
    os.utime(p, (time.time() - 10 ** 7, time.time() - 10 ** 7))   # parquet «старше» регистрации (≈ 2026-07)
    assert any("mtime" in b for b in chk(env))
    os.utime(p, (time.time() + 10 ** 6, time.time() + 10 ** 6))   # после регистрации — норма
    assert chk(env) == []


def test_check_is_by_id_not_by_position(env):
    """После регистрации строка другого листа дописана в конец и 7 строк S-A не последние: check всё равно ok."""
    reg(env)
    df = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)
    extra = df.iloc[[0]].copy()
    extra["id"] = "Z-OTHER"
    with env["trials"].open("a", encoding="utf-8", newline="") as fh:
        fh.write(extra.to_csv(index=False, header=False, lineterminator="\n"))
    assert chk(env) == []
    df2 = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)
    assert list(df2["id"].tail(1)) == ["Z-OTHER"] and list(df2["id"].tail(8).head(7)) == list(R.IDS)


def test_mutation_check_by_position_caught_by_appended_foreign_row(env, monkeypatch):
    """Мутант: сторож берёт «последние 7 строк» вместо выборки по id — после строки другого листа в конце он ложно падает."""
    def by_position(trials=R.TRIALS, a_dir=R.A_DIR, ckpt=R.CKPT):
        last = R._read(trials).tail(7)
        return [] if list(last["id"]) == list(R.IDS) else ["не последние 7 строк"]
    reg(env)
    assert by_position(env["trials"]) == []                      # до дописи позиционный мутант сходится с честным
    df = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)
    extra = df.iloc[[0]].copy()
    extra["id"] = "Z-OTHER"
    with env["trials"].open("a", encoding="utf-8", newline="") as fh:
        fh.write(extra.to_csv(index=False, header=False, lineterminator="\n"))
    assert by_position(env["trials"]) != [] and chk(env) == []   # мутант расходится с честным — тест по id ловит его


def test_check_flags_missing_duplicate_wrong_time_and_wrong_n(env):
    reg(env)
    df = pd.read_csv(env["trials"], comment="#", dtype=str, keep_default_na=False)
    keep = df[df["id"] != "S-A2s"]
    keep.to_csv(env["trials"], index=False, lineterminator="\n")
    assert chk(env) != []                                        # 6 из 7
    pd.concat([df, df[df["id"] == "S-A2"]]).to_csv(env["trials"], index=False, lineterminator="\n")
    assert chk(env) != []                                        # дубль
    d3 = df.copy()
    d3.loc[d3["id"] == "S-A0", "registered_at"] = "11.11.2026 17:31:00"
    d3.to_csv(env["trials"], index=False, lineterminator="\n")
    assert chk(env) != []                                        # две метки
    df.to_csv(env["trials"], index=False, lineterminator="\n")
    c = json.loads(env["ckpt"].read_text(encoding="utf-8"))
    for field, bad in (("n", 827), ("n_wide", 3359), ("rows", 4)):
        c2 = json.loads(json.dumps(c))
        c2[R.KEY][field] = bad
        env["ckpt"].write_text(json.dumps(c2, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        assert chk(env) != [], field


def test_check_reads_only_hashes_before_after_and_writing_check_is_caught(env, monkeypatch):
    reg(env)
    p = env["a_dir"] / "a0.parquet"
    p.write_bytes(b"x")
    os.utime(p, (time.time() + 10 ** 6, time.time() + 10 ** 6))
    before = snap(env)
    assert chk(env) == []
    assert snap(env) == before
    orig = R.check

    def writing_check(trials=R.TRIALS, a_dir=R.A_DIR, ckpt=R.CKPT):
        out = orig(trials, a_dir, ckpt)
        trials.write_text(trials.read_text(encoding="utf-8") + "\n", encoding="utf-8")       # мутант № 123
        return out
    monkeypatch.setattr(R, "check", writing_check)
    chk(env)
    assert snap(env) != before                                   # хэш-сравнение ловит запись


def test_require_refuses_unknown_id_and_accepts_registered(env):
    with pytest.raises(SystemExit):
        R.require("S-A2", trials=env["trials"])
    reg(env)
    for i in ("S-A0", "S-A1", "S-A2", "S-A2p"):
        R.require(i, trials=env["trials"])
    with pytest.raises(SystemExit):
        R.require("S-A9", trials=env["trials"])


def test_checkpoint_with_foreign_format_refused_before_any_write(env):
    c = json.loads(env["ckpt"].read_text(encoding="utf-8"))
    env["ckpt"].write_text(json.dumps(c, indent=4), encoding="utf-8")
    before = snap(env)
    with pytest.raises(SystemExit):
        reg(env)
    assert snap(env) == before


def test_real_files_untouched_and_real_a_dir_state_is_read_only():
    """Реальный реестр: check только читает (хэши до и после равны); autouse-охрана дополнительно сравнивает содержимое после теста."""
    before = {str(p): sha(p) for p in GUARDED}
    R.check()
    assert {str(p): sha(p) for p in GUARDED} == before
    assert A_DIR == R.A_DIR
