"""PR 3: путь замка и реестра от корня основной копии (git common-dir); привязка заморозки к sha списка кандидатов.
Мутации — в конце файла (подложенная ошибка -> детектор обязан упасть)."""
import os
import subprocess

from fastbt import sweep_val as sv
from fastbt.sweep_val_dry import make_factory, make_meta, make_spec

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True, env=GIT_ENV)


def repo_s_worktree(tmp):
    main, wt = tmp / "main", tmp / "wt"
    main.mkdir()
    git(main, "init", "-q")
    (main / "f").write_text("x")
    git(main, "add", "f")
    git(main, "commit", "-qm", "i")
    git(main, "worktree", "add", "-q", str(wt))
    return main.resolve(), wt.resolve()


def ok_puti_ot_common_dir(tmp) -> bool:
    main, wt = repo_s_worktree(tmp)
    P = sv.Puti.default(wt)
    return bool(P.lock == main / "data" / "fastbt" / "VALIDATION_A_OPENED.json"
                and P.reestr == main / "docs" / "otbor" / "validation_reestr.csv" and P.zhurnal.is_relative_to(main)
                and not P.lock.is_relative_to(wt))


def ok_otkaz_posle_pravki_spiska(tmp) -> bool:
    """Список правили после заморозки и перевыпустили «да» с новым sha: скрипт всё равно отказывает."""
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    meta2 = meta[:-1]
    stroka = f"да, открыть validation A′, sha {sv.sha_kandidatov(meta2)}"
    P.zhurnal.write_text(f"### Ж-099 · 28.10.2026, 09:30\n{stroka}\n", "utf-8")
    spec["da_vladeltsa"]["stroka"] = stroka
    reads = []
    try:
        sv.run(P, spec, meta2, make_factory(tmp, lambda: reads.append(1)), reps_lw=100, reps_k2=5)
    except sv.Stop:
        return bool(not reads and not P.lock.exists() and not P.reestr.exists())
    return False


def test_puti_ot_common_dir_a_ne_ot_worktree(tmp_path):
    assert ok_puti_ot_common_dir(tmp_path)


def test_otkaz_pri_pravke_spiska_posle_zamorozki_dazhe_s_novym_da(tmp_path):
    assert ok_otkaz_posle_pravki_spiska(tmp_path)


def test_mut_koren_ot_rabochey_kopii(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "koren_repo", lambda root=None: sv.Path(root))
    assert not ok_puti_ot_common_dir(tmp_path)


def test_mut_zamorozka_ne_privyazana_k_sha(monkeypatch, tmp_path):
    real = sv.check_da

    def check(spec, sha, P):
        real({**spec, "zamorozheno_sha": sha}, sha, P)
    monkeypatch.setattr(sv, "check_da", check)
    assert not ok_otkaz_posle_pravki_spiska(tmp_path)
