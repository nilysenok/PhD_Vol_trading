"""Обновление кода на сервере без передачи данных: scripts/server/perenos/obnovit_kod.sh на tmp-фикстуре (без сети и без реальных данных)."""
import hashlib
import io
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
OLD = "a" * 64
STUB_VERSII = "import sys\nprint('OK versii' if 'ok' in open(sys.argv[1]).read() else 'FAIL')\n"
STUB_MANIFEST = "import os, sys\nopen('manifest_called', 'w').close()\nsys.exit(1 if os.path.exists('data_bad') else 0)\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _paket(root: Path, kod: str = "print(1)\n", manifest: str = "m1\n", reqs: str = "ok==1\n", otmetka_dannye: str = "SAME") -> str:
    """Развёрнутый пакет: старый kod/<OLD> (только чтение), .venv, отметка proverka/OK; новый tar кода в корне. Возвращает sha tar."""
    root.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = kod.encode()
        ti = tarfile.TarInfo("kod/fastbt/x.py")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    tar = buf.getvalue()
    tsha = _sha(tar)
    (root / "kod_new123.tar.gz").write_bytes(tar)
    (root / "VERSIYA").write_text(f"commit_short=new123\ntar_sha256={tsha}\nnumba_threads=1\nnumba_layer=workqueue\nnakladka=\n")
    for n in ("obshchee.sh", "obnovit_kod.sh"):
        (root / n).write_text((SH / n).read_text())
    (root / "manifest_dannyh.tsv").write_text(manifest)
    (root / "requirements_server.txt").write_text(reqs)
    (root / "versii.py").write_text(STUB_VERSII)
    (root / "manifest_proverka.py").write_text(STUB_MANIFEST)
    (root / "data").mkdir(exist_ok=True)
    (root / ".venv/bin").mkdir(parents=True, exist_ok=True)
    py = root / ".venv/bin/python3"
    if not py.exists():
        py.symlink_to(sys.executable)
    old = root / "kod" / OLD / "fastbt"
    old.mkdir(parents=True, exist_ok=True)
    (old / "old.py").write_text("old\n")
    (root / "kod" / OLD / "fastbt/old.py").chmod(stat.S_IRUSR)
    (root / "proverka").mkdir(exist_ok=True)
    msha = _sha(manifest.encode()) if otmetka_dannye == "SAME" else otmetka_dannye
    (root / "proverka/OK").write_text(f"kod={OLD}\ndannye={msha}\netalony=x\nrezhim=1\n")
    sums = "".join(f"{_sha(p.read_bytes())}  ./{p.relative_to(root)}\n" for p in sorted(root.rglob("*"))
                   if p.is_file() and "data" not in p.relative_to(root).parts[:1] and p.name != "SHA256SUMS"
                   and p.relative_to(root).parts[0] not in ("kod", "proverka", ".venv"))
    (root / "SHA256SUMS").write_text(sums)
    return tsha


def _run(root: Path, script: str = "obnovit_kod.sh") -> subprocess.CompletedProcess:
    return subprocess.run(["bash", script], cwd=root, capture_output=True, text=True, env={**os.environ, "PERENOS_SYSTEM_SITE": "1"})


def _staryj_na_meste(root: Path) -> bool:
    return (root / "kod" / OLD / "fastbt/old.py").read_text() == "old\n" if (root / "kod" / OLD / "fastbt/old.py").exists() else False


def test_novyj_kod_raspakovan_staryj_na_meste(tmp_path):
    tsha = _paket(tmp_path)
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "kod" / tsha / "fastbt/x.py").read_text() == "print(1)\n"
    assert not os.access(tmp_path / "kod" / tsha / "fastbt/x.py", os.W_OK)           # только чтение
    assert _staryj_na_meste(tmp_path)
    assert "proverka_yakor.sh" in r.stdout                                          # напоминание о новой отметке
    assert _run(tmp_path).returncode == 0                                           # повтор идемпотентен


def test_dannye_ne_trogayutsya_pri_tom_zhe_manifeste(tmp_path):
    _paket(tmp_path)
    (tmp_path / "data_bad").write_text("x")           # сверка данных упала бы, если бы её вызвали
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout
    assert not (tmp_path / "manifest_called").exists()


def test_izmenivshijsya_manifest_sverka_dannyh_otkaz_pri_nesovpadenii(tmp_path):
    _paket(tmp_path, otmetka_dannye="0" * 64)
    ok = _run(tmp_path)
    assert ok.returncode == 0 and (tmp_path / "manifest_called").exists()
    (tmp_path / "data_bad").write_text("x")
    bad = _run(tmp_path)
    assert bad.returncode != 0 and "данные не совпадают" in bad.stdout


def test_drugoj_requirements_otkaz_s_instrukciej(tmp_path):
    _paket(tmp_path, reqs="novyj==2\n")
    r = _run(tmp_path)
    assert r.returncode != 0 and "razvernut.sh" in r.stdout
    assert [p.name for p in (tmp_path / "kod").iterdir()] == [OLD]                    # ничего не распаковано


def test_zhivoj_progon_otkaz(tmp_path):
    _paket(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/sobol_train4.pid").write_text(str(os.getpid()))
    r = _run(tmp_path)
    assert r.returncode != 0 and "прогон идёт" in r.stdout
    (tmp_path / "logs/sobol_train4.pid").write_text("999999")                          # мёртвый pid не мешает
    assert _run(tmp_path).returncode == 0


def test_isporchennyj_paket_otkaz_po_sha256sums(tmp_path):
    _paket(tmp_path)
    (tmp_path / "requirements_server.txt").write_text("ok==1 \n")
    r = _run(tmp_path)
    assert r.returncode != 0 and "SHA256SUMS" in r.stdout


def test_mutaciya_udalyaet_staryj_kod_lovitsya(tmp_path):
    _paket(tmp_path)
    s = (SH / "obnovit_kod.sh").read_text()
    mut = s.replace("raspakovat_kod\n", f"raspakovat_kod\nchmod -R u+w kod/{OLD}; find kod/{OLD} -delete\n")
    assert mut != s
    (tmp_path / "mut.sh").write_text(mut)
    _run(tmp_path, "mut.sh")
    assert not _staryj_na_meste(tmp_path)             # проверка «старый на месте» из остальных тестов падает на мутации


def test_obnovit_kod_ne_udalyaet_nichego():
    s = (SH / "obnovit_kod.sh").read_text()
    assert "rm " not in s and "-delete" not in s
