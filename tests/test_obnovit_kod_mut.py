"""Мутации к obnovit_kod.sh / sobrat_perenos.sh (проверка PR #764): каждая подложенная ошибка обязана менять наблюдаемый результат.
Исходный скрипт — зелёный по тому же предикату; мутант — красный."""
import importlib.util
import os
import subprocess
from pathlib import Path

_p = Path(__file__).with_name("test_obnovit_kod.py")
_s = importlib.util.spec_from_file_location("test_obnovit_kod_base", _p)
B = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B)

SH = B.SH
SOBRAT = Path(__file__).resolve().parents[1] / "scripts/server/sobrat_perenos.sh"


def _mut(tmp: Path, a: str, b: str) -> subprocess.CompletedProcess:
    s = (SH / "obnovit_kod.sh").read_text()
    assert a in s, a
    (tmp / "mut.sh").write_text(s.replace(a, b, 1))
    return B._run(tmp, "mut.sh")


def test_mut_a_udalenie_starogo_kod(tmp_path):
    B._paket(tmp_path)
    _mut(tmp_path, "raspakovat_kod\nkod_cel", f"raspakovat_kod\nchmod -R u+w kod/{B.OLD}; rm -rf kod/{B.OLD}\nkod_cel")
    assert not B._staryj_na_meste(tmp_path)


def test_mut_b_bez_sverki_sha256sums(tmp_path):
    B._paket(tmp_path)
    (tmp_path / "requirements_server.txt").write_text("ok==1 \n")    # пакет испорчен после SHA256SUMS
    assert B._run(tmp_path).returncode != 0                           # оригинал отказывает
    r = _mut(tmp_path, 'sha_check "$P" SHA256SUMS >/dev/null ||', 'true ||')
    assert r.returncode == 0 or "SHA256SUMS" not in r.stdout          # мутант пропустил


def test_mut_v_pereozdanie_venv_pri_teh_zhe_pinah(tmp_path):
    B._paket(tmp_path)
    marker = tmp_path / ".venv/marker"
    marker.write_text("m")
    assert B._run(tmp_path).returncode == 0 and marker.exists()      # оригинал .venv не трогает
    B._paket(tmp_path.parent / (tmp_path.name + "_m"))
    t2 = tmp_path.parent / (tmp_path.name + "_m")
    (t2 / ".venv/marker").write_text("m")
    _mut(t2, 'say "3/5 .venv и requirements_server.txt"', 'say "3/5"; rm -rf .venv/marker')
    assert not (t2 / ".venv/marker").exists()


def test_mut_g_zhivoj_progon_ne_otkaz(tmp_path):
    B._paket(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/sobol_train4.pid").write_text(str(os.getpid()))
    assert B._run(tmp_path).returncode != 0
    r = _mut(tmp_path, '! kill -0 "$(cat "$f")" 2>/dev/null ||', 'true ||')
    assert r.returncode == 0                                           # мутант не отказал: предикат «оригинал != 0» их различает


def _spisok(tmp: Path, ne_tot_zhe: bool, skript: str) -> list:
    out, tmpd, prev = tmp / "out", tmp / "tmpd", tmp / "prev"
    for d in (out / "data", tmpd, prev):
        d.mkdir(parents=True, exist_ok=True)
    (out / "manifest_dannyh.tsv").write_text("m1\n")
    (out / "a.sh").write_text("x")
    (out / "data/d.csv").write_text("d")
    (prev / "manifest_dannyh.tsv").write_text("m2\n" if ne_tot_zhe else "m1\n")
    (tmpd / "spisok_dannyh.txt").write_text("d.csv\n")
    s = Path(skript).read_text()
    i = s.index("# список для ./obnovit_kod.sh")
    j = s.index('mv "$TMP/obn"')
    blok = s[i:j] + 'mv "$TMP/obn" "$OUT/spisok_obnovleniya.txt"\n'
    pre = 'say(){ :; }\nsha_file() { shasum -a 256 "$1" | cut -d" " -f1; }\nset -euo pipefail\n'
    subprocess.run(["bash", "-c", pre + blok], check=True, env={**os.environ, "OUT": str(out), "TMP": str(tmpd), "PERENOS_PREDYDUSHCHIJ": str(prev)})
    return (out / "spisok_obnovleniya.txt").read_text().split()


def test_mut_d_dannye_v_spiske_pri_tom_zhe_manifeste(tmp_path):
    ok = _spisok(tmp_path / "ok", False, str(SOBRAT))
    assert "a.sh" in ok and not any(x.startswith("data/") for x in ok)
    izm = _spisok(tmp_path / "izm", True, str(SOBRAT))
    assert "data/d.csv" in izm                                          # манифест изменился — данные едут
    mut = tmp_path / "sobrat_mut.sh"
    s = SOBRAT.read_text()
    a = '!= "$(sha_file "$OUT/manifest_dannyh.tsv")"'
    assert a in s
    mut.write_text(s.replace(a, '== "$(sha_file "$OUT/manifest_dannyh.tsv")"', 1))   # условие перевёрнуто
    bad = _spisok(tmp_path / "mut", False, str(mut))
    assert any(x.startswith("data/") for x in bad)                      # мутант кладёт data/ при том же манифесте


def test_bez_predydushchego_dannye_ne_edut(tmp_path):
    out, tmpd = tmp_path / "out", tmp_path / "tmpd"
    (out / "data").mkdir(parents=True)
    tmpd.mkdir()
    (out / "manifest_dannyh.tsv").write_text("m")
    (out / "data/d.csv").write_text("d")
    (tmpd / "spisok_dannyh.txt").write_text("d.csv\n")
    s = SOBRAT.read_text()
    blok = s[s.index("# список для ./obnovit_kod.sh"):s.index('mv "$TMP/obn"')] + 'mv "$TMP/obn" "$OUT/spisok_obnovleniya.txt"\n'
    env = {k: v for k, v in os.environ.items() if k != "PERENOS_PREDYDUSHCHIJ"}
    subprocess.run(["bash", "-c", 'say(){ :; }\nsha_file(){ :; }\nset -euo pipefail\n' + blok], check=True, env={**env, "OUT": str(out), "TMP": str(tmpd)})
    assert not any(x.startswith("data/") for x in (out / "spisok_obnovleniya.txt").read_text().split())
