"""Отметка снятого замка в пакете переноса (решение дирижёра 29.10, класс B). Отметка — только во временном каталоге теста.
Мутации: отметка не добавлена в tar / добавлена в SHA256SUMS после их подсчёта (тесты tar и сумм падают); razvernut не проверяет
sha256 предрегистрации в отметке (тест отказа падает)."""
import hashlib
import os
import subprocess
from pathlib import Path

SRV = Path(__file__).resolve().parents[1] / "scripts/server"
PRE = "ab" * 32


def _bash(tmp: Path, code: str, **env) -> subprocess.CompletedProcess:
    (tmp / "obshchee.sh").write_text((SRV / "perenos/obshchee.sh").read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    e = {k: v for k, v in os.environ.items() if not k.startswith("PERENOS_")}
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; . {SRV}/otmetka_paketa.sh; {code}"], cwd=tmp, env={**e, **env},
                          capture_output=True, text=True)


def _tar(tmp: Path, otm: str) -> subprocess.CompletedProcess:
    (tmp / "src/kod/fastbt").mkdir(parents=True)
    (tmp / "src/kod/fastbt/a.py").write_text("x = 1\n")
    code = f"tar -cf kod.tar -C src kod && otmetka_v_tar kod.tar {otm} && gzip -n < kod.tar > kod.tar.gz && tar -tzf kod.tar.gz"
    return _bash(tmp, code)


def test_otmetka_popadaet_v_tar_po_tomu_zhe_puti_i_v_summy_koda(tmp_path):
    otm = tmp_path / "wf_zamok_snyat.json"
    otm.write_text('{"sha256": "%s"}' % PRE)
    r = _tar(tmp_path, str(otm))
    assert r.returncode == 0 and "kod/docs/resheniya/wf_zamok_snyat.json" in r.stdout and "kod/fastbt/a.py" in r.stdout
    (tmp_path / "x").mkdir()
    subprocess.run(["tar", "-xzf", str(tmp_path / "kod.tar.gz"), "-C", str(tmp_path / "x")], check=True)
    sums = _bash(tmp_path, "sha_spisok x/kod").stdout                      # как kod/<sha>.SHA256SUMS на сервере
    assert "./docs/resheniya/wf_zamok_snyat.json" in sums
    assert hashlib.sha256(otm.read_bytes()).hexdigest() in sums


def test_bez_otmetki_tar_bez_nejo(tmp_path):
    r = _tar(tmp_path, str(tmp_path / "net.json"))
    assert r.returncode == 0 and "wf_zamok_snyat" not in r.stdout and "kod/fastbt/a.py" in r.stdout
    (tmp_path / "pusto.json").write_text("")
    (tmp_path / "2").mkdir()
    assert "wf_zamok_snyat" not in _tar(tmp_path / "2", str(tmp_path / "pusto.json")).stdout        # пустая отметка — не отметка


def test_otmetka_v_sha256sums_paketa(tmp_path):
    pak = tmp_path / "pak"
    pak.mkdir()
    (pak / "VERSIYA").write_text("v\n")
    (pak / "wf_zamok_snyat.json").write_text('{"sha256": "%s"}' % PRE)
    s = _bash(tmp_path, "summy_paketa pak").stdout
    assert "./wf_zamok_snyat.json" in s and "./VERSIYA" in s and "SHA256SUMS" not in s
    (pak / "data").mkdir()
    (pak / "data/d.csv").write_text("x")
    assert "data/" not in _bash(tmp_path, "summy_paketa pak").stdout          # data/ — по манифесту


def test_sobrat_kopiruet_otmetku_do_podschyota_summ():
    t = (SRV / "sobrat_perenos.sh").read_text()
    assert t.index('cp "$OTMETKA_WF" "$OUT/wf_zamok_snyat.json"') < t.index('summy_paketa "$OUT"')     # мутация: после -> не в суммах
    assert "otmetka_v_tar" in t


def _kod(tmp: Path, soderzhimoe):
    k = tmp / "kod/ff/docs/resheniya"
    if soderzhimoe is not None:
        k.mkdir(parents=True)
        (k / "wf_zamok_snyat.json").write_text(soderzhimoe)


def test_razvernut_net_otmetki_ne_otkaz(tmp_path):
    _kod(tmp_path, None)
    r = _bash(tmp_path, "otmetka_paketa")
    assert r.returncode == 0 and "отметки wf в пакете нет" in r.stdout


def test_razvernut_pechataet_sha_otmetki_i_predregistracii(tmp_path):
    s = '{"sha256": "%s"}' % PRE
    _kod(tmp_path, s)
    r = _bash(tmp_path, "otmetka_paketa")
    assert r.returncode == 0 and PRE in r.stdout and hashlib.sha256(s.encode()).hexdigest() in r.stdout
    assert _bash(tmp_path, "otmetka_paketa", PERENOS_PREREG_SHA=PRE).returncode == 0


def test_razvernut_otkaz_sha_predregistracii(tmp_path):
    _kod(tmp_path, '{"sha256": "%s"}' % PRE)
    assert _bash(tmp_path, "otmetka_paketa", PERENOS_PREREG_SHA="cd" * 32).returncode != 0         # другой SHA
    (tmp_path / "kod/ff/docs/resheniya/wf_zamok_snyat.json").write_text('{"data": "x"}')
    r = _bash(tmp_path, "otmetka_paketa")
    assert r.returncode != 0 and "FAIL" in r.stdout                                                # нет sha256 внутри
    (tmp_path / "kod/ff/docs/resheniya/wf_zamok_snyat.json").write_text('{"sha256": "%s"}' % PRE)
    (tmp_path / "wf_zamok_snyat.json").write_text('{"sha256": "%s", "x": 1}' % PRE)               # корень пакета ≠ код
    assert _bash(tmp_path, "otmetka_paketa").returncode != 0
