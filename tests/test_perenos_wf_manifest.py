"""Перенос на сервер, лист P-2 (Ж-136): верхняя дата манифеста и акт снятия замка. Файлы данных — синтетика во временном каталоге.
Мутации: `--do` > 2021-12-31 принят без отметки (тест отказа падает); умолчание сдвинуто на 2026 (тест 2022 падает);
акт создаёт отметку или правит zamki.json (тест «ничего не снято» падает)."""
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

SRV = Path(__file__).resolve().parents[1] / "scripts/server"


def _load(name):
    s = importlib.util.spec_from_file_location(name, SRV / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


M, MP, AKT = _load("manifest_perenos"), _load("manifest_proverka"), _load("akt_snyatiya_zamka")
DO = "2026-02-28"


def _csv(p: Path, daty):
    p.write_text("TRADEDATE,CLOSE\n" + "".join(f"{d},1\n" for d in daty))


def _spisok(tmp: Path, shablon="*.csv") -> Path:
    s = tmp / "spisok.tsv"
    s.write_text(f"{shablon}\tt\n")
    return s


def test_umolchanie_2021_data_2026_otvergaetsya(tmp_path):
    _csv(tmp_path / "a.csv", ["2021-12-30", "2026-02-27"])
    r = M.sobrat(str(tmp_path), [("a.csv", "t", "")])
    assert not r["prinyato"] and "2026-02-27" in r["otvergnuto"][0][1]


def test_do_wf_prinimaet_do_2026_02_28_a_dalshe_net(tmp_path):
    _csv(tmp_path / "a.csv", ["2021-12-30", "2026-02-28"])
    _csv(tmp_path / "b.csv", ["2021-12-30", "2026-03-02"])
    r = M.sobrat(str(tmp_path), [("*.csv", "t", "")], pd.Timestamp(DO))
    assert [x[0] for x in r["prinyato"]] == ["a.csv"] and "2026-03-02" in r["otvergnuto"][0][1]


def test_imya_s_godom_posle_do_ne_chitaetsya_a_do_nego_chitaetsya(tmp_path):
    _csv(tmp_path / "ryad_2024.csv", ["2021-12-30"])
    _csv(tmp_path / "ryad_2027.csv", ["2021-12-30"])
    r = M.sobrat(str(tmp_path), [("*.csv", "t", "")], pd.Timestamp(DO))
    assert [x[0] for x in r["prinyato"]] == ["ryad_2024.csv"] and "202x" in r["otvergnuto"][0][1]
    assert not M.sobrat(str(tmp_path), [("ryad_2024.csv", "t", "")])["prinyato"]       # умолчание — по-прежнему отказ


def test_sborshchik_bez_otmetki_otkaz_do_chteniya(tmp_path, capsys):
    _csv(tmp_path / "a.csv", ["2021-12-30"])
    arg = ["--koren", str(tmp_path), "--spisok", str(_spisok(tmp_path)), "--vyhod", str(tmp_path / "o/m.tsv"),
           "--do", DO, "--otmetka", str(tmp_path / "net.json")]
    assert M.main(arg) == 1 and "нет отметки" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()                                     # ничего не записано
    (tmp_path / "net.json").write_text("")                                   # пустая отметка — не отметка
    assert M.main(arg) == 1
    (tmp_path / "net.json").write_text('{"sha256": "x"}')
    assert M.main(arg) == 0 and (tmp_path / "o/m.tsv").exists()
    assert M.main(["--koren", str(tmp_path), "--spisok", str(_spisok(tmp_path)), "--vyhod", str(tmp_path / "o/m2.tsv")]) == 0   # умолчание без отметки


def _mf(tmp: Path, md: str) -> Path:
    data = b"x\n"
    m = tmp / "m.tsv"
    m.write_text("\t".join(MP.SHAPKA) + f"\ndata/a.csv\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t{md}\tx\n")
    return m


def test_proverka_manifesta_do_i_otmetka(tmp_path, capsys):
    m = _mf(tmp_path, "2026-02-28")
    assert MP.main(["--manifest", str(m), "--koren", str(tmp_path / "k")]) == 1                          # умолчание
    assert "> 2021-12-31" in capsys.readouterr().out
    net = ["--manifest", str(m), "--koren", str(tmp_path / "k"), "--do", DO, "--otmetka", str(tmp_path / "net.json")]
    assert MP.main(net) == 1 and "нет отметки" in capsys.readouterr().out
    (tmp_path / "net.json").write_text("{}")
    ok = MP.prochitat(m, MP.granica_wf(DO, tmp_path / "net.json"))
    assert ok[0][3] == "2026-02-28"
    assert MP.prochitat(_mf(tmp_path, "2021-12-31"))                                                      # умолчание: 2021 — можно


def test_proverka_manifesta_data_posle_do_otvergaetsya(tmp_path):
    (tmp_path / "o.json").write_text("{}")
    m = _mf(tmp_path, "2026-03-02")
    try:
        MP.prochitat(m, MP.granica_wf(DO, tmp_path / "o.json"))
    except MP.Otkaz as e:
        assert "2026-03-02" in str(e)
    else:
        raise AssertionError("дата после 2026-02-28 принята")


def _derevo(root: Path):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_akt_pechataet_komandy_i_nichego_ne_snimaet(tmp_path, capsys, monkeypatch):
    prereg = tmp_path / "prereg_v2.md"
    prereg.write_text("предрегистрация v2 (синтетика)\n")
    zamki = tmp_path / ".claude/hooks/zamki.json"
    zamki.parent.mkdir(parents=True)
    zamki.write_text('{"v": 1}')
    do = _derevo(tmp_path), zamki.read_bytes()
    monkeypatch.chdir(tmp_path)
    assert AKT.main([str(prereg)]) == 0
    out = capsys.readouterr().out
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert sha in out and "zamki.json" in out and "wf_zamok_snyat.json" in out and "--do 2026-02-28" in out
    assert (_derevo(tmp_path), zamki.read_bytes()) == do                      # ни файлов, ни правок
    assert not (tmp_path / "docs/resheniya/wf_zamok_snyat.json").exists()


def test_akt_bez_predregistracii_otkaz(tmp_path, capsys):
    assert AKT.main([str(tmp_path / "net.md")]) == 1 and "FAIL akt" in capsys.readouterr().out
    assert AKT.main([]) == 1
