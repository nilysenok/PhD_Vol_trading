"""Тестировщик PR #723: мутации, которые тесты автора не ловили.
(а) обрезка имени режет `div_` у свода без даты; (б) обрезка не режет дату у справочников с датой в начале (путь usech(), не usech_csv);
(д) правило усечения свода смотрит одну дату (record_date или announce_pit) вместо обеих — на реальном своде отброшено 0, не видно;
(е) manifest_proverka принимает свод с другим sha; (г) `--kub v5` требует свод (ось дивидендов есть только в v6)."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRV = REPO / "scripts/server"


def _mod(imya):
    s = importlib.util.spec_from_file_location(imya, SRV / f"{imya}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


U, MPR = _mod("usech_2021"), _mod("manifest_proverka")
SVOD = "docs/issledovaniya/div_sverka_svod.csv"
SHAPKA = "ticker,record_date,announce_pit,dps_rub,godna\n"


def _zapisat(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _usech(tmp_path: Path, svod_text: str):
    koren, vyhod = tmp_path / "koren", tmp_path / "koren/data/server_2021"
    _zapisat(koren / SVOD, svod_text)
    _zapisat(koren / "docs/issledovaniya/2026-10-27_sektory.csv", "ticker,sektor,accessed\nA,x,2026-01-01\n")
    zap = U.usech(koren, vyhod)
    return vyhod / "docs/issledovaniya", {z["rel"]: z for z in zap}


def test_usech_imena_svod_ne_rezhetsya_sektory_rezhutsya(tmp_path):
    papka, zap = _usech(tmp_path, SHAPKA + "A,2021-06-01,2021-05-01,1.0,1\n")
    assert sorted(p.name for p in papka.iterdir()) == ["div_sverka_svod.csv", "sektory.csv"], "(а)/(б): имена после усечения"
    assert zap[SVOD]["dst"] == papka / "div_sverka_svod.csv"
    assert zap["docs/issledovaniya/2026-10-27_sektory.csv"]["dst"] == papka / "sektory.csv"


def test_usech_svoda_obe_daty_dolzhny_byt_do_2022(tmp_path):
    papka, zap = _usech(tmp_path, SHAPKA + "OK,2021-12-30,2021-12-01,1.0,1\n"
                        "REC2022,2022-01-10,2021-12-20,1.0,1\n"            # отсечка в 2022, объявлено в 2021 → нет
                        "ANN2022,2021-12-20,2022-01-05,1.0,1\n")           # отсечка до 2022, объявлено в 2022 → нет (point-in-time)
    tikery = [ln.split(",")[0] for ln in (papka / "div_sverka_svod.csv").read_text(encoding="utf-8").splitlines()[1:]]
    assert tikery == ["OK"] and zap[SVOD]["st"].otbrosheno == 2


def test_pravilo_svoda_dve_daty():
    r = [p for p in U.PRAVILA if p[0] == SVOD]
    assert len(r) == 1 and r[0][1] == "csv" and set(r[0][2]) == {"record_date", "announce_pit"}


def _stroka_manifesta(fajl: Path, put: str) -> str:
    h = hashlib.sha256(fajl.read_bytes()).hexdigest()
    return f"{put}\t{h}\t{fajl.stat().st_size}\t2021-12-31\tx\t{SVOD}\t{h}\n"


def test_manifest_proverka_svod_sha_ne_sovpadaet_otkaz(tmp_path):
    put = "data/server_2021/" + SVOD
    koren = tmp_path / "perenos"
    fajl = koren / SVOD                      # 7 полей: на сервере файл лежит по istochnik (путь, по которому читает код)
    _zapisat(fajl, (REPO / SVOD).read_text(encoding="utf-8"))
    man = _zapisat(tmp_path / "m.tsv", "put\tsha256\tbajty\tmax_data\tchitatel\tistochnik\tsha256_istochnika\n"
                   + _stroka_manifesta(REPO / SVOD, put))
    assert MPR.main(["--manifest", str(man), "--koren", str(koren)]) == 0                    # контроль: честный свод проходит
    text = fajl.read_text(encoding="utf-8")
    i = next(k for k, c in enumerate(text) if c.isdigit())
    fajl.write_text(text[:i] + str((int(text[i]) + 1) % 10) + text[i + 1:], encoding="utf-8")   # тот же размер, другой sha
    assert fajl.stat().st_size == (REPO / SVOD).stat().st_size
    assert MPR.main(["--manifest", str(man), "--koren", str(koren)]) == 1


def _argv(kub):
    return argparse.Namespace(rol=None, t_vyhod=None, kub=[kub], k3=False, n1=None, zamorozka=None, strategii="S1", seed=None,
                              period="train4")


def test_v5_ne_trebuet_svod_a_v6_trebuet(monkeypatch):
    from fastbt import progon_server, sobytiya_e1
    vyzovy = []
    monkeypatch.setattr(sobytiya_e1, "trebovat_svod", lambda *a, **k: vyzovy.append(1) or (_ for _ in ()).throw(SystemExit("svod")))
    with pytest.raises(SystemExit, match="svod"):
        progon_server.kuby_iz_argv(_argv("v6"), argparse.ArgumentParser())
    assert vyzovy == [1], "v6 обязан вызвать trebovat_svod на старте"
    vyzovy.clear()
    try:
        progon_server.kuby_iz_argv(_argv("v5"), argparse.ArgumentParser())
    except SystemExit as e:
        assert str(e) != "svod", "(г): v5 не должен требовать свод"
    except Exception:
        pass                                                   # остальное тело v5 здесь не проверяется
    assert vyzovy == [], "(г): v5 вызвал trebovat_svod"
