"""Л10: драйвер Соболя на кубе v5 (Ж-127, Ж-129 п. 8, Ж-130) на синтетическом оценщике (данных нет). Мутации — test_mut_*."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import fastbt.canon_v5 as cv5
import fastbt.ledger_v2 as lm
import fastbt.progon_sobol as ps

T = 8
# Эталон пути designs/progon, снятый на базе до правки (kanon-v22-pereschet 63b2b369): точки S1…S6 [0, 300) и ledger 2.1 S1, S4 [0, 200).
ETALON_TOCHKI = "834f75d2d7d3cd0884d79b1d1d118a803fc67b92050290f5265cf26f22936277"
ETALON_LEDGER = "5cd56b6a08838aa12e87ca2fdcc1569d7bf9aed3010a1822d33213664fe98628"


class Fake:
    """evaluate_v2 по контракту EvaluatorV2: ряды зависят только от точки и периода; fdr — атрибут, как у EvaluatorV2."""
    fdr = None

    def evaluate_v2(self, pt, period):
        kl = lm.klyuch("S1", pt, period, 0.0004, None)
        rng = np.random.default_rng(int(lm.klyuch_sha(kl, "a", "b")[:8], 16))
        tik = rng.normal(0, 0.01, (17, T)).astype(np.float32).astype(np.float64)
        return {"by_year": {str(y): 0.5 for y in range(2016, 2020)}, "T": T, "mediana_kv": 0.1}, np.nanmean(tik, 0), tik


class FakeT0d(Fake):
    fdr = "t0d"


def sozdat(ocen=Fake, start="2016-01-04"):
    daty = np.arange(np.datetime64(start), np.datetime64(start) + T)
    return lambda sid: ps.Sreda(ocen(), "kod-fake", "vhody-fake", daty, [f"T{i:02d}" for i in range(17)])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sozdat())
    monkeypatch.setattr(ps, "_W", {})


@pytest.fixture
def fake5(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sozdat(FakeT0d))
    monkeypatch.setattr(ps, "_W", {})


def stroki(root, sid):
    out = []
    for p in sorted((Path(root) / sid).glob("p[0-9]*/rows.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines()]
    return out


def sha_kataloga(root: Path) -> str:
    """Все файлы каталога побайтно; из строк — без t_sek, из manifest — без даты создания."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        b = f.read_bytes()
        if f.name == "rows.jsonl":
            b = "\n".join(json.dumps({k: v for k, v in json.loads(x).items() if k != "t_sek"}, sort_keys=True)
                          for x in b.decode().splitlines()).encode()
        elif f.name == "manifest.json":
            b = json.dumps({k: v for k, v in json.loads(b).items() if k != "sozdan"}, sort_keys=True).encode()
        h.update(str(f.relative_to(root)).encode() + b"\0" + b)
    return h.hexdigest()


def test_put_progon_tochki_pobajtno():
    pts = {s: ps.tochki(ps.load_kub(s), 0, 300) for s in ps.SIDS}
    assert hashlib.sha256(json.dumps(pts, sort_keys=True).encode()).hexdigest() == ETALON_TOCHKI


def test_put_progon_ledger_pobajtno(fake, tmp_path):
    ps.schet(["S1", "S4"], 0, 200, "sobol1", tmp_path, 1, False, False, lm.V21)    # 1 процесс: раскладка по p{NN} детерминирована
    assert sha_kataloga(tmp_path) == ETALON_LEDGER


def rz5(period="train4", seed="seed", k3=False, fdr="t0d"):
    return ps.Rezhim("v5", period, seed, k3, fdr)


def args(*argv):
    ap = ps.argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    for k, d in (("--kub", "progon"), ("--period", "train4"), ("--n1", None)):
        ap.add_argument(k, default=d)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k3", action="store_true")
    return ps.rezhim(ap.parse_args(list(argv)), ["S1", "S3"], ap)


def test_v5_kub_chitaetsya_kanon_i_yakorya():
    for sid in ("S1", "S3", "S5"):
        c = ps.load_kub(sid, rz5())
        pts = ps.tochki(c, 0, 256, rz5())
        ya, (ref, sch) = c["yakorya_progon"], cv5.tochki_v5(c, 0, 256)
        assert [i for i, _ in pts] == [i for i, _ in ref] and len(pts) + sch["vybrosheno"] == 256
        assert all(p == {**ya, **r} for (_, p), (_, r) in zip(pts, ref))
        assert all(cv5.canon_v5(sid, {k: v for k, v in p.items() if k not in ya}) is not None for _, p in pts)
        assert ps.kub_put(sid, rz5()).parent.name == "v5" and ps.kub_put(sid).parent.name == "progon"


def klyuchi(root, sid):
    return {r["klyuch_sha"]: r["klyuch"]["period"] for r in stroki(root, sid)}


def v2_otdelno(root_t4, root_v2) -> bool:
    """v2: ключи не пересекаются с train4 той же точки, в ключе и строке — v2, manifest — окно и прогрев В2."""
    t4, v2 = klyuchi(root_t4, "S1"), klyuchi(root_v2, "S1")
    m = json.loads((Path(root_v2) / "S1" / "manifest.json").read_text())
    return (len(v2) > 0 and not set(t4) & set(v2) and set(v2.values()) == {"v2"} and set(t4.values()) == {"train4"}
            and (m["period"], m["okno_s"], m["progrev_s"]) == ("v2", "2015-01-01", "2014-08-26"))


def test_v2_otdelnyj_klyuch_i_mutaciya_v2_v_klyuch_train4(fake5, monkeypatch, tmp_path):
    ps.schet(["S1"], 0, 40, "sobol1_seed1", tmp_path / "t4", 1, False, False, lm.V, rz5())
    ps.schet(["S1"], 0, 40, "sobol1_seed1", tmp_path / "v2", 1, False, False, lm.V, rz5("v2"))
    assert v2_otdelno(tmp_path / "t4", tmp_path / "v2")
    kl = lm.klyuch
    monkeypatch.setattr(lm, "klyuch", lambda sid, pt, period, k, f: kl(sid, pt, "train4", k, f))   # мутация: v2 в ключ train4
    monkeypatch.setattr(ps, "_W", {})
    ps.schet(["S1"], 0, 40, "sobol1_seed1", tmp_path / "m", 1, False, False, lm.V, rz5("v2"))
    assert not v2_otdelno(tmp_path / "t4", tmp_path / "m")


def test_v2_ledger21_otkaz_do_starta():
    ps.proverit_ledger(rz5("v2"), lm.V21)              # Б7: ledger 2.1 считает 2015–2019 — v2 принят; отказ остаётся для окна шире годов
    ps.proverit_ledger(rz5("v2"), lm.V)
    ps.proverit_ledger(rz5(), lm.V21)


def test_v2_katalogi_i_period_ne_iz_kuba():
    assert args("--kub", "v5", "--period", "v2")[1].name == "v2_s1" and args("--kub", "v5")[1].name == "train4_s1"
    assert args()[0] == ps.R0 and args()[1] == ps.LEDGER
    with pytest.raises(SystemExit):
        args("--period", "v2")                                   # designs/progon — только train4
    with pytest.raises(ValueError, match="periody"):
        ps.tochki({**ps.load_kub("S1", rz5()), "periody": ["train4"]}, 0, 4, rz5("v2"))


def test_seed2_drugie_tochki_i_svoj_katalog(fake5, tmp_path):
    c = ps.load_kub("S1", rz5())
    assert ps.tochki(c, 0, 64, rz5()) != ps.tochki(c, 0, 64, rz5(seed="seed2"))
    assert args("--kub", "v5", "--seed", "2")[0].seed == "seed2" and args("--kub", "v5", "--seed", "2")[1].name == "train4_s2"
    ps.schet(["S1"], 0, 20, "sobol1_seed2", tmp_path, 1, False, False, lm.V21, rz5(seed="seed2"))
    m = json.loads((tmp_path / "S1" / "manifest.json").read_text())
    assert (m["sobol_seed"], m["sobol_klyuch"], m["setka"], m["fdr"]) == (20261031, "seed2", "v5", "t0d")


def vne_k3(sid, pts) -> int:
    """Число значений вне к3: оси раздела k3 — из списка, остальные оси куба — yakor (после canon_v5)."""
    c = ps.load_kub(sid, rz5())
    k3, bad = c["k3"], 0
    for _, p in pts:
        for n, g in cv5.osi(c):
            sec = next(s for s in cv5.SEKCII if n in c.get(s, {}))
            dop = k3[n] if n in k3 else [c[sec][n].get("yakor", g[0])]
            bad += p[n] not in dop and p[n] != (cv5.canon_v5(sid, {**p, n: dop[0]}) or {}).get(n)
    return bad


def test_k3_tolko_osi_v4_i_mutaciya_polnyj_kub():
    for sid in ps.SIDS:
        c = ps.load_kub(sid, rz5())
        n = ps.obem_k3(c)
        pts = ps.tochki(c, 0, n + 50, rz5(k3=True))
        assert pts and max(i for i, _ in pts) < n and vne_k3(sid, pts) == 0
        assert ps.n1_po_sid({sid: c}, None, rz5(k3=True)) == {sid: n}
        assert vne_k3(sid, ps.tochki(c, 0, 64, rz5())) > 0           # мутация: k3 считает полный куб
    with pytest.raises(SystemExit):
        args("--kub", "v5", "--k3", "--seed", "2")


def test_fdr_iz_kuba_v_kluche_i_stroke(fake5, tmp_path):
    ps.schet(["S1"], 0, 10, "k3", tmp_path, 1, False, False, lm.V21, rz5(k3=True))
    r = stroki(tmp_path, "S1")
    assert r and {x["fdr"] for x in r} == {"t0d"} and {x["klyuch"]["fdr"] for x in r} == {"t0d"}
    assert {x["istochnik"] for x in r} == {"k3"} and json.loads((tmp_path / "S1" / "manifest.json").read_text())["k3"] is True


def test_fdr_ne_podderzhan_otkaz(fake, tmp_path):
    with pytest.raises(ValueError, match="fdr"):                  # оценщик считает fdr = None, куб — t0d
        ps.schet(["S1"], 0, 5, "sobol1_seed1", tmp_path, 1, False, False, lm.V21, rz5())
    with pytest.raises(ValueError, match="fdr"):                  # режим, которого нет в fdr.py
        ps.tochki({**ps.load_kub("S1", rz5()), "fdr": "t9"}, 0, 4, rz5(fdr="t9"))
    with pytest.raises(ValueError, match="fdr"):                  # fdr прохода ≠ fdr куба
        ps.tochki(ps.load_kub("S1", rz5()), 0, 4, rz5(fdr=None))


def test_eval_v5_ne_slit_otkaz(monkeypatch):
    monkeypatch.setattr(ps, "OCENSHCHIK_V5", ("fastbt.net_takogo_modulya", "EvaluatorV5"))
    with pytest.raises(SystemExit, match="Л9"):
        ps.ocenshchik("S1", None, rz5())


def test_n1_iz_kuba_i_pereopredelenie():
    kubs = {s: ps.load_kub(s, rz5()) for s in ("S1", "S3")}
    assert ps.n1_po_sid(kubs, None, rz5()) == {"S1": 32768, "S3": 32768}
    assert ps.n1_po_sid(kubs, "S1=65536", rz5()) == {"S1": 65536, "S3": 32768}
    assert ps.n1_po_sid(kubs, "65536", rz5()) == {"S1": 65536, "S3": 65536}
    with pytest.raises(ValueError):
        ps.n1_po_sid(kubs, "S9=1", rz5())


def test_v5_rezyum_i_n1_po_sid(fake5, tmp_path):
    a = ps.schet(["S1", "S3"], 0, {"S1": 120, "S3": 60}, "sobol1_seed1", tmp_path, 1, False, False, lm.V21, rz5())
    b = ps.schet(["S1", "S3"], 0, {"S1": 120, "S3": 60}, "sobol1_seed1", tmp_path, 1, False, False, lm.V21, rz5())
    assert 0 < a["S1"]["n"] <= 120 and 0 < a["S3"]["n"] <= 60 and b["S1"]["n"] == b["S3"]["n"] == 0
    assert max(r["sobol_i"] for r in stroki(tmp_path, "S1")) < 120 and max(r["sobol_i"] for r in stroki(tmp_path, "S3")) < 60
