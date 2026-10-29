"""Б7: ledger читает и пишет период В2 (2015–2019, Ж-113) — синтетика с известным ответом, данных и сети нет.
Мутация (mutations.md № 15 по духу, граница периода): год 2015 отрезан (годы ledger v2 = train4) — G1, G3, G4 падают."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_progon_server import Fake5, fake, rz5  # noqa: F401  (fake — фикстура)

import fastbt.ledger_v2 as L
import fastbt.progon_sobol as ps
import fastbt.svod_server_v5 as SVOD
from fastbt import stats

_sp = importlib.util.spec_from_file_location("svodka_ledger", Path(__file__).resolve().parents[1] / "scripts/server/svodka_ledger.py")
SV = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(SV)

D2 = np.arange(np.datetime64("2015-01-01"), np.datetime64("2020-01-01"))
D2 = D2[np.is_busday(D2)]
D4 = D2[D2 >= np.datetime64("2016-01-01")]
MF = {"kod": "K", "vhody": "V"}


def ryady(T, seed):
    g = np.random.default_rng(seed)
    p = g.normal(0, 1e-3, T)
    return p, p[None, :] + g.normal(0, 1e-3, (17, T))


def row(sha, v=L.V21):
    return {"v": v, "klyuch_sha": sha, "status": "ok", "kod": "K", "vhody": "V",
            "by_year": {"2015": 1.0, "2019": 1.0}, "by_quarter": {"2015Q1": 1.0}}


def sh_goda(x, daty, y):
    m = daty.astype("datetime64[Y]").astype(int) + 1970 <= y
    return x[m].mean() / x[m].std(ddof=1) * np.sqrt(252)


def test_g1_zapis_chtenie_2015(tmp_path):
    p, tk = ryady(D2.size, 1)
    w = L.Pisatel21(tmp_path, "S1", 0, {**MF, "T": D2.size, "period": "v2"}, daty=D2)
    w.add(row("a"), p, tk)
    w.close()
    ch = L.Chitatel(tmp_path, "S1")
    assert ch.v == L.V21 and ch.gody == L.GODY_V2 and ch.manifest["nbars"][:4] == L.nbars_kvartalov(D2, L.GODY_V2)[:4].tolist()
    assert len(ch.manifest["nbars"]) == 20 and sum(ch.manifest["nbars"]) == D2.size and ch.manifest["nbars"][0] > 0   # 2015Q1 есть
    assert ch.summy().shape == (1, 20, stats.NF) and ch.summy(sub="tik").shape == (1, 17, 20, stats.NF)
    assert ch.rows()[0]["by_year"]["2015"] == 1.0                       # by_year 2015 принят читателем


def test_g3_okno_2015_izvestnyj_otvet(tmp_path):
    p, tk = ryady(D2.size, 2)
    w = L.Pisatel21(tmp_path, "S1", 0, {**MF, "T": D2.size, "period": "v2"}, daty=D2)
    w.add(row("a"), p, tk)
    w.close()
    ch = L.Chitatel(tmp_path, "S1")
    for t in (2016, 2017, 2019, 2020):
        assert ch.okno(t)[0][0] == pytest.approx(sh_goda(p, D2, t - 1), abs=1e-12)     # окно [2015-01-01; 31.12(t−1)]
        assert ch.okno(t, sub="tik")[0][0][3] == pytest.approx(sh_goda(tk[3], D2, t - 1), abs=1e-12)
    with pytest.raises(ValueError):
        ch.okno(2015)                                                    # окна, кончающегося в 2014, нет


def test_g1_train4_otkazyvaet_2015_kak_prezhde(tmp_path):
    with pytest.raises(ValueError, match="вне 2016"):
        L.Pisatel21(tmp_path, "S1", 0, {**MF, "T": D2.size, "period": "train4"}, daty=D2)
    d = tmp_path / "S2"
    w = L.Pisatel21(tmp_path, "S2", 0, {**MF, "T": D4.size}, daty=D4)
    w.add(row("b") | {"by_year": {"2016": 1.0}}, *ryady(D4.size, 3))
    w.close()
    r = json.loads((d / "p00/rows.jsonl").read_text().splitlines()[0])
    r["by_year"] = {"2015": 1.0}
    (d / "p00/rows.jsonl").write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="вне 2016"):                    # Chitatel train4 отвергает by_year 2015
        L.Chitatel(tmp_path, "S2")


def test_g2_train4_pobaytno_prezhniy(tmp_path):
    """Эталон: sha256 шардов, снятый со старого кода (origin/kanon-v22-pereschet до Б7), seed 7, daty 2016–2019."""
    g = np.random.default_rng(7)
    p = g.normal(0, 1e-3, D4.size)
    tk = p[None, :] + g.normal(0, 1e-3, (17, D4.size))
    w = L.Pisatel21(tmp_path, "S1", 0, {"T": D4.size, "kod": "K", "vhody": "V", "period": "train4"}, daty=D4)
    w.add(row("a") | {"by_year": {}, "by_quarter": {}}, p, tk)
    w.close()
    sha = {s: hashlib.sha256((tmp_path / "S1/p00" / s / "00000.npy").read_bytes()).hexdigest() for s in ("port", "tik")}
    assert sha == {"port": "5e7ffdc94d8b50c791c1d666bab1480ff611d7617cdc396cff9cd310107da44e",
                   "tik": "1371e90f340848b47444b3c02ce9883a3ee4517c358594af560a1c8706f4547f"}
    assert "period" in json.loads((tmp_path / "S1/manifest.json").read_text()) and len(L.nbars_kvartalov(D4)) == L.NSEG == 16


def sreda2015(sid):
    daty = np.arange(np.datetime64("2015-12-28"), np.datetime64("2015-12-28") + 8)      # 2015Q4 + 2016Q1
    return ps.Sreda(Fake5(), "kod-fake", "vhody-fake", daty, [f"T{i:02d}" for i in range(17)])


@pytest.fixture
def kat(fake, monkeypatch, tmp_path):                                    # noqa: F811
    monkeypatch.setattr(ps, "fabrika", sreda2015)
    rz = rz5("v2")
    ps.proverit_ledger(rz, L.V21)                                        # драйвер v2 в 2.1 не отказывает
    ps.schet(["S1"], 0, {"S1": 40}, "sobol1_seed1", tmp_path / "v2_s1", 1, False, False, L.V21, rz)
    return tmp_path


def test_g4_svodka_i_svod_na_kataloge_v2_21(kat):
    root = kat / "v2_s1"
    ch = L.Chitatel(root, "S1")
    mf = json.loads((root / "S1/manifest.json").read_text())
    assert ch.v == L.V21 and mf["period"] == "v2" and len(mf["nbars"]) == 20
    assert mf["nbars"][3] == 4 and mf["nbars"][4] == 4 and sum(mf["nbars"]) == 8     # 2015Q4 и 2016Q1
    rows = SV.svodka_vse([root], 3)
    assert rows[0]["sverka"] == "ok" and rows[-1]["sverka"] == "ok" and rows[0]["ok"] == ch.n_unique() > 0, rows
    assert rows[0]["top_sharpe_okna"] != "—" and SV.main([str(root)]) == 0
    paths = SVOD.svod(root, kat / "out")                                 # без отказа читателя
    assert len(paths) == 4 and all(p.exists() for p in paths)


def test_mutaciya_god_2015_otrezan_lovitsya(kat, monkeypatch, tmp_path):
    """Мутант: gody_perioda возвращает 2016–2019 для v2 — запись 2015 падает, тест G1/G3 это ловит."""
    monkeypatch.setattr(L, "gody_perioda", lambda period=None: L.GODY)
    with pytest.raises(ValueError, match="вне 2016"):
        L.Pisatel21(tmp_path / "m", "S1", 0, {**MF, "T": D2.size, "period": "v2"}, daty=D2)
    with pytest.raises(ValueError, match="вне 2016"):                    # читатель мутанта: окно до конца 2015 недоступно
        L.Chitatel(kat / "v2_s1", "S1").okno(2016)
