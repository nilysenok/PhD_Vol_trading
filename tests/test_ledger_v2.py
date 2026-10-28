"""Л-1: ledger v2.0 (контракт docs/otchety/2026-10-28_progon_etap1.md §1). Синтетика с известным ответом;
мутации М-L1…М-L9 подкладываются в отдельной копии (тело PR), каждая роняет хотя бы один тест ниже."""
import json

import numpy as np
import pytest

from fastbt import ledger_v2 as L

T = 6
PT = dict(k_s=1.5, R=2.0, k_tr="net", bezubytok="net", tau=5, vhod=0.5, t_vyhod="close", ed_lim="sigma",
          zhizn="den", lestnica="net", gap="vhod", k_x="net", storona="obe", filtr="net")


def man(kod="K1", vhody="V1"):
    return {"T": T, "kod": kod, "vhody": vhody}


def row(sha, status="ok", kod="K1", vhody="V1", **kw):
    return {"v": "2.0", "klyuch_sha": sha, "status": status, "kod": kod, "vhody": vhody,
            "by_year": {"2016": 1.0}, "by_quarter": {"2016Q1": 1.0}, **kw}


def ex(seed):
    g = np.random.default_rng(seed)
    port = g.normal(0, 1e-3, T)
    dev = g.normal(0, 1e-3, (17, T))
    return port, port[None, :] + dev - dev.mean(axis=0)[None, :]


def fill(root, shas, shard=2, potok=0, **kw):
    w = L.Pisatel(root, "S1", potok, man(), shard)
    for i, s in enumerate(shas):
        w.add(row(s, **kw), *ex(i))
    return w


def test_tok():
    assert (L.tok(1), L.tok(1.67), L.tok(5), L.tok("net"), L.tok(np.float64(0.5))) == ("1.0", "1.67", "5.0", "net", "0.5")


def test_canon_v2_pravila():
    a = L.canon_v2("S1", {**PT, "k_s": "net", "R": 3.0, "k_tr": 1.0, "bezubytok": 0.5})
    assert (a["R"], a["k_tr"], a["bezubytok"], a["k_x"]) == ("net",) * 3 + ("net",)
    b = L.canon_v2("S1", {**PT, "k_x": 1.0})
    assert b["k_x"] == "net" and b["R"] == 2.0 and b["k_tr"] == "net"
    c = L.canon_v2("S1", {**PT, "vhod": "rynok", "ed_lim": "atr14", "zhizn": 12, "lestnica": 0.5, "gap": "propusk"})
    assert (c["ed_lim"], c["zhizn"], c["lestnica"], c["gap"]) == ("sigma", "den", "net", "vhod")
    assert L.canon_v2("S5", {**PT, "storona": "short_shag1"})["storona"] == "obe"
    assert L.canon_v2("S1", {**PT, "storona": "short_shag1"})["storona"] == "short_shag1"


def sha(pt, kom=0.0004, fdr=None, kod="K", vh="V", sid="S1"):
    return L.klyuch_sha(L.klyuch(sid, pt, "train4", kom, fdr), kod, vh)


def test_klyuch_razlichimost_i_skhlopyvanie():
    base = sha(PT)
    assert sha({**PT, "t_vyhod": "1500"}) != base                      # М-L1
    assert sha(PT, kom=0.0005) != base                                 # М-L2
    assert sha(PT, fdr="t0") != base and sha(PT, kod="K2") != base and sha(PT, vh="V2") != base
    assert sha({**PT, "tau": 5.0, "k_s": 1.5}) == base                 # 5 и 5.0 — один токен
    n = {**PT, "k_s": "net"}
    assert sha({**n, "k_tr": 2.0}) == sha({**n, "k_tr": "net"})        # М-L9
    assert sha({**PT, "k_tr": 2.0}) != sha({**PT, "k_tr": "net"})
    assert sha({**PT, "storona": "short_shag1"}, sid="S5") == sha({**PT, "storona": "obe"}, sid="S5")
    kl = L.klyuch("S1", PT, "train4", 0.0004, None)
    assert kl["komissiya"] == "0.0004" and kl["fdr"] == "null" and kl["period"] == "train4"


def dates(*s):
    return np.array(s, "datetime64[D]")


def test_kvartaly_granitsa_i_ddof1():
    d = dates("2016-03-30", "2016-03-31", "2016-04-01", "2016-04-02", "2016-12-31", "2017-01-01", "2017-01-02")
    x = [1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 4.0]
    kv = L.kvartaly(d, np.array(x))
    assert len(kv) == 16 and list(kv)[:2] == ["2016Q1", "2016Q2"] and list(kv)[-1] == "2019Q4"
    k = np.sqrt(252)
    assert kv["2016Q1"] == pytest.approx(2 / np.sqrt(2) * k)           # ddof = 1: sd([1,3]) = √2
    assert kv["2016Q2"] == pytest.approx(3 / np.sqrt(2) * k)           # 01.04 — уже Q2
    assert kv["2017Q1"] == pytest.approx(3 / np.sqrt(2) * k) and kv["2016Q4"] == 0.0  # один день — σ нет, SR = 0
    assert np.isnan(kv["2018Q3"]) and sum(np.isnan(v) for v in kv.values()) == 12


def test_kvartaly_otkaz():
    with pytest.raises(ValueError):
        L.kvartaly(dates("2019-12-31", "2020-01-02"), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        L.kvartaly(dates("2016-01-04"), np.array([1.0, 2.0]))


def test_tselevaya():
    q = {f"{g}Q{i}": float(k) for k, (g, i) in enumerate([(g, i) for g in range(2016, 2020) for i in range(1, 5)], 1)}
    assert L.tselevaya(q) == (8.5, 1.0)
    q["2018Q1"] = -1.0
    assert L.tselevaya(q) == (7.5, 15 / 16)                    # 16 значений: -1, 1…8, 10…16
    q["2018Q2"] = float("nan")                                         # М-L6: nan не положительный
    med, dol = L.tselevaya(q)
    assert np.isnan(med) and dol == 14 / 16
    z = dict.fromkeys(q, 0.0)
    assert L.tselevaya(z) == (0.0, 0.0)


def test_zapis_chtenie_pobajtno(tmp_path):
    fill(tmp_path, ["a", "b", "c", "d", "e"]).close()
    c = L.Chitatel(tmp_path, "S1")
    assert [r["klyuch_sha"] for r in c.rows()] == list("abcde")
    m, t = c.matrica(), c.tik()
    assert m.dtype == np.float64 and m.shape == (T, 5) and t.dtype == np.float32 and t.shape == (5, 17, T)
    for i in range(5):
        p, tk = ex(i)
        assert m[:, i].tobytes() == p.tobytes()                        # М-L3: port float64 побайтно
        assert t[i].tobytes() == tk.astype(np.float32).tobytes()
        assert np.abs(np.nanmean(t[i].astype(float), axis=0) - m[:, i]).max() <= 1e-8   # тождество §1.4
    assert c.rows()[3]["excess"] == {"potok": "p00", "shard": 1, "stroka": 1}
    assert c.matrica(c.rows()[::-1])[:, 0].tobytes() == ex(4)[0].tobytes()


def test_pisatel_otkaz_float32_i_formy(tmp_path):
    w = L.Pisatel(tmp_path, "S1", 0, man())
    p, tk = ex(0)
    with pytest.raises(TypeError):
        w.add(row("a"), p.astype(np.float32), tk)
    with pytest.raises(TypeError):
        w.add(row("a"), p, tk[:16])


def test_shard_256_zakryvaetsya_sam(tmp_path):
    w = fill(tmp_path, [f"s{i}" for i in range(256)], shard=256)
    assert (tmp_path / "S1/p00/port/00000.npy").exists() and len(w.done()) == 256
    assert L.Chitatel(tmp_path, "S1").n_unique() == 256


def test_obryv_posle_add_bez_close(tmp_path):                         # М-L8
    w = fill(tmp_path, ["a", "b", "c"], shard=4)
    rp = tmp_path / "S1/p00/rows.jsonl"
    assert not rp.exists() or rp.read_text() == ""
    assert L.Chitatel(tmp_path, "S1").rows() == []
    assert L.Pisatel(tmp_path, "S1", 0, man()).done() == set() and w.done() == set()


def test_obryv_mezhdu_rows_i_sha_vosstanovlenie(tmp_path):
    fill(tmp_path, ["a", "b", "c", "d"]).close()                       # шарды 0 и 1
    sums = tmp_path / "S1/SHA256SUMS.txt"
    sums.write_text("".join(sums.read_text().splitlines(True)[:2]))    # шард 1 без SHA
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path, "S1")
    w = L.Pisatel(tmp_path, "S1", 0, man(), 2)
    assert w.done() == {"a", "b"}
    for i, s in enumerate(["c", "d"], 2):
        w.add(row(s), *ex(i))
    w.close()
    c = L.Chitatel(tmp_path, "S1")
    assert [r["klyuch_sha"] for r in c.rows()] == list("abcd") and len((tmp_path / "S1/p00/rows.jsonl").read_text().splitlines()) == 4


def test_dubli_n_unique(tmp_path):
    fill(tmp_path, ["a", "b", "a", "c"]).close()
    c = L.Chitatel(tmp_path, "S1")
    assert c.n_unique() == 3 and [r["klyuch_sha"] for r in c.rows()] == list("abc")
    assert c.matrica()[:, 0].tobytes() == ex(0)[0].tobytes()           # берётся первая строка
    w = fill(tmp_path / "x", ["a", "a"])
    w.close()
    assert L.Chitatel(tmp_path / "x", "S1").n_unique() == 1


def test_ne_schitana_i_nan_tikera(tmp_path):
    w = L.Pisatel(tmp_path, "S1", 0, man(), 2)
    w.add(row("n", status="ne_schitana"), None, None)
    p, tk = ex(1)
    tk[3, 2] = np.nan
    w.add(row("o"), p, tk)
    c = L.Chitatel(tmp_path, "S1")
    assert w.done() == {"n", "o"} and c.n_unique() == 1 and len(c.rows(None)) == 2
    assert c.rows(None)[0]["excess"] is None and np.isnan(c.tik()[0, 3, 2]) and np.isfinite(c.tik()[0, 3, 1])


def test_chitatel_otkazy(tmp_path):
    fill(tmp_path, ["a", "b"]).close()
    sums = tmp_path / "S1/SHA256SUMS.txt"
    ok = sums.read_text()
    sums.write_text(ok.splitlines(True)[0])                            # tik-шард без строки SHA
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path, "S1")
    sums.write_text(ok.replace(ok[:3], "000", 1))                      # sha не совпал
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path, "S1")
    sums.write_text(ok)
    assert L.Chitatel(tmp_path, "S1").n_unique() == 2


def test_chitatel_versii_goda_kod(tmp_path):
    fill(tmp_path / "a", ["a", "b"], by_year={"2020": 1.0}).close()
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path / "a", "S1")
    fill(tmp_path / "b", ["a", "b"], by_quarter={"2021Q1": 1.0}).close()
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path / "b", "S1")
    fill(tmp_path / "c", ["a", "b"], v="1.0").close()
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path / "c", "S1")
    mp = tmp_path / "d/S1/manifest.json"
    fill(tmp_path / "d", ["a", "b"]).close()
    mp.write_text(json.dumps({**json.loads(mp.read_text()), "v": "1.0"}))
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path / "d", "S1")


def test_dva_kod_otkaz_i_sverka(tmp_path):
    fill(tmp_path, ["a", "b"]).close()
    fill(tmp_path, ["c", "d"], potok=1, kod="K2").close()
    with pytest.raises(ValueError):
        L.Chitatel(tmp_path, "S1")
    assert L.Chitatel(tmp_path, "S1", sverka_kod=True).n_unique() == 4
    with pytest.raises(ValueError):
        L.Pisatel(tmp_path, "S1", 2, man(kod="K9"))
    with pytest.raises(ValueError):
        L.Pisatel(tmp_path, "S1", 2, man(vhody="V9"))


def test_dva_potoka_i_dubl_mezhdu_nimi(tmp_path):
    fill(tmp_path, ["a", "b"]).close()
    fill(tmp_path, ["b", "c"], potok=1).close()
    c = L.Chitatel(tmp_path, "S1")
    assert c.n_unique() == 3 and {r["excess"]["potok"] for r in c.rows()} == {"p00", "p01"}
