"""П-1, лист Л-1: контракт ledger v2.0 (etap1 §1.1-1.7). Тесты написаны до кода; красные, пока fastbt/ledger_v2.py нет."""
import inspect
import json
import math
import os

import numpy as np
import pytest

from . import BAZA, KOD, VHODY, TIKERY, T, pisat, row, ryady, stroki_jsonl


def _lm():
    import fastbt.ledger_v2 as m
    return m


# ---------- §1.5 API ----------
def test_api_imena_i_signatury():
    m = _lm()
    assert (m.FORMAT, m.V) == ("etap1-ledger", "2.0")
    pars = lambda f: list(inspect.signature(f).parameters)
    assert pars(m.tok) == ["v"]
    assert pars(m.canon_v2) == ["sid", "pt"]
    assert pars(m.klyuch) == ["sid", "pt", "period", "komissiya", "fdr"]
    assert pars(m.klyuch_sha) == ["kl", "kod", "vhody"]
    assert pars(m.kvartaly) == ["daty", "x"]
    assert pars(m.tselevaya) == ["by_quarter"]
    assert pars(m.Pisatel.__init__) == ["self", "root", "sid", "potok", "manifest", "shard"]
    assert inspect.signature(m.Pisatel.__init__).parameters["shard"].default == 256
    assert pars(m.Pisatel.add) == ["self", "row", "ex_port", "ex_tik"]
    for nm in ("done", "close"):
        assert hasattr(m.Pisatel, nm)
    for nm in ("rows", "matrica", "tik", "n_unique"):
        assert hasattr(m.Chitatel, nm)
    assert "sverka_kod" in inspect.signature(m.Chitatel.__init__).parameters


def test_tok_chisla_i_stroki():
    m = _lm()
    assert m.tok("net") == "net" and m.tok("rv22") == "rv22"
    assert m.tok(1) == "1.0" and m.tok(1.67) == "1.67" and m.tok(5.0) == "5.0"
    assert m.tok(0.0004) == "0.0004"


# ---------- §1.1 ключ ----------
def test_klyuch_kanonicheskaya_zapis_i_sha():
    m = _lm()
    kl = m.klyuch("S1", m.canon_v2("S1", BAZA), "train4", 0.0004, "null")
    assert kl["tau"] == "5.0" and kl["komissiya"] == "0.0004" and kl["fdr"] == "null"
    assert kl["period"] == "train4" and "sobytiya" not in kl
    assert kl["t_vyhod"] == "close"
    assert all(isinstance(v, str) for v in kl.values())
    sha = m.klyuch_sha(kl, "k", "v")
    import hashlib
    kanon = json.dumps(kl, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert sha == hashlib.sha256((kanon + "|k|v|2.0").encode()).hexdigest()


ZNACHENIYA = {"ed_lim": "atr14", "zhizn": "36", "lestnica": "0.5", "gap": "propusk", "t_vyhod": "1500",
              "k_x": "net", "storona": "long", "filtr": "cbr", "z_entry": 2.33, "sigma": "rv66",
              "q_sily": 0.8, "razmer": 20, "tau": 10, "rezhim": "imoex_ma200", "proverka": "bars5m"}


@pytest.mark.parametrize("os_", sorted(ZNACHENIYA))
def test_kazhdaya_os_menyaet_klyuch_sha(os_):
    m = _lm()
    a = m.klyuch_sha(m.klyuch("S1", m.canon_v2("S1", BAZA), "train4", 0.0004, "null"), KOD, VHODY)
    pt = dict(BAZA, **{os_: ZNACHENIYA[os_]})
    b = m.klyuch_sha(m.klyuch("S1", m.canon_v2("S1", pt), "train4", 0.0004, "null"), KOD, VHODY)
    assert a != b, f"ось {os_} не различима в ключе (М-L1)"


def test_k_x_razlichim_pri_k_s_net():
    m = _lm()
    f = lambda kx: m.klyuch_sha(m.klyuch("S1", m.canon_v2("S1", dict(BAZA, k_s="net", R="net", k_x=kx)),
                                         "train4", 0.0004, "null"), KOD, VHODY)
    assert f("0.5") != f("1.0") != f("net")


def test_komissiya_fdr_kod_vhody_v_klyuche():
    m = _lm()
    c = m.canon_v2("S1", BAZA)
    sha = lambda k=0.0004, f="null", kod=KOD, vh=VHODY: m.klyuch_sha(m.klyuch("S1", c, "train4", k, f), kod, vh)
    assert sha() != sha(k=0.0005), "komissiya не в ключе (М-L2)"
    assert sha() != sha(f="t0") and sha() != sha(kod="другой") and sha() != sha(vh="другие")
    assert sha() == sha()


# ---------- §1.3 canon_v2 ----------
def test_canon_v2_pravila():
    m = _lm()
    r = m.canon_v2("S1", dict(BAZA, k_s="net", R=2.0, k_tr=2.0, bezubytok="plus1sigma"))
    assert (r["R"], r["k_tr"], r["bezubytok"]) == ("net",) * 3
    r = m.canon_v2("S1", dict(BAZA, k_s=2.0, k_x="1.0"))
    assert r["k_x"] == "net"
    r = m.canon_v2("S1", dict(BAZA, vhod="rynok", ed_lim="atr14", zhizn="12", lestnica="0.5", gap="propusk"))
    assert (r["ed_lim"], r["zhizn"], r["lestnica"], r["gap"]) == ("sigma", "den", "net", "vhod")
    assert m.canon_v2("S5", dict(BAZA, storona="short_shag1"))["storona"] == "obe"
    assert m.canon_v2("S1", dict(BAZA, storona="short_shag1"))["storona"] == "short_shag1"
    # sigma при vhod=rynok сохраняется
    assert m.canon_v2("S1", dict(BAZA, vhod="rynok", sigma="rv66"))["sigma"] == "rv66"


def test_canon_v2_ne_menyaet_vhod():
    m = _lm()
    pt = dict(BAZA, k_s="net", k_tr=2.0)
    m.canon_v2("S1", pt)
    assert pt["k_tr"] == 2.0


def test_n_unique_skhlopyvaet_k_tr_pri_k_s_net(tmp_path):
    """М-L9: canon_v2 не схлопывает k_tr при k_s=net."""
    m = _lm()
    pisat(m, tmp_path, 0)
    w2 = m.Pisatel(tmp_path, "S1", 1, _man(), shard=4)
    for i, ktr in enumerate([1.5, 2.0, 3.0, "net"]):
        pt = m.canon_v2("S1", dict(BAZA, k_s="net", R="net", k_tr=ktr))
        kl = m.klyuch("S1", pt, "train4", 0.0004, "null")
        r = dict(row(m, i), klyuch=kl, klyuch_sha=m.klyuch_sha(kl, KOD, VHODY))
        w2.add(r, *ryady(i))
    w2.close()
    assert m.Chitatel(tmp_path, "S1").n_unique() == 1


def _man():
    from . import manifest
    return manifest()


# ---------- kvartaly / tselevaya ----------
def _dni():
    d = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
    wd = (d.astype("datetime64[D]").astype(int) + 3) % 7  # 0 = пн
    return d[wd < 5]


def _ozhid(daty, x, lo, hi):
    v = x[(daty >= np.datetime64(lo)) & (daty < np.datetime64(hi))]
    return float(np.mean(v) / np.std(v, ddof=1) * math.sqrt(252))


def test_kvartaly_izvestnyj_otvet():
    m = _lm()
    daty = _dni()
    rng = np.random.default_rng(7)
    x = rng.normal(0.0005, 0.01, len(daty))
    x[daty == np.datetime64("2016-04-01")] = 0.2  # последний день Q1 / граница (М-L5)
    x[daty == np.datetime64("2016-07-01")] = -0.2
    r = m.kvartaly(daty, x)
    assert list(r) == [f"{y}Q{q}" for y in range(2016, 2020) for q in range(1, 5)]
    gr = {1: ("01-01", "04-01"), 2: ("04-01", "07-01"), 3: ("07-01", "10-01"), 4: ("10-01", "+1-01-01")}
    for y in range(2016, 2020):
        for q, (a, b) in gr.items():
            lo = f"{y}-{a}"
            hi = f"{y + 1}-01-01" if q == 4 else f"{y}-{b}"
            assert r[f"{y}Q{q}"] == pytest.approx(_ozhid(daty, x, lo, hi), rel=1e-12), f"{y}Q{q}"


def test_kvartaly_ddof1_ne_ddof0():
    """М-L7: на коротком ряду ddof=0 даёт другое число."""
    m = _lm()
    daty = _dni()
    x = np.zeros(len(daty))
    q1 = (daty >= np.datetime64("2016-01-01")) & (daty < np.datetime64("2016-04-01"))
    x[q1] = np.tile([0.01, -0.005, 0.02], q1.sum())[: q1.sum()]
    exp = _ozhid(daty, x, "2016-01-01", "2016-04-01")
    assert m.kvartaly(daty, x)["2016Q1"] == pytest.approx(exp, rel=1e-12)
    n = q1.sum()
    assert exp != pytest.approx(exp * math.sqrt((n - 1) / n), rel=1e-6)


def test_kvartal_nulevoj_sigma_daet_nol_i_pustoj_nan():
    m = _lm()
    daty = _dni()
    x = np.zeros(len(daty))
    assert m.kvartaly(daty, x)["2016Q3"] == 0.0
    sh = daty < np.datetime64("2019-01-01")  # нет дней 2019
    r = m.kvartaly(daty[sh], x[sh])
    assert math.isnan(r["2019Q2"])


def test_tselevaya_mediana_i_dolya():
    m = _lm()
    bq = {f"q{i}": float(i - 7) for i in range(16)}  # -7…8
    med, dol = m.tselevaya(bq)
    assert med == pytest.approx(0.5) and dol == pytest.approx(8 / 16)


def test_tselevaya_nan_ne_polozhitelen():
    """М-L6: один nan-квартал: целевая nan, доля считает nan не положительным."""
    m = _lm()
    bq = {f"q{i}": 1.0 for i in range(16)}
    bq["q3"] = float("nan")
    med, dol = m.tselevaya(bq)
    assert math.isnan(med) and dol == pytest.approx(15 / 16)


# ---------- §1.4 запись ----------
def test_zapis_chtenie_port_float64_pobajtno(tmp_path):
    """М-L3: port во float64 (значение, не представимое во float32, возвращается побайтно)."""
    m = _lm()
    pisat(m, tmp_path, 4)
    ch = m.Chitatel(tmp_path, "S1")
    mat = ch.matrica()
    assert mat.dtype == np.float64 and mat.shape == (T, 4)
    for j in range(4):
        assert mat[:, j].tobytes() == ryady(j)[0].tobytes()
    tk = ch.tik()
    assert tk.dtype == np.float32 and tk.shape == (4, 17, T)
    assert tk[2].tobytes() == ryady(2)[1].tobytes()


def test_tozhdestvo_nanmean_tikerov_port(tmp_path):
    """М-L4 на уровне читателя: nanmean по тикерам = port ≤ 1e-8 после float32."""
    m = _lm()
    pisat(m, tmp_path, 4)
    ch = m.Chitatel(tmp_path, "S1")
    d = np.nanmean(ch.tik().astype(np.float64), axis=1).T - ch.matrica()
    assert np.max(np.abs(d)) <= 1e-8


def test_poryadok_zapisi_shard_do_stroki(tmp_path, monkeypatch):
    """М-L8: если переименование шарда падает, строк шарда в rows.jsonl нет."""
    m = _lm()
    boom = lambda *a, **k: (_ for _ in ()).throw(OSError("обрыв"))
    monkeypatch.setattr(os, "replace", boom)
    monkeypatch.setattr(os, "rename", boom)
    with pytest.raises(OSError):
        pisat(m, tmp_path, 4, shard=4)
    assert stroki_jsonl(tmp_path) == []


def test_obryv_do_zakrytiya_shardа_chitatel_ne_vidit_stroki(tmp_path):
    m = _lm()
    pisat(m, tmp_path, 6, shard=4, zakryt=False)  # шард 0 закрыт (4), в буфере 2 строки
    ch = m.Chitatel(tmp_path, "S1")
    assert len(ch.rows()) == 4


def test_done_vozvrashchaet_zakrytye(tmp_path):
    m = _lm()
    pisat(m, tmp_path, 6, shard=4, zakryt=False)
    w = m.Pisatel(tmp_path, "S1", 0, _man(), shard=4)
    assert w.done() == {row(m, i)["klyuch_sha"] for i in range(4)}


# ---------- §1.6 читатель ----------
def test_chitatel_otkaz_shard_bez_sha_i_nesovpadenie(tmp_path):
    m = _lm()
    pisat(m, tmp_path, 4)
    sums = tmp_path / "S1" / "SHA256SUMS.txt"
    sums.write_text("")
    with pytest.raises(ValueError):
        m.Chitatel(tmp_path, "S1").rows()
    sums.write_text("0" * 64 + "  p00/port/00000.npy\n" + "0" * 64 + "  p00/tik/00000.npy\n")
    with pytest.raises(ValueError):
        m.Chitatel(tmp_path, "S1").rows()


def test_chitatel_otkaz_versiya(tmp_path):
    m = _lm()
    pisat(m, tmp_path, 4)
    p = tmp_path / "S1" / "manifest.json"
    mf = json.loads(p.read_text())
    mf["v"] = "1.0"
    p.write_text(json.dumps(mf))
    with pytest.raises(ValueError):
        m.Chitatel(tmp_path, "S1").rows()


def test_chitatel_otkaz_dva_kod_i_sverka_kod(tmp_path):
    m = _lm()
    pisat(m, tmp_path, 4, potok=0)
    pisat(m, tmp_path, 4, potok=1, kod="kod-B")
    with pytest.raises(ValueError):
        m.Chitatel(tmp_path, "S1").rows()
    assert len(m.Chitatel(tmp_path, "S1", sverka_kod=True).rows()) == 8


def test_chitatel_otkaz_god_vne_2016_2019(tmp_path):
    m = _lm()
    w = pisat(m, tmp_path, 0)
    w2 = m.Pisatel(tmp_path, "S1", 3, _man(), shard=4)
    for i in range(4):
        r = row(m, i)
        r["by_year"]["2020"] = 0.1
        w2.add(r, *ryady(i))
    w2.close()
    with pytest.raises(ValueError):
        m.Chitatel(tmp_path, "S1").rows()


def test_dubli_klyuch_sha_pervaya_stroka_n_unique(tmp_path):
    m = _lm()
    pisat(m, tmp_path, 0)
    w = m.Pisatel(tmp_path, "S1", 0, _man(), shard=4)
    for n, i in enumerate((0, 1, 0, 1)):
        r = row(m, i)
        r["sharpe"] = float(n)  # дубль отличается значением; берётся первая строка
        w.add(r, *ryady(i))
    w.close()
    ch = m.Chitatel(tmp_path, "S1")
    assert [x["sharpe"] for x in ch.rows()] == [0.0, 1.0]
    assert ch.n_unique() == 2 and len(ch.rows()) == 2
    assert ch.matrica().shape == (T, 2)
