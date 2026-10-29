"""V6-15: ledger v2.2 (setka, rol, klyuch_n) и читатель объединения v5 ∪ v6 (Ж-132 п. 4). Синтетика с известным ответом.
Мутации (monkeypatch в тесте): klyuch_n не растёт при объединении (константа) — N_k = 1 вместо 3; строка без хэша кода принята —
проверка строки отключена. Ключ строки (klyuch_sha) с хэшем кода: смена kod меняет klyuch_sha, но не klyuch_n."""
import numpy as np
import pytest

import fastbt.ledger_v2 as L
from fastbt import stats

DATY = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
DATY = DATY[np.is_busday(DATY)]
T = DATY.size
KL = {k: L.klyuch("S1", {"a": float(k)}, "train4", 0.0004, None) for k in range(4)}


def strok(k, v, kod, rol=None, setka=None):
    r = {"v": v, "strategiya": "S1", "status": "ok", "klyuch": KL[k], "kod": kod, "vhody": "V",
         "klyuch_sha": L.klyuch_sha(KL[k], kod, "V"), "by_year": {"2016": 1.0}, "by_quarter": {"2016Q1": 1.0}}
    if v == L.V22:
        r.update(setka=setka, rol=rol, klyuch_n=L.klyuch_n("S1", KL[k]))
    return r


def ryad(k):
    p = np.random.default_rng(k).normal(0, 1e-3, T)
    return p, np.tile(p, (17, 1))


def zapis(root, v, kod, tochki, setka=None):
    man = {"T": T, "kod": kod, "vhody": "V", "v": v, **({"setka": setka} if setka else {})}
    w = (L.Pisatel22 if v == L.V22 else L.Pisatel21)(root, "S1", 0, man, shard=2, daty=DATY)
    for k, rol in tochki:
        w.add(strok(k, v, kod, rol, setka), *ryad(k))
    w.close()
    return L.Chitatel(root, "S1")


def para(tmp_path):
    v5 = zapis(tmp_path / "v5", L.V21, "K5", [(0, None), (1, None)])
    v6 = zapis(tmp_path / "v6", L.V22, "K6", [(1, "sobol"), (2, "morris"), (3, "kontrol")], "v6")
    return v5, v6


def test_n_k_obedinenie_bez_kontrolya_i_dublej(tmp_path):
    v5, v6 = para(tmp_path)
    assert v6.v == L.V22 and v6.manifest["setka"] == "v6" and v6.n_unique() == 3
    ob = L.Obedinenie(v5, v6)
    assert ob.n_k() == 3                                       # {0, 1, 2}: точка 1 в обоих — один раз, kontrol (3) не в счёте
    assert ob.klyuchi() == [L.klyuch_n("S1", KL[k]) for k in (0, 1, 2)]
    assert ob.summy().shape == (3, 16, stats.NF) and ob.summy("tik").shape == (3, 17, 16, stats.NF)
    assert L.Obedinenie(v6).n_k() == 2 and L.Obedinenie(v5).n_k() == 2     # одиночные читатели


def test_klyuch_sha_s_hashem_koda_klyuch_n_bez(tmp_path):
    a, b = strok(1, L.V22, "K5", "sobol", "v6"), strok(1, L.V22, "K6", "sobol", "v6")
    assert a["klyuch_sha"] != b["klyuch_sha"] and a["klyuch_n"] == b["klyuch_n"]
    assert a["klyuch_sha"] == strok(1, L.V21, "K5")["klyuch_sha"]               # klyuch_sha не меняется (кэш Д5)
    assert L.klyuch_n("S1", KL[1]) != L.klyuch_n("S2", KL[1]) and L.klyuch_n("S1", KL[1]) != L.klyuch_n("S1", KL[2])


def test_stroka_bez_hasha_koda_otkaz(tmp_path):
    man = {"T": T, "kod": "K6", "vhody": "V", "v": L.V22, "setka": "v6"}
    for pole in ("kod", "vhody"):
        w = L.Pisatel22(tmp_path / pole, "S1", 0, man, daty=DATY)
        r = strok(1, L.V22, "K6", "sobol", "v6")
        del r[pole]
        with pytest.raises(ValueError, match="нет " + pole):
            w.add(r, *ryad(1))
    r = strok(1, L.V22, "K6", "sobol", "v6")
    r["kod"] = ""
    with pytest.raises(ValueError, match="нет kod"):
        L.proverit_stroku_22(r)


def test_otkazy_rol_setka_klyuch_n_manifest(tmp_path):
    r = strok(1, L.V22, "K6", "sobol", "v6")
    for izm, msg in (({"rol": "x"}, "rol"), ({"klyuch_n": "0" * 64}, "klyuch_n"), ({"setka": "v5"}, "setka")):
        with pytest.raises(ValueError, match=msg):
            L.proverit_stroku_22({**r, **izm}, "v6")
    with pytest.raises(ValueError, match="setka"):
        L.Pisatel22(tmp_path, "S1", 0, {"T": T, "kod": "K", "vhody": "V"}, daty=DATY)


def test_chitatel_otkazyvaet_ispravlennuyu_stroku(tmp_path):
    zapis(tmp_path / "v6", L.V22, "K6", [(1, "sobol")], "v6")
    rp = next((tmp_path / "v6" / "S1").glob("p*/rows.jsonl"))
    rp.write_text(rp.read_text(encoding="utf-8").replace('"kod": "K6"', '"kod": ""'), encoding="utf-8")
    with pytest.raises(ValueError, match="нет kod"):
        L.Chitatel(tmp_path / "v6", "S1")


def test_obedinenie_otkazy(tmp_path):
    v5, v6 = para(tmp_path)
    with pytest.raises(ValueError):
        L.Obedinenie()
    s2 = L.Chitatel(tmp_path / "v6", "S1")
    s2.manifest = {**s2.manifest, "strategiya": "S2"}
    with pytest.raises(ValueError, match="strategiya"):
        L.Obedinenie(v5, s2)


def test_dubl_sovpal_odin_raz_ne_sovpal_otkaz(tmp_path, monkeypatch):
    v5, v6 = para(tmp_path)                                    # точка 1 в обоих с разным kod, ряды одинаковы
    assert L.Obedinenie(v5, v6).n_k() == 3
    orig = ryad
    monkeypatch.setitem(globals(), "ryad", lambda k: (orig(k)[0] + (1e-6 if k == 1 else 0.0), orig(k)[1]))
    v6b = zapis(tmp_path / "v6b", L.V22, "K6", [(1, "sobol")], "v6")      # код изменил точку 1
    with pytest.raises(ValueError, match=r"S1.*K5 ≠ K6.*max\|Δ\| = \d") as e:
        L.Obedinenie(v5, v6b)
    assert L.klyuch_n("S1", KL[1]) in str(e.value)


def test_mutaciya_dubl_svoditsya_molcha(tmp_path, monkeypatch):
    v5, _ = para(tmp_path)
    orig = ryad
    monkeypatch.setitem(globals(), "ryad", lambda k: (orig(k)[0] + (1e-6 if k == 1 else 0.0), orig(k)[1]))
    v6b = zapis(tmp_path / "v6b", L.V22, "K6", [(1, "sobol")], "v6")
    monkeypatch.setattr(L.Obedinenie, "_sverit_dubl", staticmethod(lambda *a, **k: None))
    assert L.Obedinenie(v5, v6b).n_k() == 2                    # мутация молча свела расхождение — тест выше упал бы


def test_mutaciya_klyuch_n_ne_rastyot_pri_obedinenii(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "klyuch_n", lambda sid, kl: "const")   # все точки «одна» — N не растёт при объединении
    v5, v6 = para(tmp_path)
    with pytest.raises(ValueError, match="разошлись"):          # разные точки под одним ключом ловит сверка дублей
        L.Obedinenie(v5, v6)
    monkeypatch.setattr(L.Obedinenie, "_sverit_dubl", staticmethod(lambda *a, **k: None))
    assert L.Obedinenie(v5, v6).n_k() == 1 != 3               # тест n_k == 3 выше упал бы


def test_mutaciya_stroka_bez_hasha_koda_prinyata(tmp_path, monkeypatch):
    man = {"T": T, "kod": "K6", "vhody": "V", "v": L.V22, "setka": "v6"}
    r = strok(1, L.V22, "K6", "sobol", "v6")
    del r["kod"]
    monkeypatch.setattr(L, "proverit_stroku_22", lambda *a, **k: None)
    w = L.Pisatel22(tmp_path, "S1", 0, man, daty=DATY)
    w.add(r, *ryad(1))                                         # мутация принимает строку — тест отказа выше упал бы
    w.close()
    monkeypatch.undo()
    with pytest.raises(ValueError, match="нет kod"):
        L.Chitatel(tmp_path, "S1")
