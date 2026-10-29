"""Л7: ledger v2.1 (суммы сегментов по кварталам), IG-6, куб v: 3. Синтетика с известным ответом.
Мутации (подкладываются в отдельной копии): № 6 — выход на конце окна не платится (F_X = 0); окно на сегмент дальше
(захватывает 1 января года t); ядро Шарпа ddof = 0; все — падают на тестах ниже (см. test_mut_*)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import fastbt.ledger_v2 as L
import fastbt.progon_sobol as ps
from fastbt import stats
from fastbt.metriki_e import sr_dnevnoy

DATY = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
DATY = DATY[np.is_busday(DATY)]
T = DATY.size
KOM = 0.002


def konec(t):
    """Индекс за последним днём года t−1 (срез x[:konec] — окно [2016-01-01; 31.12(t−1)])."""
    return int((DATY < np.datetime64(f"{t}-01-01")).sum())


def pozicii(seed=1):
    g = np.random.default_rng(seed)
    pos = g.choice([-1.0, 0.0, 1.0], size=T)
    for t in (2017, 2018, 2019):
        pos[konec(t) - 1] = 1.0          # позиция открыта на конце окна — выход должен быть оплачен
    lr = g.normal(0, 0.01, T)
    net = pos * lr - np.abs(np.diff(pos, prepend=0.0)) * KOM
    return pos, lr, net


def test_window_po_summam_ravno_pryamomu_ryadu():
    _, _, x = pozicii()
    nb = L.nbars_kvartalov(DATY)
    s = L.summy(DATY, x)
    assert s.shape == (16, stats.NF) and nb.sum() == T
    for t in (2017, 2018, 2019):
        sh = stats.window(s, nb, L.segment_do_goda(t), 0.0)[0]
        assert abs(sh - sr_dnevnoy(x[:konec(t)]) * np.sqrt(252)) <= 1e-12


def test_tik_17_ryadov_srazu():
    g = np.random.default_rng(3)
    x = g.normal(0, 1e-3, (17, T))
    nb, s = L.nbars_kvartalov(DATY), L.summy(DATY, x)
    assert s.shape == (17, 16, stats.NF)
    sh = stats.window(s, nb, L.segment_do_goda(2019), 0.0)[0]
    pr = [sr_dnevnoy(r[:konec(2019)]) * np.sqrt(252) for r in x]
    assert np.max(np.abs(sh - pr)) <= 1e-12


def test_vyhod_na_konce_okna_mutaciya_6():
    pos, lr, net = pozicii()
    nb = L.nbars_kvartalov(DATY)
    for t in (2017, 2018, 2019):
        pryamo = stats.sharpe_window(pos, lr, 0, konec(t), KOM)           # эталон: окно платит выход (Н5)
        s = L.summy(DATY, net, poz=pos)
        assert abs(stats.window(s, nb, L.segment_do_goda(t), KOM)[0] - pryamo) <= 1e-12
        mut = stats.window(L.summy(DATY, net), nb, L.segment_do_goda(t), KOM)[0]      # F_X = 0 — выход не платится
        assert abs(mut - pryamo) > 1e-6                                   # мутация ловится


def test_okno_ne_zahvatyvaet_1_yanvarya_goda_t():
    _, _, x = pozicii()
    nb, s = L.nbars_kvartalov(DATY), L.summy(DATY, x)
    assert [L.segment_do_goda(t) for t in (2017, 2018, 2019)] == [3, 7, 11]
    for t in (2017, 2018, 2019):
        j = L.segment_do_goda(t)
        pryamo = sr_dnevnoy(x[:konec(t)]) * np.sqrt(252)
        assert abs(stats.window(s, nb, j, 0.0)[0] - pryamo) <= 1e-12
        assert abs(stats.window(s, nb, j + 1, 0.0)[0] - pryamo) > 1e-6   # мутация: лишний сегмент (дни года t)
    with pytest.raises(ValueError):
        L.segment_do_goda(2016)
    assert L.segment_do_goda(2020) == 15                                    # окно ≤ 2019 целиком
    with pytest.raises(ValueError):
        L.segment_do_goda(2021)


def test_data_2020_otkaz_i_pustoj_kvartal():
    with pytest.raises(ValueError):
        L.summy(np.array(["2020-01-02"], "datetime64[D]"), np.zeros(1))
    d = DATY[DATY < np.datetime64("2016-04-01")]
    s = L.summy(d, np.ones(d.size))
    assert (s[1:] == 0).all() and s[0, stats.F_S] == d.size


def man(v=None):
    return {"T": T, "kod": "K", "vhody": "V", **({"v": v} if v else {})}


def row(sha, v):
    return {"v": v, "klyuch_sha": sha, "status": "ok", "kod": "K", "vhody": "V",
            "by_year": {"2016": 1.0}, "by_quarter": {"2016Q1": 1.0}}


def ryady(seed):
    g = np.random.default_rng(seed)
    p = g.normal(0, 1e-3, T)
    return p, p[None, :] + g.normal(0, 1e-3, (17, T))


def test_ledger_v21_krugom_i_razmer(tmp_path):
    w = L.Pisatel21(tmp_path, "S1", 0, man(), shard=2, daty=DATY)
    dannye = [ryady(i) for i in range(3)]
    for i, (p, tk) in enumerate(dannye):
        w.add(row(f"s{i}", L.V21), p, tk)
    w.close()
    ch = L.Chitatel(tmp_path, "S1")
    assert ch.v == L.V21 and len(ch.rows()) == 3 and ch.manifest["poz_konec"] is False
    assert ch.summy().shape == (3, 16, stats.NF) and ch.summy(sub="tik").shape == (3, 17, 16, stats.NF)
    for i, (p, tk) in enumerate(dannye):
        r = [x for x in ch.rows() if x["klyuch_sha"] == f"s{i}"]
        sh = ch.okno(2019, r)[0][0]
        assert abs(sh - sr_dnevnoy(p[:konec(2019)]) * np.sqrt(252)) <= 1e-12
        shk = ch.okno(2018, r, "tik")[0][0]
        assert np.max(np.abs(shk - [sr_dnevnoy(q[:konec(2018)]) * np.sqrt(252) for q in tk])) <= 1e-12
    na_tochku = sum(f.stat().st_size for f in (tmp_path / "S1").rglob("*.npy")) / 3
    assert na_tochku < 21_000                                              # ≈ 20 КБ против ≈ 80 КБ рядами
    with pytest.raises(ValueError):                                         # comm > 0 без F_X — отказ, не «без выхода»
        ch.okno(2019, comm=KOM)


def test_ledger_v20_chitaetsya_kak_prezhde(tmp_path):
    w = L.Pisatel(tmp_path, "S1", 0, man(), shard=2)
    p, tk = ryady(5)
    w.add(row("a", L.V), p, tk)
    w.close()
    ch = L.Chitatel(tmp_path, "S1")
    assert ch.v == "2.0" and ch.matrica().shape == (T, 1) and ch.tik().shape == (1, 17, T)
    assert ch.matrica()[:, 0].tobytes() == p.tobytes()
    with pytest.raises(ValueError):
        ch.summy()


def test_versiya_neizvestna_i_smeshenie_otkaz(tmp_path):
    with pytest.raises(ValueError):
        L.Pisatel(tmp_path, "S1", 0, man("3.0"))
    L.Pisatel21(tmp_path, "S1", 0, man(), daty=DATY)
    with pytest.raises(ValueError):                                         # в каталоге 2.1 — писать 2.0 нельзя
        L.Pisatel(tmp_path, "S1", 1, man())
    w = L.Pisatel21(tmp_path / "x", "S1", 0, man())                         # без daty строку не записать
    with pytest.raises(ValueError):
        w.add(row("a", L.V21), *ryady(1))


def test_klyuch_vklyuchaet_komissiyu():
    pt = dict(k_s="net", tau=5, vhod="rynok")
    a = L.klyuch_sha(L.klyuch("S1", pt, "train4", 0.0004, None), "K", "V")
    assert a != L.klyuch_sha(L.klyuch("S1", pt, "train4", 0.0005, None), "K", "V")


def test_ig6_formula(monkeypatch, tmp_path):
    def svobodno(gb):
        monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**13, 0, int(gb * 1e9)))
    svobodno(30)                                                            # < 40 ГБ, но хватает: 1,5 × 196 608 × 20 КБ + 5 ГБ ≈ 10,9 ГБ
    ps.proverit_disk(196_608, tmp_path, ps.BAIT_V21)
    svobodno(10)
    with pytest.raises(SystemExit):
        ps.proverit_disk(196_608, tmp_path, ps.BAIT_V21)
    with pytest.raises(SystemExit):
        ps.proverit_disk(196_608, tmp_path)                                 # то же число точек рядами (80 КБ) — не хватает
    svobodno(4.9)
    with pytest.raises(SystemExit):                                         # 5 ГБ запаса — всегда
        ps.proverit_disk(1, tmp_path, ps.BAIT_V21)


def kub_v3():
    c = ps.load_kub("S1")
    f3 = {k: c["obshchie"].pop(k) for k in list(c["obshchie"])[:3]}
    c3 = {k: v for k, v in c.items() if k not in ("obshchie", "novye")}
    return {**c3, "v": 3, "filtr": c["obshchie"], "vyhod": {"stop_nach": {"setka": ["atr:2.0", "atr:3.0", "pct:1.5"]}},
            "faza3": {"consec": [1, 2]}, "msp": {"L": [1, 2]}, "yakorya_progon": f3}


def test_kub_v3_chitaetsya_faza3_i_msp_propuskayutsya():
    c = kub_v3()
    pts = ps.tochki(c, 0, 64)
    assert len(pts) == 64 and {p["stop_nach"] for _, p in pts} <= {"atr:2.0", "atr:3.0", "pct:1.5"}
    assert all("faza3" not in p and "msp" not in p and "consec" not in p and "L" not in p for _, p in pts)
    assert len({p["stop_nach"] for _, p in pts}) == 3
    with pytest.raises(ValueError):
        ps.tochki({**c, "v": 4}, 0, 2)
    with pytest.raises(ValueError):                                         # чужая ось signal по-прежнему отказ
        ps.tochki({**c, "signal": {**c["signal"], "lishnyaya": {"setka": [1, 2]}}}, 0, 2)


def test_drajver_pishet_v21_i_sverka_zelenaya(monkeypatch, tmp_path):
    class Fake:
        def evaluate_v2(self, pt, period):
            g = np.random.default_rng(int(L.klyuch_sha(L.klyuch("S1", pt, period, 0.0004, None), "a", "b")[:8], 16))
            tk = g.normal(0, 0.01, (17, T))
            return {"by_year": {str(y): 0.5 for y in range(2016, 2020)}, "T": T, "mediana_kv": 0.1}, tk.mean(0), tk
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Fake(), "kod", "vh", DATY, [f"T{i:02d}" for i in range(17)]))
    monkeypatch.setattr(ps, "_W", {})
    a = ps.schet(["S1"], 0, 20, "sobol1", tmp_path, 1, False, False, L.V21)
    assert a["S1"]["n"] == 20 and a["S1"]["oshibok"] == 0
    ch = L.Chitatel(tmp_path, "S1")
    assert ch.v == L.V21 and all(r["v"] == L.V21 for r in ch.rows())
    assert json.loads((Path(tmp_path) / "S1/manifest.json").read_text())["v"] == L.V21
    b = ps.schet(["S1"], 0, 30, "sobol1", tmp_path, 1, False, False, L.V21)
    assert b["S1"]["n"] == 10                                               # резюм: 20 закрыты
    monkeypatch.setattr(ps, "_W", {})
    assert ps.sverka(10, tmp_path, False) == 0
