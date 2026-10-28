"""Итерация 10 Ж-051 на подменённом оценщике: объединение ledger 2+3+4, сочетания «вкл/выкл», соседи, N и PBO по общей матрице."""
import pickle
import types

import pandas as pd

from fastbt import zh051_iter as I, zh051_iter10 as I10, zh051_kriteriy as K
from test_zh051_iter_drv import ALL, FakeEvr, drv, fake_ev, led_dir  # noqa: F401


class FakeEvr10(FakeEvr):
    sk = types.SimpleNamespace(sig_names=())

    def evaluate_ens(self, center, pts, mode, period="train"):
        return fake_ev({**center, "ans": mode}, period, self.n)


def _csv(real):
    def rc(path, *a, **k):
        if str(path).endswith("_ansambl.csv"):
            return pd.DataFrame(dict(nabor=["top3"], rezhim_ans=["srednee"], osi=["k_s tau"]), dtype=str)
        return real(path, *a, **k)
    return rc


def _add_ansambl(drv):
    """Колонка ансамбля итерации 2 (в настоящем ledger 2 она есть) — кладём в ledger 3 и пересчитываем строку sha256."""
    for sid in ALL:
        p, key = drv / sid / "ledger_it3.pkl", f"{sid}/ledger_it3.pkl"
        c = pickle.loads(p.read_bytes())
        c[(("ansambl", "top3", "srednee"), "train")] = ({"ansambl": "top3"}, fake_ev({"ansambl": 1}, "train"))
        p.write_bytes(pickle.dumps(c, protocol=4))
        f = drv / "SHA256SUMS.txt"
        f.write_text("".join(f"{K.sha(p)}  {key}  N=0\n" if key in ln else ln + "\n" for ln in f.read_text("utf-8").splitlines()),
                     encoding="utf-8")


def _prep(drv, monkeypatch, E):
    for it in (3, 4):
        I.main(["--it", str(it), "--strategii", *ALL])
    _add_ansambl(drv)
    monkeypatch.setattr(I10, "OUT", I.OUT)
    monkeypatch.setattr(I10, "PlusEvaluator", FakeEvr10)
    monkeypatch.setattr(I10, "elementy", lambda sid: E)
    monkeypatch.setattr(I10.A, "signal_points", lambda pts, names: pts)
    monkeypatch.setattr(I10.A, "tochki", lambda c, osi, axes: [c])
    monkeypatch.setattr(I10.pd, "read_csv", _csv(I10.pd.read_csv))


E3 = {"ans": {"nabor": "top3", "rezhim_ans": "srednee"}, "top_k": {"top_k": 5},
      "sektor": {"sektor_trend": "po_trendu", "shirina": "net", "ma_n": 100, "porog_shiriny": 0.5}}


def test_it10_tri_elementa_obshchiy_ledger_N_i_pbo(drv, monkeypatch):
    _prep(drv, monkeypatch, E3)
    r = I10.run_10("S1")
    f = I.OUT / "it10" / "S1_itog.csv"
    assert r["N_obshchee"] == r["N_do"] + r["N_novyh"] == r["N_pbo"] and r["N_novyh"] >= 1
    assert "находка #362" in f.read_text("utf-8").splitlines()[1] and r["elementy"] == "ans top_k sektor"
    assert K.sums_read()["S1/ledger_it10.pkl"][1] == r["N_obshchee"]
    assert r["kombo"]                                             # выбрано сочетание из ≥ 2 элементов


def test_it10_dva_elementa_net_sosedey_plato_ne_projdeno(drv, monkeypatch):
    _prep(drv, monkeypatch, {k: E3[k] for k in ("top_k", "sektor")})
    r = I10.run_10("S3")
    assert r["net_sosedey"] is True and r["proshla"] is False     # у пары два соседа (два одиночных элемента) < 3


def test_it10_odin_element_nechego_sochetat(drv, monkeypatch):
    _prep(drv, monkeypatch, {"top_k": {"top_k": 5}})
    r = I10.run_10("S2")
    assert r["verdikt"] == "нечего сочетать" and r["proshla"] is False
