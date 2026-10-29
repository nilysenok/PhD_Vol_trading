# -*- coding: utf-8 -*-
"""К-В1b: итерации 1 и 2 на В1 (train4: 2016–2019). Синтетика, данных и счёта нет.
Мутации: запись В1 в _ledger; целевая по 2 годам из 4; ворота сделок не в каждом году. Умолчание — golden sha256 старого кода."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from fastbt import etap1, etap1_ansambl_run as AR, etap1_poisk as P, etap1_poisk_vorota as V
from test_etap1_poisk import SETKA, World

Y4 = (2016, 2017, 2018, 2019)
NOISE4 = {y: np.random.default_rng(y).normal(size=24) for y in Y4}


class World4(World):
    """Как World, но годы периода из PERIOD_YEARS; 2016 хуже на 2,0 mSR; в 2016 сделок sdelok16 (остальные 40)."""

    def __init__(self, sdelok16=40):
        super().__init__()
        self.sdelok16 = sdelok16

    def evaluate(self, p, period):
        ys, s = V.PERIOD_YEARS[period], self.score(p)
        return dict(msr_by_year={y: s - (2.0 if y == 2016 else 0.0) for y in ys}, excess_d={y: NOISE4[y] + 0.001 * s for y in ys},
                    sdelok_y={y: (self.sdelok16 if y == 2016 else 40) for y in ys}, vyrozhdena=False,
                    storony_y={y: 80 for y in ys})        # сторон хватает всегда: отсев только по воротам сделок


def run4(w=None, **kw):
    w = w or World4()
    return P.run_strategy("S9", w.evaluate, lambda p: True, run_id="t", setka=SETKA, period="train4", **kw)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- умолчание = старый путь побайтно (хэши получены старым кодом origin/kanon-v22-pereschet, run_id="t") ----------------
GOLDEN = dict(morris="1e11f6a054980fe5ce9f265076a3679223ba208110655ccec1e031bb2d4b015c",
              faktor="f26fc41005f8191fa35005627ce16aa244353fe8bf8df72f30172192e1b0ca03",
              itog="94d2c41c0469e94fd3fd98299bac42b9f1364126f54d54ebe206ae433ebbd48a")    # itog — до Ж-090, не проверяется (см. GOLDEN_BEZ_PBO)


# Ж-080 (PR #424): в итог добавлена справочная колонка pbo_argmax; G1 — от центра плато (pbo), не от argmax.
# Полный хэш итога после добавления колонки (записан рядом с проверкой прежних колонок, не вместо неё):
NOVYE_KOLONKI = ("pbo_argmax",)
# Ж-090: итог после смены правила PBO. Хэш старого итога (a9077807…, до Ж-090) снят старым кодом 25ffeb8b; тот же итог без
# колонок PBO_ZAVISIT у старого и нового кода совпал побайтно: c73b7404… (меняются только колонки, зависящие от PBO).
PBO_ZAVISIT = ("pbo_argmax", "pbo", "G1_pbo", "ok", "verdikt")
GOLDEN_BEZ_PBO = "c73b7404dac3ac30a3052e9e8d82af1bfef620d5ead4c8ac2932e1bc53e2cead"
ITOG_S_NOVYM = "503c4fd0b76be83083e0a0cd7bcc244335ca515225337734ee30d87937c7a98b"


def _bez_novykh_kolonok(p: Path, names=NOVYE_KOLONKI) -> bytes:
    """Итог без колонок names: поля вырезаются из шапки и каждой строки (csv: в поле params есть запятые в кавычках),
    остальное — детерминированная запись; строки «#» — как есть."""
    import csv
    import io
    out, idx = [], None
    for ln in p.read_text("utf-8").split("\n"):
        if ln.startswith("#") or not ln:
            out.append(ln)
            continue
        f = next(csv.reader([ln]))
        idx = [f.index(n) for n in names] if idx is None else idx        # нет колонки — ValueError, тест падает
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow([v for i, v in enumerate(f) if i not in idx])
        out.append(buf.getvalue())
    return "\n".join(out).encode("utf-8")


def _itog_s_shapkoy(p: Path) -> tuple[list, dict]:
    lines = [ln for ln in p.read_text("utf-8").splitlines() if not ln.startswith("#")]
    head = lines[0].split(",")
    return head, dict(zip(head, lines[1].split(",")))


def test_umolchanie_train_pobaytno_staryy_put(tmp_path):
    P.run_strategy("S9", World().evaluate, lambda p: True, out=tmp_path, run_id="t", setka=SETKA)
    itog = tmp_path / "S9_itog.csv"
    assert {n: sha(tmp_path / f"S9_{n}.csv") for n in GOLDEN if n != "itog"} == {n: GOLDEN[n] for n in GOLDEN if n != "itog"}
    # (а) прежние колонки побайтно, кроме тех, что зависят от PBO: Ж-090 (#515/51867f11) сменил правило PBO (целевая — mSR
    #     за половину, годы > 0); на синтетике World центр плато даёт nan -> pbo пуст, G1 False, verdikt «не дошла: ворота».
    assert hashlib.sha256(_bez_novykh_kolonok(itog, PBO_ZAVISIT)).hexdigest() == GOLDEN_BEZ_PBO
    import csv
    rows0 = list(csv.reader(ln for ln in itog.read_text("utf-8").splitlines() if not ln.startswith("#")))
    row0 = dict(zip(rows0[0], rows0[1]))
    assert (row0["pbo"], row0["G1_pbo"], row0["ok"], row0["verdikt"]) == ("", "False", "False", "не дошла: ворота")
    head, row = _itog_s_shapkoy(itog)
    assert set(NOVYE_KOLONKI) <= set(head) and sha(itog) == ITOG_S_NOVYM                   # (б) новые колонки + полный хэш
    assert row["G1_pbo"] == str(row["pbo"] != "" and float(row["pbo"]) < V.PBO_MAX)                             # (в) G1 — от центра плато
    assert AR.paths("train") == (AR.IT1, AR.OUT, AR.LED) and AR.LED.name == "_ledger"


def test_g1_beret_tsentr_plato_a_ne_argmax(tmp_path, monkeypatch):
    """Центр плато 0,1 (< PBO_MAX = 0,2), argmax 0,9: G1 обязан быть True по центру; мутация «G1 по argmax» даёт False."""
    monkeypatch.setattr(P, "pbo", lambda M, pravilo=None: 0.1 if pravilo is not None else 0.9)
    P.run_strategy("S9", World().evaluate, lambda p: True, out=tmp_path, run_id="t", setka=SETKA)
    _, row = _itog_s_shapkoy(tmp_path / "S9_itog.csv")
    assert (float(row["pbo"]), float(row["pbo_argmax"])) == (0.1, 0.9) and row["G1_pbo"] == "True"


# --- итерация 1 на train4 --------------------------------------------------------------------------------------------------
def test_it1_train4_pishet_v_zh051_v1_itog_v2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = run4()
    d = tmp_path / "docs/tables/zh051_v1/it1"
    assert {f.name for f in d.iterdir()} == {"S9_morris.csv", "S9_faktor.csv", "S9_itog_v2.csv"}
    assert not (tmp_path / "docs/tables/zh047").exists()
    assert (d / "S9_itog_v2.csv").read_text("utf-8").startswith("# статус: разработка, В1 train 2016–2019")
    assert all(f"msr_{y}" in row for y in Y4) and row["morris_period"] == "train4"


def test_tselevaya_train4_po_chetyrem_godam(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = run4()
    assert row["obj_min_msr"] == pytest.approx(row["msr_2016"]) and row["obj_min_msr"] == pytest.approx(row["msr_2018"] - 2.0)


def test_mutaciya_tselevaya_po_dvum_godam_lovitsya(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orig = V.obj
    monkeypatch.setattr(P, "obj", lambda r, period=None: orig({**r, "msr_by_year": {y: m for y, m in r["msr_by_year"].items() if y >= 2018}}))
    monkeypatch.setattr(P, "obj_morris", lambda r, period=None: orig({**r, "msr_by_year": {y: m for y, m in r["msr_by_year"].items() if y >= 2018}}))
    with pytest.raises(AssertionError):
        test_tselevaya_train4_po_chetyrem_godam(tmp_path, monkeypatch)


def test_vorota_sdelok_v_kazhdom_godu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run4(World4(sdelok16=10))["verdikt"] == "не дошла: нет допустимой точки"      # 2016: 10 < 30 сделок
    assert run4(World4())["N_dopustimykh"] > 0


def test_mutaciya_vorota_ne_kazhdyy_god_lovitsya(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(V, "sdelok_ok", lambda sdy, period="train": bool(sdy) and all(v >= V.MIN_SDELOK for y, v in sdy.items() if y >= 2018))
    with pytest.raises(AssertionError):
        test_vorota_sdelok_v_kazhdom_godu(tmp_path, monkeypatch)


# --- итерация 2 на train4: пути и ledger -----------------------------------------------------------------------------------
class FakeEvr:
    def __init__(self, sid, inp, base, comm):
        self.w, self.sk = World4(), type("SK", (), {"sig_names": ("a",)})()

    def evaluate(self, pt, period):
        return self.w.evaluate(pt, period)

    def evaluate_ens(self, center, pts, mode, period="train"):
        return self.w.evaluate(center, period)


def run_it2(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(P, "load_setka", lambda sid, d=None: SETKA)
    monkeypatch.setattr(etap1, "build_inputs", lambda sid, *a, **k: None)
    monkeypatch.setattr(etap1, "base_point", lambda sid: {})
    monkeypatch.setattr(etap1, "anchor_s9_diff", lambda sid, evr, *a: {2018: 0.0})
    monkeypatch.setattr(AR, "AnsamblEvaluator", FakeEvr)
    monkeypatch.setattr(AR, "sverka_s9", lambda evr, sig: dict(rashozhdeniy=[], sovpalo=True))
    run4()
    return AR.run_sid("S9", "train4")


def test_it2_train4_ledger_v_ledger_v1_i_tablitsy_v1(tmp_path, monkeypatch):
    itog = run_it2(monkeypatch, tmp_path)
    t = tmp_path / "docs/tables"
    assert sorted(p.name for p in (t / "zh051/_ledger_v1/S9").iterdir()) == ["ledger_it1.pkl", "ledger_it2.pkl"]
    assert not (t / "zh051/_ledger").exists() and not (t / "zh047").exists() and not (tmp_path / "docs/otbor").exists()
    assert {f.name for f in (t / "zh051_v1/it2").iterdir()} == {"S9_ansambl.csv", "S9_itog.csv"}
    cache = pickle.loads((t / "zh051/_ledger_v1/S9/ledger_it2.pkl").read_bytes())
    assert {per for (_, per) in cache} <= {"train4", "train"} and sum(per == "train4" for (_, per) in cache) > len(cache) // 2
    assert any(k[0][0] == "ansambl" and k[1] == "train4" for k in cache)             # столбцы ансамбля — в матрице В1
    assert all(f"it1_msr_{y}" in itog for y in Y4) and itog["N_pbo"] > 20


def test_mutaciya_zapis_v1_v_ledger_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(AR, "paths", lambda period="train": (AR.IT1_V1, AR.OUT_V1, AR.LED))
    with pytest.raises((AssertionError, FileNotFoundError)):
        test_it2_train4_ledger_v_ledger_v1_i_tablitsy_v1(tmp_path, monkeypatch)
    assert (tmp_path / "docs/tables/zh051/_ledger/S9/ledger_it2.pkl").exists()       # мутация реально писала в _ledger
