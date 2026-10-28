"""К-В1a: драйвер итераций 3/4 с периодом train4 (В1: 2016–2019). Подменённый оценщик, данных нет.
Умолчание (train) пишет в старый путь; В1 — только в `_ledger_v1` и `zh051_v1`; критерий на train4 смотрит на все 4 года."""
import math
import zlib

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_poisk as P, etap1_poisk_vorota as V, zh051_iter as I, zh051_kriteriy as K
from fastbt.zh051_plato import ledger_cls


def _res(seed: int, years: tuple) -> dict:
    rng = np.random.default_rng(seed)
    x = {y: rng.normal(0.0004, 0.01, 252) for y in years}
    m = {y: float(v.mean() / v.std(ddof=1) * math.sqrt(252)) for y, v in x.items()}
    return {"v": "1", "excess_d": x, "msr_by_year": m, "by_year": m, "sdelok_y": {y: 60.0 for y in years},
            "storony_y": {y: 120.0 for y in years}, "vyrozhdena": False}


class FakeEvr:
    def __init__(self, sid, inp, base, comm):
        self.inp = inp

    def evaluate(self, p, per):
        kp = {k: v for k, v in p.items() if k != "kontrol"}
        return _res(zlib.crc32(repr(sorted(kp.items(), key=str)).encode()) % 10 ** 6, V.PERIOD_YEARS[per])


def _setup(period, tmp_path, monkeypatch):
    old, v1 = tmp_path / "old_led", tmp_path / "v1_led"
    p1dir = tmp_path / "p1"
    for d in (old, v1, p1dir):
        d.mkdir()
        if d != p1dir:
            (d / "SHA256SUMS.txt").write_text("# sha256  файл  N\n", encoding="utf-8")
    S = P.load_setka("S1")
    p1 = {k: S["yakor"][k] for k in S["axes"]}
    pd.DataFrame([{f"p_{k}": v for k, v in p1.items()}]).to_csv(p1dir / "S1_itog_v2.csv", index=False)
    monkeypatch.setattr(K, "LED", old)
    monkeypatch.setattr(I, "LED_V1", v1)
    monkeypatch.setattr(I, "OUT", tmp_path / "out_old")
    monkeypatch.setattr(I, "OUT_V1", tmp_path / "out_v1")
    monkeypatch.setattr(I, "P1_V1", p1dir)
    monkeypatch.setattr(I, "P1_OLD", p1dir)
    monkeypatch.setattr(I, "PlusEvaluator", FakeEvr)
    seen = []
    monkeypatch.setattr(I.etap1, "build_inputs", lambda sid, **kw: seen.append(kw) or object())
    monkeypatch.setattr(I.etap1, "base_point", lambda sid: {})
    monkeypatch.setattr(I.etap1, "anchor_s9_diff", lambda sid, evr: {"a": 0.0})
    led_dir = old if period == "train" else v1
    ev = lambda p, per: FakeEvr(None, None, None, None).evaluate({**I.YAKOR, **p}, per)      # noqa: E731
    L = ledger_cls(None)(ev)
    for i in range(6):
        L({**p1, "top_k": i + 1} if "top_k" in p1 else p1, period)
    L(p1, period)
    K.save_ledger("S1", L, "ledger_it2.pkl", led_dir)
    return old, v1, seen


def _run(period, tmp_path, monkeypatch):
    old, v1, seen = _setup(period, tmp_path, monkeypatch)
    I.run_it("S1", 3, period) if period != "train" else I.run_it("S1", 3)
    return {"old": (old / "S1" / "ledger_it3.pkl").exists(), "v1": (v1 / "S1" / "ledger_it3.pkl").exists(),
            "tab_v1": (tmp_path / "out_v1" / "it3" / "S1_itog.csv").exists(),
            "tab_old": (tmp_path / "out_old" / "it3" / "S1_itog.csv").exists()}, seen, tmp_path


def _kolonki(path):
    return pd.read_csv(path, comment="#").columns


def test_train4_pishet_tolko_v_ledger_v1(tmp_path, monkeypatch):
    got, seen, _ = _run("train4", tmp_path, monkeypatch)
    assert got == {"old": False, "v1": True, "tab_v1": True, "tab_old": False}
    assert seen == [{"period": "train4"}]                                    # входы строятся с прогревом 2015
    cols = _kolonki(tmp_path / "out_v1" / "it3" / "S1_itog.csv")
    assert {f"msr_{y}" for y in (2016, 2017, 2018, 2019)} <= set(cols)       # mSR по 4 годам, sdelok_* тоже в po_godam


def test_umolchanie_train_staryy_put_pobaytno_po_kolonkam(tmp_path, monkeypatch):
    got, seen, _ = _run("train", tmp_path, monkeypatch)
    assert got == {"old": True, "v1": False, "tab_v1": False, "tab_old": True}
    assert seen == [{}]                                                      # build_inputs(sid) как раньше
    cols = _kolonki(tmp_path / "out_old" / "it3" / "S1_itog.csv")
    assert "msr_2018" in cols and "msr_2016" not in cols


def test_mutaciya_zapis_v1_v_staryy_ledger_lovitsya(tmp_path, monkeypatch):
    old, v1, _ = _setup("train4", tmp_path, monkeypatch)
    L = K.load_ledger("S1", "ledger_it2.pkl", None, v1)
    K.save_ledger("S1", L, "ledger_it2.pkl", old)                             # it2 лежит и в старом каталоге
    monkeypatch.setattr(I, "LED_V1", old)                                     # подложено: В1 пишет в `_ledger`
    I.run_it("S1", 3, "train4")
    wrote_old = (old / "S1" / "ledger_it3.pkl").exists()
    assert wrote_old                                                          # мутация сработала...
    with pytest.raises(AssertionError):
        assert not wrote_old                                                  # ...а инвариант «В1 не в `_ledger`» её видит


def test_paths_train4_ne_old_led():
    led, out, _ = I.paths("train4")
    assert led == I.LED_V1 and led != K.LED and out != I.OUT and "_ledger_v1" in led.parts
    assert I.paths("train")[0] is None and I.paths("train")[1] == I.OUT
    with pytest.raises(ValueError):
        I.paths("validation")


def _krit(msr_pi, msr_p1, years):
    n = len(years)
    d = dict(tsel_pi=1.0, tsel_p1=0.5, msr_pi=msr_pi, msr_p1=msr_p1, nb_tsel=[], plato_ok=True,
             sdelok_pi={y: 60 for y in years}, s9_ok=True, g4_ok=True, pbo_do=0.1, pbo_posle=0.1, kontrol=None)
    return K.kriteriy(d, years), n


def test_kriteriy_train4_smotrit_na_vse_4_goda():
    Y4 = I.YEARS["train4"]
    pi = {2016: -1.0, 2017: 1.0, 2018: 1.0, 2019: 1.0}                       # 2016 хуже P1 на > 0,1
    p1 = {y: 0.5 for y in Y4}
    r, _ = _krit(pi, p1, Y4)
    assert r["P2"] is False and r["proshla"] is False
    # мутация «критерий смотрит только 2018/2019» пропустила бы Pi — тест обязан это видеть
    assert K.kriteriy(dict(tsel_pi=1.0, tsel_p1=0.5, msr_pi=pi, msr_p1=p1, nb_tsel=[], plato_ok=True, sdelok_pi={y: 60 for y in Y4},
                           s9_ok=True, g4_ok=True, pbo_do=0.1, pbo_posle=0.1, kontrol=None), (2018, 2019))["P2"] is True


def test_kriteriy_umolchanie_dva_goda_i_net_goda_padaet():
    r, _ = _krit({2018: 1.0, 2019: 1.0}, {2018: 0.5, 2019: 0.5}, (2018, 2019))
    assert r["P2"] is True
    with pytest.raises(KeyError):                                            # на train4 нет 2016 в данных — молча не пройдёт
        _krit({2018: 1.0, 2019: 1.0}, {2018: 0.5, 2019: 0.5}, I.YEARS["train4"])


def test_main_otkazyvaet_it10_na_v1():
    with pytest.raises(SystemExit, match="не поддержана"):
        I.main(["--it", "10", "--strategii", "S1", "--period", "train4"])
