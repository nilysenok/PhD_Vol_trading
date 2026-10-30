"""wfП-8: два блокера эталона wf. (1) конец окна: ряд кончается 2026-02-27, граница 2026-02-28/03-01 — i1 = последний день ряда,
а не len (IndexError в build_days). train4: ряд идёт дальше границы, i1 — первый день >= границы (строка Н5) — без изменений.
(2) эталон wf (ekviv.poschitat, ekviv_v6.poschitat) включает зону wf на время счёта и возвращает её. Синтетика, данных и сети нет.
Мутации: «close[i1] без обрезки» и «эталон wf без зоны» ловятся."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import ekviv  # noqa: E402
import ekviv_v6 as e6  # noqa: E402
from fastbt import etap1, etap1_data, period_v2, zona  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402

DNI = pd.bdate_range("2026-02-16", "2026-02-27").to_numpy("datetime64[ns]")      # ряд кончается 27.02 (пятница)


@pytest.mark.parametrize("end", [pd.Timestamp("2026-02-28"), pd.Timestamp("2026-03-01")])
def test_konec_okna_posledniy_den_ryada(end):
    s0, a0, i1 = etap1.window_idx(DNI, pd.Timestamp("2026-02-16"), 2026, end)
    assert i1 == len(DNI) - 1 and DNI[i1] == np.datetime64("2026-02-27")


def test_train4_pervyi_den_ne_menshe_granitsy():
    d = pd.bdate_range("2019-12-20", "2020-01-20").to_numpy("datetime64[ns]")
    i1 = etap1.window_idx(d, pd.Timestamp("2019-12-20"), 2019, pd.Timestamp("2020-01-01"))[2]
    assert d[i1] == np.datetime64("2020-01-01")                  # первый день >= границы: Н5, как раньше


def test_build_days_wf_ne_padaet(monkeypatch):
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)
    bar = pd.DataFrame(dict(date=["2026-02-17", "2026-02-17"], ts=["2026-02-17 10:00", "2026-02-17 10:05"], open=[100.0, 100.1],
                            high=[100.2, 100.3], low=[99.9, 100.0], close=[100.1, 100.2], pusto=[False, False]))
    monkeypatch.setattr(etap1_data, "load_bars5m", lambda *a, **k: bar)
    c = np.arange(100.0, 100.0 + len(DNI))
    end = pd.Timestamp("2026-02-28")
    s0, _, i1 = etap1.window_idx(DNI, pd.Timestamp("2026-02-16"), 2026, end)
    with zona.zone_ctx("wf"):
        days, _ = etap1_data.build_days("AFLT", DNI, c, s0, i1, end, start=pd.Timestamp("2026-02-16"), last_year=2026)
    assert len(days) == i1 - s0 + 1 and float(days[-1].c[0]) == c[-1]


class Fake:
    fdr = "t0d"

    def __init__(self):
        self.zony = []

    def full(self, d):
        return dict(d, a=1)

    def evaluate_v2(self, pt, period):
        self.zony.append(zona.name())
        if period == "wf" and zona.name() != "wf":
            raise PermissionError("данные ≥ 2025 закрыты")           # как data.daily_df вне зоны wf
        rng = np.random.default_rng(len(self.zony))
        p = rng.normal(0, 0.01, 6)
        return {"status": "ok", "sharpe": float(p.mean() / p.std())}, p, rng.normal(0, 0.1, 3)


def _sreda(monkeypatch, ev, daty):
    ps._W.clear()
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(ev, "kod1", f"vh_{sid}", daty, ["AFLT"]))
    monkeypatch.setattr(ps, "fast_ok", lambda pt: False)
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)


@pytest.fixture
def sreda(monkeypatch):
    ev = Fake()
    _sreda(monkeypatch, ev, np.array(["2026-01-05", "2026-02-27"], dtype="datetime64[D]"))
    yield ev
    ps._W.clear()
    zona.set_zone("dev")


def test_ekviv_wf_vklyuchaet_zonu_i_vozvrashchaet(sreda):
    rez, meta = ekviv.poschitat([], 1, "wf")
    assert meta["period"] == "wf" and len(rez) == 6 and set(sreda.zony) == {"wf"}
    assert zona.name() == "dev"


def test_ekviv_v6_wf_vklyuchaet_zonu_i_vozvrashchaet(sreda):
    rez, meta, kod = e6.poschitat(None, set(), 1, 12, "wf")
    assert any(m.startswith("v6yakor__") for m in rez) and set(sreda.zony) == {"wf"}
    assert zona.name() == "dev"


def test_zona_vozvrashchaetsya_pri_oshibke(sreda, monkeypatch):
    monkeypatch.setattr(sreda, "evaluate_v2", lambda pt, period: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        ekviv.poschitat([], 1, "wf")
    assert zona.name() == "dev"


def test_train4_zona_ne_menyaetsya(monkeypatch):
    ev = Fake()
    _sreda(monkeypatch, ev, np.array(["2016-01-04", "2016-01-09"], dtype="datetime64[D]"))
    ekviv.poschitat([], 1, "train4")
    assert set(ev.zony) == {"dev"}
    ps._W.clear()
