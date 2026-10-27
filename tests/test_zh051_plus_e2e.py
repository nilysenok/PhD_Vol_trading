"""Сквозной тест Ж-051 на синтетике: итерации 3 и 4 через PlusEvaluator с настоящими `e1_portfel` и `e1_sektor` (К3, К4).
net → побайтно родитель; top_k=1 оставляет одну сделку на старте; сектор по тренду пропускает/режет стороны."""
import numpy as np
import pandas as pd

from fastbt.e1_portfel import ball_sily
from fastbt.e1_sektor import Ctx
from test_zh051_plus import DATES, N, POS, PT, STR, TICKS, FakeInp, FakeParent, FakePlus, _rows, spy  # noqa: F401


def _sig(calls):
    return {c["t"]: c["sig"] for c in calls}


def test_net_s_nastoyaschimi_modulyami_ravno_roditelyu_it3_it4(spy):
    _rows(FakeParent, PT)
    ref, spy[:] = list(spy), []
    _rows(FakePlus, {**PT, "top_k": "net", "sektor_trend": "net", "shirina": "net", "ma_n": 50})
    assert len(ref) == len(spy) == 3
    assert all(a["sig"].tobytes() == b["sig"].tobytes() and a["w"].tobytes() == b["w"].tobytes() for a, b in zip(ref, spy))


def test_top_k_1_nastoyaschiy_raspredelitel_ostavlyaet_odnu_sdelku(spy):
    _rows(FakePlus, {**PT, "top_k": 1})
    by = _sig(spy)
    for d in (0, 20, 40):                                          # старт трёх сделок в один день, слот один
        assert sum(int(by[t][d] != 0) for t in TICKS) == 1
    b = len(by["AAA"]) - 1
    inp, sg = FakeInp(), {t: FakeInp().sigma["ewma094"][t][:b + 1] for t in TICKS}
    score = {t: ball_sily(STR[t][:b + 1], sg[t])[0] for t in TICKS}
    win = max(score, key=score.get)
    assert by[win][0] != 0 and inp is not None                     # выиграл тикер с наибольшим баллом
    assert by[win][:6].tolist() == POS[win][:6].tolist()            # сделка сохранена целиком, не по частям


def _ctx(trend_up: bool):
    s = pd.Series(np.arange(N, dtype=float) if trend_up else -np.arange(N, dtype=float), index=pd.DatetimeIndex(DATES))
    return Ctx({"MOEXMM": s}, {t: pd.Series(100.0 + np.arange(N), index=pd.DatetimeIndex(DATES)) for t in TICKS},
               {t: "MOEXMM" for t in TICKS})


def _run(spy, mode, up):
    e = FakePlus()
    e.ctx = _ctx(up)
    e._rows(e.full({**PT, "sektor_trend": mode, "ma_n": 5}), "train", False)
    return _sig(spy)


def test_sektor_po_trendu_i_protiv_nastoyaschiy_modul(spy):
    by = _run(spy, "po_trendu", up=True)                           # рост: лонг разрешён, шорт — нет; день 0 — нет истории
    assert not by["AAA"][:6].any() and (by["AAA"][20:26] == 1).all() and not by["BBB"].any()
    spy[:] = []
    by = _run(spy, "protiv", up=True)                              # против тренда: лонг запрещён, шорт разрешён
    assert not by["AAA"].any() and (by["BBB"][20:26] == -1).all()
    spy[:] = []
    by = _run(spy, "po_trendu", up=False)                          # падение: шорт разрешён
    assert (by["BBB"][20:26] == -1).all() and not by["AAA"].any()


def test_kontrol_sdvig_ne_dvigaet_progrev_dolya_dney_sohranyaetsya():
    n, warm = 700, 200
    lg = np.r_[np.zeros(warm, bool), np.tile([True, True, False, True, False, False, True, False, True, False], 50)]
    sh = np.r_[np.zeros(warm, bool), ~lg[warm:]]
    pt = {"kontrol": ("krug_sdvig", 63)}
    a, b = PE_sdvig(pt, lg, sh)
    win = slice(warm, n)
    assert abs(lg[win].mean() - a[win].mean()) < 0.01 and abs(sh[win].mean() - b[win].mean()) < 0.01
    assert not a[:warm].any() and not b[:warm].any() and not np.array_equal(a, lg)
    old = np.roll(lg, 63)                                          # мутация «сдвиг от начала»: прогрев въезжает в окно
    assert abs(lg[win].mean() - old[win].mean()) > 0.01


def PE_sdvig(pt, lg, sh):
    from fastbt.etap1_plus_eval import PlusEvaluator
    return PlusEvaluator._kontrol_sdvig(pt, lg, sh)


def test_kontrol_sdvig_menyaet_maski(spy):
    base = _run(spy, "po_trendu", up=True)["AAA"].copy()
    spy[:] = []
    e = FakePlus()
    e.ctx = _ctx(True)
    e._rows(e.full({**PT, "sektor_trend": "po_trendu", "ma_n": 5, "kontrol": ("krug_sdvig", 60)}), "train", False)
    got = _sig(spy)["AAA"]
    assert not base[:6].any() and not got[:6].any()                 # прогрев остаётся запретом (не едет в окно)
    assert (got[60:66] == 1).all()                                  # валидная часть (всегда «вверх») сдвигом не меняется
