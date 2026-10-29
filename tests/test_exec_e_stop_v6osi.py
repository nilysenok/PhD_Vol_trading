"""V6-11b (Ж-132, И16): безубыток b0 / b0_c, «стоп в плюс» sp, ступенчатый трейлинг step_s и пирамидинг pir на стоп-заявке тренда.
Известный ответ: вход buy-stop 102 (σ̂ = 0,01 → 1σ = 1,02), бары дня 16. Уровни действуют со следующего бара; гэп через стоп — по open (№ 4);
в одном баре стоп раньше добавки; цена в [low; high]. «Выкл.» = v5 побайтно — слепки в test_exec_e_stop_osi.py (оси по умолчанию).
Мутации: безубыток не включается; ось принята ядром быстрого слоя; b0_c без ранга принят.
python3 -m pytest tests/test_exec_e_stop_v6osi.py -q"""
import numpy as np
import pytest
from test_exec_e_stop_osi import LONG, N0, NAN, P, _day, _flat, _rnd

from fastbt import exec_e, exec_e_stop, exec_e_v6, fast_etap1, fast_exec


def _go(bars, p, rang=None):
    days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]
    n = len(days)
    u = {k: np.full(n, NAN) for k in ("up", "dn", "xl", "xs")}
    u["up"][N0 - 1], u["dn"][N0 - 1] = 102.0, 98.0
    return exec_e.run("T", days, np.zeros(n), np.full(n, 0.01), p, urovni=u, rang=rang)


UP = (102, 104, 102.6, 103.5)                                     # проход +1σ (103,02): уровни со следующего бара
RANG = np.zeros(N0 + 2)
RANG[N0 - 1] = 0.5


@pytest.mark.parametrize("p,kw", [
    (P(b=5.0, b0=1.0), {}),
    (P(b=5.0, b0_c=2.0), dict(rang=RANG)),                        # порог 2·0,5 = 1σ
])
def test_bezubytok(p, kw):
    t = _go([LONG, [UP, (103, 103, 101.9, 102)]], p, **kw).trades[0]
    assert (t.reason, t.exit_px, t.exit_bar) == ("bu", 102.0, 1)
    assert _go([LONG, [UP, (103, 103, 101.9, 102)]], P(b=5.0)).trades[0].reason is None


def test_bezubytok_gep_cherez_stop_po_open():
    t = _go([LONG, [UP, (101, 101.5, 100.5, 101)]], P(b=5.0, b0=1.0)).trades[0]
    assert (t.reason, t.exit_px) == ("bu", 101.0)                  # open 101 хуже уровня 102 (мутация № 4)


def test_bezubytok_ne_v_tom_zhe_bare():
    """Проход +1σ и возврат к входу в одном баре: уровень ещё не включён, выхода нет."""
    assert _go([LONG, [(102, 104, 101.5, 102.2)]], P(b=5.0, b0=1.0)).trades[0].reason is None


def test_stop_v_plyus():
    t = _go([LONG, [UP, (103, 103, 102.4, 102.9)]], P(b=5.0, sp=(1.0, 0.5))).trades[0]
    assert (t.reason, t.exit_px) == ("trail", pytest.approx(102 * 1.005, abs=1e-9))


def test_stupenchatyj_trejl():
    ent = [(101, 103, 101.5, 102.5)]                                  # low бара входа выше начального стопа 100,98
    t = _go([ent, [UP, (103, 103, 101.9, 102)]], P(b=1.0, step_s=1.0)).trades[0]       # 1 шаг: 100,98 + 1,02 = 102
    assert (t.reason, t.exit_px) == ("trail", pytest.approx(102.0, abs=1e-9))


def test_piramidin_dobavka_i_obshchij_vykhod():
    r = _go([LONG, [UP, (103, 103, 99, 100)]], P(b=2.0, pir=0.5))
    a, add = r.trades
    assert (add.reason, add.frac, add.parent, add.entry_day, add.entry_bar) == ("pir", 0.5, 0, N0 + 1, 0)
    assert add.entry_px == pytest.approx(102 * 1.01, abs=1e-9)
    assert a.exit_px == add.exit_px == pytest.approx(102 * 0.98, abs=1e-9) and a.reason == "stop"


def test_piramidin_stop_ranshe_dobavki():
    r = _go([LONG, [(102, 104, 99, 100)]], P(b=2.0, pir=0.5))
    assert len(r.trades) == 1 and r.trades[0].reason == "stop"       # в баре стоп первым: добавки нет


def test_b0_c_bez_ranga_ne_prinimaetsya():
    with pytest.raises(ValueError, match="rang"):
        _go([LONG], P(b=5.0, b0_c=2.0))
    with pytest.raises(ValueError):
        _go([LONG], P(b=5.0, b0=1.0, b0_c=2.0), rang=RANG)           # взаимоисключающие
    with pytest.raises(ValueError, match="b"):
        _go([LONG], P(b0=1.0))                                       # нужен b


def test_tsena_v_low_high_vse_osi():
    p = P(b=2.0, a=3.0, b0=1.0, sp=(1.5, 0.5), step_s=1.0, pir=0.5, ed_s="atr", chand=1.5, k_e=0.1, comm=0.0004)
    n = 0
    for seed in range(8):
        B, u = _rnd(seed)
        r = exec_e.run("T", B, np.zeros(len(B)), np.full(len(B), 0.012), p, urovni=u)
        for t in r.trades:
            for dy, br, px in ((t.entry_day, t.entry_bar, t.entry_px), (t.exit_day, t.exit_bar, t.exit_px)):
                if dy is not None and px != B[dy].c[-1]:
                    assert B[dy].l[br] - 1e-9 <= px <= B[dy].h[br] + 1e-9
                    n += 1
        assert abs(r.pos).max() <= 2.0 + 1e-9                         # риск пирамидинга ограничен
    assert n > 20


def test_mutaciya_bezubytok_ne_vklyuchaetsya_lovitsya(monkeypatch):
    bars, p = [LONG, [UP, (103, 103, 101.9, 102)]], P(b=5.0, b0=1.0)
    assert _go(bars, p).trades[0].reason == "bu"
    monkeypatch.setattr(exec_e_v6.V6, "v6_after", lambda s, sg, hi, lo: None)
    assert _go(bars, p).trades[0].reason != "bu"


def test_novye_osi_na_stop_zayavke_idut_etalonom(monkeypatch):
    for kw in (dict(b0=1.0), dict(sp=(1.0, 0.5)), dict(step_s=1.0), dict(pir=0.5)):
        p = P(b=2.0, **kw)
        assert fast_exec.ne_znaet(p) and not fast_etap1._yadro_v5(p, None, None, None), kw
        days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in [LONG, [UP, (103, 103, 99, 100)]]]
        u = {k: np.full(len(days), NAN) for k in ("up", "dn", "xl", "xs")}
        u["up"][N0 - 1], u["dn"][N0 - 1] = 102.0, 98.0
        sg = np.full(len(days), 0.01)
        a = exec_e.run("T", days, np.zeros(len(days)), sg, p, urovni=u)
        b = fast_etap1.run("T", days, np.zeros(len(days)), sg, p, urovni=u)
        assert a.trades and a.trades == b.trades and a.gross.tobytes() == b.gross.tobytes(), kw
    p = P(b=2.0, b0=1.0)
    monkeypatch.setattr(exec_e_stop, "stop_novye", lambda p: [])
    assert fast_exec.ne_znaet(p) == []                                  # подложенная ошибка: ядро берёт ось → тест выше упал бы
