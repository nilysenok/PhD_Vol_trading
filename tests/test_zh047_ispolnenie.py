"""Ж-047, тесты 2.1–2.5 (контракт §2): тейк по касанию (мутация № 2), стоп, порядок в баре, лимит, шорт.
Синтетика, сеть и данные не используются. Мутация: monkeypatch (`mutate` — подмена `_Run._scan`, либо
Params с тиком 0 = «касание»); сценарий на мутации обязан упасть AssertionError. До К2 — xfail(strict)."""
import pytest

from fastbt import exec_e
from test_zh047_pomoshch import (F, L_LIM, P, S_LONG, T_LONG, TICK, WAIT, D, first, go, mutate, wait)



def sc_take():                                                     # 2.1
    r = go(F(100.0, 2) + [(100, T_LONG, 100, 100)] + F(100.0))
    assert first(r).reason == "signal", "касание уровня тейка не выход"
    r = go(F(100.0, 2) + [(100, T_LONG + TICK, 100, 100)] + F(100.0))
    assert (first(r).reason, first(r).exit_px) == ("take", T_LONG)
    r = go([(114.0, 115.0, 113.0, 114.0)])
    assert (first(r).reason, first(r).exit_px) == ("take", 114.0), "гэп через тейк — по открытию"


def sc_stop():                                                     # 2.2
    r = go(F(100.0) + [(100, 100, S_LONG, 100)])
    assert (first(r).reason, first(r).exit_px) == ("stop", S_LONG), "low == S — стоп по S"
    r = go([(92.0, 93.0, 91.0, 92.0)])
    assert (first(r).reason, first(r).exit_px) == ("stop", 92.0), "гэп через стоп — по открытию"


def sc_order():                                                    # 2.3
    r = go([(100.0, T_LONG + 1.0, S_LONG - 1.0, 100.0)])
    assert (first(r).reason, first(r).exit_px) == ("stop", S_LONG)


def sc_limit():                                                    # 2.4 (лимит L = 100·(1 − k·σ), k = 1)
    kw = dict(p=P(k=1.0, R=None, b=None))
    assert go([(100, 100, L_LIM, 100)], **kw).trades == [], "касание лимита — нет сделки"
    r = go([(100, 100, L_LIM - TICK, 100)], **kw)
    assert first(r).entry_px == L_LIM and first(r).entry_day == 1
    assert first(go([(92.0, 93.0, 91.0, 92.0)], **kw)).entry_px == 92.0, "гэп под лимит — по открытию"
    assert go(F(100.0), **kw).trades == [], "неисполненный лимит — нет сделки"


def sc_short():                                                    # 2.5: S = P(1 + σ), T = P(1 − 2σ) = 87.5, L = 106.25
    kw = dict(p=P(side=-1))
    r = go([(100, 100, 87.5, 100)], **kw)
    assert first(r).reason == "signal", "шорт: касание тейка не выход"
    r = go([(100, 100, 87.5 - TICK, 100)], **kw)
    assert (first(r).reason, first(r).exit_px) == ("take", 87.5)
    r = go([(100, 106.25, 100, 100)], **kw)
    assert (first(r).reason, first(r).exit_px) == ("stop", 106.25)
    r = go([(108.0, 109.0, 107.0, 108.0)], **kw)
    assert first(r).exit_px == 108.0, "шорт: гэп через стоп — по открытию"
    kl = dict(p=P(side=-1, k=1.0, R=None, b=None))
    assert go([(100, 106.25, 100, 100)], **kl).trades == []
    assert first(go([(100, 106.25 + TICK, 100, 100)], **kl)).entry_px == 106.25


def sc_short_long_levels():                                        # шорт не должен считать уровни как лонг
    r = go([(100, 100, S_LONG, 100)], p=P(side=-1))
    assert not r.trades or first(r).reason != "stop", "для шорта low = 93.75 не стоп"


def tick0(monkeypatch):
    orig = exec_e.Params
    monkeypatch.setattr(exec_e, "Params", lambda **kw: orig(**{**kw, "tick": 0.0}))


def side_long(monkeypatch):
    orig = exec_e.Params
    monkeypatch.setattr(exec_e, "Params", lambda **kw: orig(**{**kw, "side": 1}))


@pytest.mark.parametrize("sc", [sc_take, sc_stop, sc_order, sc_limit, sc_short, sc_short_long_levels])
def test_pravilo_zeleno(sc):
    sc()


def shim_params(monkeypatch):
    """Самопроверка сценариев: Params дополнен полями вручную, проход — эталонный `scan_ref` без мутации."""
    orig = exec_e.Params

    def mk(**kw):
        extra = {k: kw.pop(k) for k in ("R", "h", "tick", "side", "ktr", "bu") if k in kw}
        p = orig(**kw)
        for k, v in extra.items():
            object.__setattr__(p, k, v)
        return p
    monkeypatch.setattr(exec_e, "Params", mk)
    mutate(monkeypatch, "")


@pytest.mark.parametrize("sc", [sc_take, sc_stop, sc_order])
def test_scenarii_soglasovany_s_etalonnym_prohodom(monkeypatch, sc):
    shim_params(monkeypatch)
    sc()


def test_mut_take_po_kasaniyu_tick0(monkeypatch):                  # мутация № 2
    tick0(monkeypatch)
    with pytest.raises(AssertionError):
        sc_take()


def test_mut_take_touch_in_scan(monkeypatch):
    mutate(monkeypatch, "take_touch")
    with pytest.raises(AssertionError):
        sc_take()


@pytest.mark.parametrize("mut", ["stop_strict", "gap_at_level"])   # `<` вместо `≤`; мутация № 4
def test_mut_stop(monkeypatch, mut):
    mutate(monkeypatch, mut)
    with pytest.raises(AssertionError):
        sc_stop()


def test_mut_stop_take_order(monkeypatch):                         # мутация № 3
    mutate(monkeypatch, "take_first")
    with pytest.raises(AssertionError):
        sc_order()


def test_mut_limit_touch_fills(monkeypatch):                       # мутация № 2 для лимита (❓3: один тик)
    tick0(monkeypatch)
    with pytest.raises(AssertionError):
        sc_limit()


def test_mut_short_levels_as_long(monkeypatch):
    side_long(monkeypatch)
    with pytest.raises(AssertionError):
        sc_short()


def test_mut_limit_checked_on_day_d():                             # лимит смотрит бары дня d, а не d+1
    days = [D([(100, 100, 100, 100), (100, 100, L_LIM - 1.0, 100), (100, 100, 100, 100)]),
            D(F(100.0, 2)), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 1, 0], [0.0625] * 3, P(k=1.0, R=None, b=None))
    assert r.trades == [], "бары дня d (где low ниже лимита) не исполняют заявку на d+1"


def sc_dolya():                                                    # 2.4: доля исполнения лимита = сделки / заявки
    days = [D(F(100.0)), D([(100, 100, L_LIM - TICK, 100)]), D(F(100.0)), D(F(100.0)), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 0, 1, 0, 0], [0.0625] * 5, P(k=1.0, R=None, b=None))
    assert len(r.trades) == 1 and r.dolya_ispolneniya == 0.5, "заявок 2, исполнена 1"


def test_dolya_ispolneniya():
    sc_dolya()


def test_mut_dolya_always_one(monkeypatch):
    orig = exec_e.run

    def lie(*a, **kw):
        r = orig(*a, **kw)
        r.dolya_ispolneniya = 1.0
        return r
    monkeypatch.setattr(exec_e, "run", lie)
    with pytest.raises(AssertionError):
        sc_dolya()


def test_mutatsii_lovyatsya_na_etalonnom_prohode_soglasovany(monkeypatch):
    """Живая проверка тестов уже сейчас: каждая мутация `scan_ref` роняет свой сценарий (пока нет кода К2)."""
    from test_zh047_vyhody import sc_bu, sc_trail
    cases = [("take_touch", sc_take), ("stop_strict", sc_stop), ("gap_at_level", sc_stop),
             ("take_first", sc_order), ("trail_same_bar", sc_trail), ("be_same_bar", sc_bu)]
    for mut, sc in cases:
        with monkeypatch.context() as m:
            shim_params(m)
            mutate(m, mut)
            with pytest.raises(AssertionError):
                sc()


for _n in [n for n in dir() if n.startswith("test_") and "soglasovany" not in n]:   # ждёт К2 (снять после слияния)
    globals()[_n] = wait(2)(globals()[_n])
