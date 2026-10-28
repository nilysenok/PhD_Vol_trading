"""Ж-047, тесты 2.1–2.5 (контракт §2): тейк по касанию (мутация № 2), стоп, порядок в баре, лимит, шорт, доля исполнения.
Синтетика, сеть и данные не используются. Каждый сценарий: зелёный на коде К2 (exec_e) и обязан упасть
(AssertionError) на мутации через monkeypatch правила `exec_e_run.*` или `Run._scan`."""
import pytest

from fastbt import exec_e, exec_e_run
from test_zh047_pomoshch import F, L_LIM, P, S_LONG, T_LONG, TICK, D, first, go, mutate, patch_rule


def sc_take():                                                     # 2.1
    r = go(F(100.0, 2) + [(100, T_LONG, 100, 100)] + F(100.0))
    assert first(r).reason == "signal", "касание уровня тейка не выход"
    r = go(F(100.0, 2) + [(100, T_LONG + TICK, 100, 100)] + F(100.0))
    assert (first(r).reason, first(r).exit_px) == ("take", T_LONG)
    r = go([(114.0, 115.0, 113.0, 114.0)])
    assert (first(r).reason, first(r).exit_px) == ("take", 114.0), "гэп через тейк — по открытию"


def sc_stop():                                                     # 2.2 (stop_touch=True: l ≤ S)
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


def sc_dolya():                                                    # 2.4: доля исполнения = сделки / заявки
    days = [D(F(100.0)), D([(100, 100, L_LIM - TICK, 100)]), D(F(100.0)), D(F(100.0)), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 0, 1, 0, 0], [0.0625] * 5, P(k=1.0, R=None, b=None))
    assert len(r.trades) == 1 and r.fill_rate == 0.5, "заявок 2, исполнена 1"


def sc_short():                                                    # 2.5: прямые правила шорта (side=−1, sig=1)
    kw = dict(p=P(side=-1))                                        # S = 100·(1+σ) = 106.25, T = 100·(1−2σ) = 87.5
    r = go([(100, 100, 87.5, 100)], **kw)
    assert first(r).side == -1 and first(r).reason == "signal", "шорт: касание тейка не выход"
    r = go([(100, 100, 87.5 - TICK, 100)], **kw)
    assert (first(r).reason, first(r).exit_px) == ("take", 87.5), "цель ниже входа, проход на тик"
    r = go([(100, 106.25, 100, 100)], **kw)
    assert (first(r).reason, first(r).exit_px) == ("stop", 106.25), "стоп выше входа на k_s·σ, касание"
    r = go([(100, 107.0, 80.0, 100)], **kw)
    assert (first(r).reason, first(r).exit_px) == ("stop", 106.25), "стоп и тейк в одном баре — первым стоп"
    r = go([(108.0, 109.0, 107.0, 108.0)], **kw)
    assert first(r).exit_px == 108.0, "шорт: гэп через стоп — по открытию"
    kl = dict(p=P(side=-1, k=1.0, R=None, b=None))                 # лимит шорта L = 106.25
    assert go([(100, 106.25, 100, 100)], **kl).trades == []
    assert first(go([(100, 106.25 + TICK, 100, 100)], **kl)).entry_px == 106.25


def sc_short_not_long_levels():
    r = go([(100, 100, S_LONG, 100)], p=P(side=-1))
    assert first(r).reason == "signal", "для шорта low = 93.75 не стоп"


@pytest.mark.parametrize("sc", [sc_take, sc_stop, sc_order, sc_limit, sc_dolya, sc_short, sc_short_not_long_levels])
def test_pravilo_zeleno(sc):
    sc()


def test_scan_copy_without_mutation_equals_real(monkeypatch):      # копия `_scan` без мутации = боевой код
    mutate(monkeypatch, "")
    for sc in (sc_take, sc_stop, sc_order, sc_short):
        sc()


def tick_none(f):                                                   # ослабление: тик 0 = касание считается
    return lambda *a: f(*a[:-1], 0.0)


def test_mut_take_po_kasaniyu(monkeypatch):                        # мутация № 2 (обязательна)
    patch_rule(monkeypatch, "take_hit", tick_none)
    with pytest.raises(AssertionError):
        sc_take()


def test_mut_take_po_kasaniyu_shortt(monkeypatch):
    patch_rule(monkeypatch, "take_hit", tick_none)
    with pytest.raises(AssertionError):
        sc_short()


def test_mut_stop_strict(monkeypatch):                             # `<` вместо `≤`
    patch_rule(monkeypatch, "stop_hit", lambda f: lambda sd, o, h, l, S, touch: f(sd, o, h, l, S, False))
    with pytest.raises(AssertionError):
        sc_stop()


def test_mut_stop_gap_at_level(monkeypatch):                       # мутация № 4: гэп по уровню, а не по открытию
    patch_rule(monkeypatch, "stop_hit", lambda f: lambda sd, o, h, l, S, touch: (f(sd, o, h, l, S, touch)[0], S))
    with pytest.raises(AssertionError):
        sc_stop()
    with pytest.raises(AssertionError):
        sc_short()


def test_mut_stop_take_order(monkeypatch):                         # мутация № 3
    mutate(monkeypatch, "take_first")
    with pytest.raises(AssertionError):
        sc_order()
    with pytest.raises(AssertionError):
        sc_short()


def test_mut_limit_touch_fills(monkeypatch):                       # мутация № 2 для лимита (❓3: один тик)
    patch_rule(monkeypatch, "limit_mask", lambda f: lambda sd, h, l, L, tick: f(sd, h, l, L, 0.0))
    with pytest.raises(AssertionError):
        sc_limit()


def test_mut_short_levels_as_long(monkeypatch):                    # уровни шорта со знаком лонга
    patch_rule(monkeypatch, "stop_level", lambda f: lambda sd, *a: f(1, *a))
    patch_rule(monkeypatch, "take_level", lambda f: lambda sd, *a: f(1, *a))
    with pytest.raises(AssertionError):
        sc_short()


def test_mut_fill_rate_always_one(monkeypatch):
    orig = exec_e.run

    def lie(*a, **kw):
        r = orig(*a, **kw)
        r.n_fills = r.n_orders
        return r
    monkeypatch.setattr(exec_e, "run", lie)
    with pytest.raises(AssertionError):
        sc_dolya()


def test_limit_not_filled_by_bars_of_day_d():                      # лимит смотрит бары d+1, а не дня d
    days = [D([(100, 100, 100, 100), (100, 100, L_LIM - 1.0, 100), (100, 100, 100, 100)]),
            D(F(100.0, 2)), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 1, 0], [0.0625] * 3, P(k=1.0, R=None, b=None))
    assert r.trades == [], "бары дня d (low ниже лимита) не исполняют заявку на d+1"
