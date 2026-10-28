"""П-2 (проверяющий): оси исполнения Л-2 по контракту progon_etap1 §1.1 — синтетика с известным ответом.
Код Л-2 (fastbt/exec_e2.py, новые поля Params, run(..., ed=)) пишется параллельно: до него тесты красные.
Мутации (подкладываются в отдельной копии): лимит на баре >= N при zhizn; гэп исполнен при propusk; ed не меняет L;
bar_plus берёт open того же бара; stop_tik в лучшую сторону; касание = исполнение (№2); тейк раньше стопа (№3);
гэп по уровню (№4); комиссия выхода на конце окна не взята (№5)."""
import hashlib

import numpy as np
import pytest

from fastbt.exec_e import Day, Params, run

C = 0.0004
SG = 0.02
L = 100.0 * (1 - 0.5 * SG)        # 99


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False)


def F(x, n=1):
    return [(x, x, x, x)] * n


def go(days, sig, sg=SG, ed=None, **kw):
    s = np.full(len(days), sg)
    if ed is None:
        return run("T", days, sig, s, Params(**kw))
    return run("T", days, sig, s, Params(**kw), ed=np.asarray(ed, float))


def fill_day(b, n, low=98.0):
    """День из n плоских баров по 100; на баре b low = low (лимит 99 пробит, касаний нет)."""
    rows = F(100.0, n)
    rows[b] = (100.0, 100.0, low, 100.0)
    return D(rows)


# --- zhizn: исполнение только на барах b < N ---
@pytest.mark.parametrize("N", [12, 36, 60])
def test_zhizn_granitsa_obe_storony(N):
    d0 = D(F(100.0))
    t = go([d0, fill_day(N - 1, 70), D(F(100.0))], [1, 0, 0], k=0.5, zhizn=N).trades
    assert len(t) == 1 and t[0].entry_px == L and t[0].entry_bar == N - 1
    assert go([d0, fill_day(N, 70), D(F(100.0))], [1, 0, 0], k=0.5, zhizn=N).trades == []
    assert go([d0, fill_day(N, 70), D(F(100.0))], [1, 0, 0], k=0.5).trades[0].entry_bar == N    # умолчание: весь день


def test_zhizn_short_mirror_and_block():
    d0, hi = D(F(100.0)), 102.0
    rows = F(100.0, 20)
    rows[12] = (100.0, hi, 100.0, 100.0)
    assert go([d0, D(rows), D(F(100.0)), D(F(100.0))], [-1, -1, -1, 0], k=0.5, zhizn=12).trades == []   # блок до смены sig
    rows = F(100.0, 20)
    rows[11] = (100.0, hi, 100.0, 100.0)
    t = go([d0, D(rows), D(F(100.0))], [-1, 0, 0], k=0.5, zhizn=12).trades[0]
    assert (t.side, t.entry_px, t.entry_bar) == (-1, 101.0, 11)


# --- gap = propusk ---
def test_gap_propusk_long_snyata_i_blok():
    gap = D([(98.9, 99.5, 98.0, 99.0)] + F(100.0, 3))
    days = [D(F(100.0)), gap, D(F(100.0)), D(F(100.0))]
    assert go(days, [1, 1, 1, 1], k=0.5, tick=0.01, gap_propusk=True).trades == []     # o0 <= L − tick, блок sd
    t = go(days, [1, 0, 0, 0], k=0.5, tick=0.01).trades[0]                             # умолчание: по open
    assert t.entry_px == 98.9 and t.entry_bar == 0


def test_gap_propusk_ne_gap_ispolnyaetsya():
    d1 = D([(99.5, 99.5, 98.0, 99.0)] + F(100.0))     # open выше L: обычное исполнение по L
    t = go([D(F(100.0)), d1, D(F(100.0))], [1, 0, 0], k=0.5, tick=0.01, gap_propusk=True).trades[0]
    assert t.entry_px == L


def test_gap_propusk_short():
    d1 = D([(101.1, 102.0, 100.5, 101.0)] + F(100.0))     # шорт: L = 101, o0 >= L + tick
    assert go([D(F(100.0)), d1, D(F(100.0))], [-1, 0, 0], k=0.5, tick=0.01, gap_propusk=True).trades == []
    assert go([D(F(100.0)), d1, D(F(100.0))], [-1, 0, 0], k=0.5, tick=0.01).trades[0].entry_px == 101.1


# --- ed: единица лимита; стоп/тейк/размер по σ̂ ---
def test_ed_dvigaet_tolko_limit():
    days = [D(F(100.0)), D(F(100.0) + [(100, 100, 98.5, 100)] + F(100.0)), D(F(100.0))]
    assert go(days, [1, 0, 0], k=0.5, ed=[0.04] * 3).trades == []          # L = 98, low 98.5 не дошёл
    assert go(days, [1, 0, 0], k=0.5, ed=[0.01] * 3).trades[0].entry_px == 99.5     # L = 100·(1 − 0.5·0.01)
    days[1] = D(F(100.0) + [(100, 100, 90.0, 100)] + F(100.0))
    t = go(days, [1, 1, 1], k=0.5, b=1.0, ed=[0.04] * 3).trades[0]
    assert t.entry_px == 98.0 and t.reason == "stop" and t.exit_px == pytest.approx(98.0 * (1 - SG))   # стоп от σ̂, не от ed


def test_ed_shortovyj_limit():
    d1 = D(F(100.0) + [(100, 101.2, 100, 100)])
    t = go([D(F(100.0)), d1, D(F(100.0))], [-1, 0, 0], k=0.5, ed=[0.01] * 3).trades[0]
    assert t.entry_px == pytest.approx(100.5)


def test_ed_beret_den_signala_d():
    days = [D(F(100.0)), D(F(100.0) + [(100, 100, 98.5, 100)]), D(F(100.0))]
    assert go(days, [1, 0, 0], k=0.5, ed=[0.04, 0.0, 0.0]).trades == []    # ed[d], а не ed[d+1]
    assert len(go(days, [1, 0, 0], k=0.5, ed=[0.0, 0.04, 0.04]).trades) == 1


# --- bar_plus: исполнение по open следующего бара, если он хуже уровня ---
def bp_days(o_next):
    return [D(F(100.0)), D([(100, 100, 100, 100), (100, 100, 98.5, 99.5), (o_next, o_next, o_next, o_next), (100, 100, 100, 100)]),
            D(F(100.0))]


def test_bar_plus_limit_hudshij_open():
    assert go(bp_days(99.4), [1, 0, 0], k=0.5, bar_plus=1).trades[0].entry_px == 99.4     # покупка выше L — хуже
    assert go(bp_days(98.7), [1, 0, 0], k=0.5, bar_plus=1).trades[0].entry_px == L        # ниже L — лучше: по уровню
    assert go(bp_days(99.4), [1, 0, 0], k=0.5).trades[0].entry_px == L


def bp_exit(o_next, **kw):
    rows = F(100.0) + [(100, 102.5, 100, 102.0), (o_next, o_next, o_next, o_next), (101, 101, 101, 101)]
    return go([D(F(100.0)), D(rows), D(F(100.0))], [1, 1, 1], a=1.0, **kw).trades[0]


def test_bar_plus_tejk():
    assert bp_exit(101.5, bar_plus=1).exit_px == 101.5     # T = 102; продажа ниже — хуже
    assert bp_exit(102.4, bar_plus=1).exit_px == 102.0     # выше — лучше: по уровню
    assert bp_exit(101.5).exit_px == 102.0


def test_bar_plus_stop_i_poslednij_bar():
    rows = F(100.0) + [(100, 100, 97.5, 98.0), (97.0, 97.0, 97.0, 97.0), (99, 99, 99, 99)]
    t = go([D(F(100.0)), D(rows), D(F(100.0))], [1, 1, 1], b=1.0, bar_plus=1).trades[0]
    assert (t.reason, t.exit_px) == ("stop", 97.0)         # S = 98, следующий open хуже
    last = D([(100, 100, 100, 100), (100, 100, 97.5, 98.5)])       # стоп на последнем баре дня → close Н6
    t = go([D(F(100.0)), last, D(F(100.0))], [1, 1, 1], b=1.0, bar_plus=1).trades[0]
    assert (t.reason, t.exit_px) == ("stop", 98.5)


# --- stop_tik ---
def test_stop_tik_huzhe_na_tik():
    days = [D(F(100.0)), D(F(100.0) + [(100, 100, 97.5, 98.5)]), D(F(100.0))]
    assert go(days, [1, 1, 1], b=1.0, tick=0.01, stop_tik=1).trades[0].exit_px == pytest.approx(97.99)
    assert go(days, [1, 1, 1], b=1.0, tick=0.01).trades[0].exit_px == pytest.approx(98.0)
    days[1] = D(F(100.0) + [(100, 102.5, 100, 101.5)])
    t = go(days, [-1, -1, -1], b=1.0, tick=0.01, stop_tik=1).trades[0]
    assert t.exit_px == pytest.approx(102.01)               # шорт: стоп 102 + тик


# --- валидация ---
@pytest.mark.parametrize("kw", [dict(t_vyhod="1600"), dict(zhizn=7), dict(bar_plus=-1), dict(stop_tik=-1)])
def test_check_params_otkazyvaet(kw):
    with pytest.raises(ValueError):
        go([D(F(100.0)), D(F(100.0))], [1, 0], k=0.5, **kw)


# --- умолчания побайтно ---
def lenta(seed):
    r = np.random.default_rng(seed)
    days, px = [], 100.0
    for _ in range(14):
        n = int(r.integers(6, 20))
        c = px * np.exp(np.cumsum(r.normal(0, 0.006, n)))
        o = np.r_[px, c[:-1]] * (1 + r.normal(0, 0.002, n))
        h, lo = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
        days.append(Day(o, h, lo, c, False))
        px = float(c[-1])
    p = Params(k=0.5, b=1.0, a=1.5, tau=int(r.integers(2, 5)), f=["propusk", "zakrytie_d1", "povtor"][seed % 3], comm=C)
    return days, r.integers(-1, 2, 14), np.full(14, 0.01 + 0.001 * (seed % 5)), p


def digest(res):
    h = hashlib.sha256()
    for a in (res.gross, res.cost, res.pos, res.free, res.fdr):
        h.update(np.ascontiguousarray(a, float).tobytes())
    for t in res.trades:
        h.update(repr((t.d_sig, t.entry_day, t.entry_bar, t.entry_px, t.exit_day, t.exit_bar, t.exit_px, t.reason, t.side)).encode())
    h.update(repr((res.n_orders, res.n_fills, res.x_orders, res.x_fills)).encode())
    return h.hexdigest()


def tridcat():
    h = hashlib.sha256()
    for s in range(30):
        days, sig, sg, p = lenta(s)
        h.update(digest(run("T", days, sig, sg, p)).encode())
    return h.hexdigest()


ETALON = "67195863344e622e190c34be30e1bea0c6876276e8c3cce7322695629134cf24"           #sha256 по 30 лентам прежнего exec_e.run (снят на 91d72c9 до Л-2)


def test_umolchaniya_pobajtno_prezhnij_run():
    assert tridcat() == ETALON


def test_ed_ravno_sigma_pobajtno_kak_bez_ed():
    h = hashlib.sha256()
    for s in range(30):
        days, sig, sg, p = lenta(s)
        h.update(digest(run("T", days, sig, sg, p, ed=sg)).encode())
    assert h.hexdigest() == ETALON
