"""К-Ф3 (Ж-065 п. 3): выход лимиткой (Params.vyhod_lim), контракт §1.3 В, мутации М-В1…М-В6 (§1.5).
Лонг, σ = 1/16, k = 1, тик 0,25. Вход лимитом L = 93,75 на d1 (бар 0); сигнал 0 на close d1 (94) → заявка выхода
X = ceil_tick(94·1,0625) = 100,0; исполнение при h ≥ 100,25 (проход на тик). Синтетика, данные не читаются.
python3 -m pytest tests/test_vyhod_limit.py -q"""
import numpy as np
import pytest

from fastbt import etap1_run, exec_e, exec_e_run as R_
from fastbt.etap1_eval import YAKOR, is_anchor, params_of, v11
from test_zh047_pomoshch import D, F, P, SG

C = 0.0004
ENTRY = [(100.0, 100.0, 93.5, 94.0)]           # вход по L = 93,75; close d1 = 94
FILL = [(94.0,) * 4, (94, 100.25, 94, 100)]    # h = X + тик → исполнено по X = 100,0
TOUCH = [(94.0,) * 4, (94, 100.0, 94, 99)]     # h = X ровно → касание, не исполнено
BOTH = [(94, 101.0, 87.0, 90)]                 # стоп (87,89) и X в одном баре
GAP = [(84.0,) * 4]                            # d2 после отсечки D = 10: цена упала на D
FLAT = F(100.0)
SIG = (1, 0, 0, 0)


def pr(lim=True, **kw):
    return P(k=1.0, R=None, b=kw.pop("b", None), reentry=False, vyhod_lim=lim, **kw)


def go(d2, p=None, sig=SIG, div=None, nalog=1.0, d1=ENTRY):
    days = [D(FLAT), D(d1), D(d2), D(FLAT)]
    return exec_e.run("T", days, list(sig), np.full(4, SG), p or pr(), div=div, nalog=nalog)


def chk_fill():
    r = go(FILL)
    t = r.trades[0]
    assert (t.reason, t.exit_day, t.exit_bar, t.exit_px) == ("lim", 2, 1, 100.0)
    assert (r.x_orders, r.x_fills, r.n_orders, r.n_fills) == (1, 1, 1, 1)
    assert r.cost[1] == pytest.approx(C) and r.cost[2] == 0.0, "комиссия исполнения внутри дня — в строке начала интервала"


def chk_touch():
    r = go(TOUCH)
    t = r.trades[0]
    assert (t.reason, t.exit_day, t.exit_px) == ("signal", 2, 99.0), "касание не исполняет: выход по close d+1"
    assert (r.x_orders, r.x_fills) == (1, 0) and r.cost[2] == pytest.approx(C)


def chk_not_same_day():
    """М-В2: бар решения (d1) уже проходит X, но заявка действует только с d2."""
    r = go(FILL, d1=[(100.0, 101.0, 93.5, 94.0)])
    assert r.trades[0].exit_day == 2


def chk_stop_first():
    r = go(BOTH, p=pr(b=1.0, stop_touch=True))
    t = r.trades[0]
    assert t.reason == "stop" and t.exit_px == pytest.approx(93.75 * (1 - SG)) and r.x_fills == 0


def chk_dividend():
    r = go(GAP, p=pr(b=1.0, stop_touch=True), div=np.array([0, 0, 10.0, 0]), nalog=0.87)
    t = r.trades[0]
    assert t.reason == "signal" and t.exit_day == 2, "стоп сдвинут на D: 77,89 не задет"
    assert r.gross.sum() == pytest.approx(np.log((84 + 8.7) / 93.75)), "выход получает D·0,87"


def chk_side():
    """М-В3: шорт (side = −1, вход лимитом выше close): X ниже close, зеркально."""
    spy = []
    orig = R_.Run._leave

    def leave(s, i, j, nb, close, reason):
        orig(s, i, j, nb, close, reason)
        spy.append((s.tr.side, close, s.xo["X"], s.xo["due"], j))
    R_.Run._leave = leave
    try:
        go(FILL)
        days = [D(FLAT), D([(100.0, 106.5, 100.0, 106.0)]), D([(106.0,) * 4]), D(FLAT)]
        exec_e.run("T", days, [1, 0, 0, 0], np.full(4, SG), pr(side=-1))
    finally:
        R_.Run._leave = orig
    (sd1, c1, x1, due1, j1), (sd2, c2, x2, _, _) = spy
    assert sd1 == 1 and x1 > c1 and due1 == j1 + 1, "лонг: X выше close, due = j + 1 (М-В2)"
    assert sd2 == -1 and x2 < c2, "шорт: X ниже close"


def chk_gap_open():
    """Гэп через X вверх (open 102 > X): исполнение по open, а не по уровню 100 (mutations.md № 4)."""
    r = go([(102.0, 103.0, 101.0, 102.0)])
    t = r.trades[0]
    assert (t.reason, t.exit_day, t.exit_bar) == ("lim", 2, 0) and t.exit_px == 102.0 and r.x_fills == 1


def chk_m6():
    for kw in (dict(k=None), dict(k=1.0, check="zakrytie")):
        try:
            go(FILL, p=P(R=None, b=None, vyhod_lim=True, **kw))
            err = None
        except Exception as e:                       # noqa: BLE001 — при мутации падает не ValueError, а что-то позже
            err = e
        assert isinstance(err, ValueError), f"{kw}: ждали ValueError, получено {err!r}"


CHECKS = [chk_fill, chk_touch, chk_not_same_day, chk_stop_first, chk_dividend, chk_side, chk_gap_open, chk_m6]


@pytest.mark.parametrize("chk", CHECKS, ids=lambda c: c.__name__)
def test_pravila_vyhoda_limitkoy(chk):
    chk()


def test_vyhod_lim_false_staryy_put_pobaytovo():
    a, b = go(FILL, p=pr(lim=False)), go(FILL, p=P(k=1.0, R=None, b=None, reentry=False))
    assert (a.x_orders, a.x_fills) == (0, 0) and a.trades[0].reason == "signal" and a.trades[0].exit_day == 1
    for f in ("gross", "cost", "pos", "free"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_zayavka_na_poslednem_dne_okna_ne_stavitsya():
    days = [D(FLAT), D(ENTRY)]
    r = exec_e.run("T", days, [1, 0], np.full(2, SG), pr())
    assert r.x_orders == 0 and r.trades[0].reason == "signal" and r.trades[0].exit_day == 1


def test_stroki_komissii_v_run_e1_lim_i_zapas():
    sd = {}
    for nm, d2 in (("lim", FILL), ("close", TOUCH)):
        days = [D(FLAT), D(ENTRY), D(d2), D(FLAT)]
        r = etap1_run.run_e1("T", days, np.array(SIG), np.full(4, SG), pr(), win=(0, 3))
        sd[nm] = (r.sides.tolist(), r.x_orders, r.x_fills)
    assert sd["lim"] == ([1.0, 1.0, 0.0], 1, 1) and sd["close"] == ([1.0, 0.0, 1.0], 1, 0)


def test_params_of_osi_vyhod_i_v11():
    pt = {**YAKOR, "vhod": 1.0, "proverka": "bary", "vyhod": "lim"}
    assert params_of(pt, 0.25, 5.0, False, C).vyhod_lim is True and not is_anchor(pt)
    assert params_of({**pt, "vyhod": "net"}, 0.25, 5.0, False, C).vyhod_lim is False
    assert params_of({k: v for k, v in pt.items() if k != "vyhod"}, 0.25, 5.0, False, C).vyhod_lim is False
    base = {"prichiny": {"stop": 0.0}, "v": "1.0"}
    assert v11(dict(base), None, {}, [], []) == base, "оба флага выключены: строка v1.0 как была"
    r = etap1_run.run_e1("T", [D(FLAT), D(ENTRY), D(FILL), D(FLAT)], np.array(SIG), np.full(4, SG), pr(), win=(0, 3))
    o = v11({"prichiny": {}, "v": "1.0"}, None, {"vyhod": "lim"}, [r], r.trades)
    assert (o["v"], o["fdr"], o["vyhod"], o["dolya_ispolneniya_vyhoda"], o["prichiny"]["lim"]) == ("1.1", None, "lim", 1.0, 1.0)
    assert v11({"prichiny": {}, "v": "1.0"}, "t0", {}, [], [])["v"] == "1.1"


# ---- мутации М-В1…М-В6 ----
def scan_x_first(s, i, j, d, b0):
    """Копия Run._scan: заявка выхода проверена раньше стопа (М-В5)."""
    sd = s.tr.side
    S, _ = s._levels(j)
    for b in range(b0, len(d.c)):
        o, h, l = d.o[b], d.h[b], d.l[b]
        if s.xo is not None and R_.limit_mask(-sd, h, l, s.xo["X"], s.p.tick):
            s.x_fills += 1
            return s._exit(i, j, b, R_.limit_px(-sd, o, s.xo["X"]), "lim")
        if S is not None:
            hit, px = R_.stop_hit(sd, o, h, l, S, s.p.stop_touch)
            if hit:
                return s._exit(i, j, b, px, s.sr)


def leave_same_day(s, i, j, nb, close, reason):
    """Мутация М-В2: заявка исполняется в день решения (по барам d1 до close)."""
    d = s.days[i]
    X = R_.limit_level(-s.tr.side, close, s.p.k, s.sigma[i], s.p.tick)
    hit = np.flatnonzero(R_.limit_mask(-s.tr.side, d.h, d.l, X, s.p.tick))
    if hit.size:
        return s._exit(i, j, int(hit[0]), X, "lim")
    s._exit(i, j, nb - 1, close, reason)


def leave_wrong_side(orig):
    def f(s, i, j, nb, close, reason):
        s.tr.side = -s.tr.side                      # уровень строится по стороне сделки, а не зеркальной
        try:
            orig(s, i, j, nb, close, reason)
        finally:
            if s.tr is not None:
                s.tr.side = -s.tr.side
    return f


def exit_no_dv(orig):
    def f(s, i, j, b, px, reason):
        s.dv = 0.0
        return orig(s, i, j, b, px, reason)
    return f


def check_noop(p):
    return None


MUTS = [
    ("М-В1_kasanie_ispolnyaet", lambda m: m.setattr(R_, "limit_mask", lambda sd, h, l, L, tick, o=R_.limit_mask: o(sd, h, l, L, 0.0)),
     chk_touch),
    ("М-В2_due_j", lambda m: m.setattr(R_.Run, "_leave", leave_same_day), chk_not_same_day),
    ("М-В3_ne_zerkalno", lambda m: m.setattr(R_.Run, "_leave", leave_wrong_side(R_.Run._leave)), chk_side),
    ("М-В4_dv_ne_primenen", lambda m: m.setattr(R_.Run, "_exit", exit_no_dv(R_.Run._exit)), chk_dividend),
    ("М-В5_X_ranshe_stopa", lambda m: m.setattr(R_.Run, "_scan", scan_x_first), chk_stop_first),
    ("М-В7_gep_po_urovnyu", lambda m: m.setattr(R_, "limit_px", lambda sd, o, L: L), chk_gap_open),
    ("М-В6_tihiy_otkat", lambda m: m.setattr(exec_e, "_check_params", check_noop), chk_m6),
]


@pytest.mark.parametrize("name,mut,chk", MUTS, ids=[m[0] for m in MUTS])
def test_mutatsiya_lovitsya(monkeypatch, name, mut, chk):
    mut(monkeypatch)
    with pytest.raises(AssertionError):
        chk()


def test_mutatsiya_sdvig_stopa_na_D_lovitsya(monkeypatch):
    monkeypatch.setattr(R_.Run, "_stop_shift", lambda s, S: S)
    with pytest.raises(AssertionError):
        chk_dividend()
