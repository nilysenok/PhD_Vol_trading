"""П-9 (проверяющий), листы Л-21 (`tejk_signal=rodnoy_uroven`) и Л-22 (`vhod_tip=stop_zayavka`). Пишется ДО кода, красные до Л-21/Л-22.
Источник: docs/resheniya/resheniya.md Ж-091 п. 1.1–1.4 (заявки к 18:50 d, исполнение внутри d+1).

ОТСТУПЛЕНИЕ ОТ ЗАДАНИЯ (находка): в коде S3 = Donchian (`dc_window`), S4 = Supertrend (`atr_period`, `multiplier`) — signals/s34.py
 S3_GRID/S4_GRID; текст Ж-091 п. 1.2 тоже S3 = канал, S4 = ST. Но fastbt/designs/progon/kub_S3.yaml несёт atr_period/multiplier, а kub_S4.yaml —
 dc_window: файлы куба перепутаны. Тесты следуют коду и Ж-091 (S3 = dc_window, S4 = atr_period+multiplier); проверка куба — только по оси.

ТОЛКОВАНИЕ [Ж-091 п. 1.1] (дирижёр, 28.10): при rodnoy_uroven выход по сигналу выключен, τ остаётся, t_vyhod схлопывается в open2 (не close; tau не net).
 test_canon_rodnoy_skhlopyvaet_t_vyhod перенацелен: было t_vyhod=close и tau=net. Значения куба для rodnoy_uroven — в test_p9_kuby.py.
ПОДГОНКА ПОД КОД (28.10, после PR #459 Л-21 и #462 Л-22; требования Ж-091 те же, меняются только сигнатуры):
 П1. Л-21: `rodnoy_uroven(sid, sp, dn, sd=1)` — kwarg `sd`, не `side`; в движок уровень идёт как `tejk_M` формы [2, days] (строка 0 лонг, 1 шорт), не `rod_L`;
     canon при rodnoy схлопывает ещё и tau (= net) — добавлено в test_canon_rodnoy_skhlopyvaet_t_vyhod.
 П2. Л-22: уровни — `stop_zayavki.urovni(sid, sp, td, ind, storona)` по ОДНОМУ тикеру (td: close/high/low/volume; ind: adx14, vol_sma20, vf_pass),
     ключи up/dn (вход buy-/sell-stop), xl/xs (выход лонга/шорта); фильтры сигнала и прогрев WARMUP=200 внутри (до прогрева nan).
     В движок — `run(..., urovni=dict(up, dn, xl, xs))`, вход OCO-парой, sig вход не задаёт; `Params(stop_touch=True, exit_signal=False)` обязательны,
     k_e в Params при stop_zayavka — ValueError. canon при stop_zayavka у S1/S2/S5/S6 — ValueError. Было А4/А5 ниже — заменено.
 Параллельность Л-21 и Л-22: ветки конфликтуют в exec_e.py, ledger_v2.py, etap1_eval_v2.py; тесты Л-21 гоняются на ветке Л-21, Л-22 — на ветке Л-22.

ДОПУЩЕНИЯ (в контракте сигнатур нет; поправить здесь, если Л-21/Л-22 назвали иначе):
 А1. `fastbt.tejk_signal.rodnoy_uroven(sid, sig_params, dannye_do_d, side=1) -> ndarray[ticker, d]`; dannye_do_d как у `uroven_M`
     (close/high/low/volume + dates). side=+1 лонг (S1: SMA − z_exit·STD, S6: VWAP − 0,5·dev), side=−1 шорт (знак +); S2/S5 от side не зависят.
     STD — `calc_std` (ddof=1), dev — из `s56_kern._vwap`. Столбец d использует только столбцы <= d.
 А2. Движок: `Params.tejk == "rodnoy_uroven"`; `run(..., rod_L=ndarray[len(days)])`: rod_L[d] — уровень, рассчитанный вечером d для стороны сделки;
     на день j действует rod_L[j-1] (обновление каждый вечер); сделка, открытая на баре b, снимает заявку с бара b+1.
     Лимит выхода — проход на тик (лонг high >= L + tick, шорт low <= L − tick), цена max/min(open, L). Сигнальный выход (exit_signal) при нём выключен.
 А3. `canon_v2`: при tejk_signal == rodnoy_uroven → t_vyhod = "close", R = k_x = net; при других значениях t_vyhod не трогается.
 А4. `fastbt.stop_zayavki.urovni(sid, sp, dn) -> dict` ключи buy, sell, exit_long, exit_short (ndarray[ticker, d], nan — нет уровня), только S3/S4.
     S3: buy = max(high, w), sell = min(low, w) по барам <= d (w = dc_window, окно включает d); S4: линия ST дня d: buy = линия там, где она выше close d,
     sell — где ниже. exit_long = sell, exit_short = buy (обратная граница / линия ST).
 А5. Движок: `Params.vhod_tip` (podtverzhdenie | stop_zayavka); `run(..., stop_L={"vhod": arr, "vyhod": arr})`, arr[d] — вечер d для стороны сделки
     (vhod — стоп входа, vyhod — стоп выхода). Заявка живёт день d+1; на день j действует arr[j-1]. Вход: лонг high >= L → цена max(open, L);
     шорт low <= L → min(open, L) (касание, без тика). Выход лонга: low <= L → min(open, L); шорта: high >= L → max(open, L). Заявка выхода не активна
     в баре входа. При stop_zayavka лимит k_e (`k`) не применяется: результат с k = 0,5 равен результату с k = None (или ValueError).
 А6. `canon_v2`: при vhod_tip == stop_zayavka → vhod = "rynok" и далее как для rynok (ed_lim=sigma, zhizn=den, lestnica=net, gap=vhod).
 А7. Куб: tejk_signal.setka = [net, k_srednemu, pol_puti, rodnoy_uroven] (S1, S2, S5, S6); vhod_tip.setka = [podtverzhdenie, stop_zayavka],
     yakor podtverzhdenie (S3, S4); vhod_tip нет в S1/S2/S5/S6, rodnoy_uroven нет в S3/S4. test_p6::test_os_v_kube (setka == 3 значения) покраснеет — это ожидаемо.
Мутации владельца 1–6 и свои 7–11:
 1 уровень по d+1 -> test_rod_d_plus_1_i_obnovlenie, test_stop_vhod_d_plus_1_i_obnovlenie, test_stop_vyhod_gep_i_obnovlenie, test_urovni_*_ne_smotryat_vpered;
 2 гэп по уровню -> test_rod_gep_*, test_stop_vhod_gep_*, test_stop_vyhod_gep_i_obnovlenie, test_stop_short;
 3 активна в баре исполнения -> test_rod_ne_v_bare_vkhoda, test_stop_vyhod_ne_v_bare_vkhoda;
 4 не обновлён вечером -> те же *_obnovlenie, test_rodnoy_uroven_kazhdyy_vecher;
 5 rodnoy при t_vyhod -> test_canon_rodnoy_skhlopyvaet_t_vyhod, test_klyuch_rodnoy_t_vyhod_odin, test_rod_signal_vyhod_vyklyuchen;
 6 stop_zayavka + k_e -> test_canon_stop_skhlopyvaet_vhod, test_klyuch_stop_vhod_odin, test_stop_k_e_ne_primenyaetsya;
 7 проход на тик у стоп-заявки -> test_stop_vhod_kasanie, test_stop_vyhod_kasanie; 8 rodnoy по касанию -> test_rod_prohod_na_tik;
 9 vhod_tip в S1/S2/S5/S6 -> test_kub_vhod_tip_tolko_S3_S4; 10 rodnoy в S3/S4 -> test_kub_rodnoy_tolko_S1_S2_S5_S6;
 11 podtverzhdenie не побайтно -> test_podtverzhdenie_pobaytno, test_canon_podtverzhdenie_ne_trogaet, test_dannye_podtverzhdenie_pobaytno.
Данные <= 2019: FASTBT_BARS1M=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D25/bars1m
Запуск: python3 -m pytest tests/proverka_progon/test_p9_zayavki.py -q --tb=short"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastbt import bars5m, cache
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

_MAIN_BARS = cache.MAIN / "data" / "regen" / "D25" / "bars1m"
if not bars5m.SRC.exists() and _MAIN_BARS.exists():
    bars5m.SRC = _MAIN_BARS

TK = 0.01
KUB = Path("fastbt/designs/progon")
OSI4 = ["net", "k_srednemu", "pol_puti", "rodnoy_uroven"]
TIPY = ["podtverzhdenie", "stop_zayavka"]
ROD = "rodnoy_uroven"
STOP = "stop_zayavka"


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def F(x=100.0, n=1):
    return [(x, x, x, x)] * n


def tr0(r):
    return r.trades[0]


def go_rod(days, L, side=1, sig=None, tejk=ROD, **kw):
    """Рынок на close d (k = None), σ̂ = 0,02; rod_L[d] — вечер d."""
    n = len(days)
    kw.setdefault("comm", 0.0)
    p = Params(side=side, tick=TK, check="bary", tejk=tejk, **kw)
    s = np.ones(n) if sig is None else np.asarray(sig, float)
    row = np.broadcast_to(np.asarray(L, float), (n,))
    return run("T", days, s, np.full(n, 0.02), p, tejk_M=np.vstack([row, row]))                 # строка 0 — лонг, 1 — шорт (П1)


def go_stop(days, vhod, vyhod, side=1, **kw):
    """Вход OCO-парой на день d+1 (П2): лонг — up = vhod, шорт — dn = vhod; выход — xl/xs = vyhod."""
    n = len(days)
    kw.setdefault("comm", 0.0)
    p = Params(side=side, tick=TK, check="bary", vhod_tip=STOP, stop_touch=True, exit_signal=False, **kw)
    a, nan = np.broadcast_to(np.asarray(vhod, float), (n,)).copy(), np.full(n, np.nan)
    x = np.broadcast_to(np.asarray(vyhod, float), (n,)).copy()
    u = dict(up=a if side == 1 else nan, dn=a if side == -1 else nan, xl=x, xs=x)
    return run("T", days, np.ones(n), np.full(n, 0.02), p, urovni=u)


# ===== Л-21: rodnoy_uroven, исполнение (лимит выхода на уровне; вход рынком на close d = 100) =====
def test_rod_prohod_na_tik():                                                 # мутация 8: по касанию — выход на high == L
    assert tr0(go_rod([D(F()), D([(100, 104.0, 100, 100)]), D(F())], 104.0)).reason is None
    t = tr0(go_rod([D(F()), D([(100, 104.01, 100, 100)]), D(F())], 104.0))
    assert t.exit_day == 1 and t.exit_px == pytest.approx(104.0)


def test_rod_d_plus_1_i_obnovlenie():                                         # мутации 1 и 4: на день j действует L[j-1], обновляется каждый вечер
    days = [D(F()), D([(100, 103, 100, 100)]), D([(100, 102, 100, 100)]), D(F())]
    t = tr0(go_rod(days, [105, 101, 200, 200]))
    assert t.exit_day == 2 and t.exit_px == pytest.approx(101.0)             # день 1: L0 = 105 не достигнут; день 2: L1 = 101. L[d+1] дал бы выход в день 1; устаревший L0 — вовсе без выхода


def test_rod_gep_long():                                                      # мутация 2
    t = tr0(go_rod([D(F()), D([(103, 104, 102.5, 103)]), D(F())], 101.0))
    assert t.exit_day == 1 and t.exit_px == pytest.approx(103.0)


def test_rod_gep_short():                                                     # мутация 2 (зеркало)
    t = tr0(go_rod([D(F()), D([(94, 95, 93, 94)]), D(F())], 96.0, side=-1))
    assert t.side == -1 and t.exit_px == pytest.approx(94.0)
    assert tr0(go_rod([D(F()), D([(97, 97, 96.0, 97)]), D(F())], 96.0, side=-1)).reason is None              # касание шорта — нет выхода
    assert tr0(go_rod([D(F()), D([(97, 97, 95.98, 97)]), D(F())], 96.0, side=-1)).exit_px == pytest.approx(96.0)


def test_rod_ne_v_bare_vkhoda():                                              # мутация 3: лимитный вход 99 в баре 0, high 102 > L = 101
    days = [D(F()), D([(100, 102, 98.9, 100), (100, 101.5, 100, 100)]), D(F())]
    t = tr0(go_rod(days, 101.0, k=0.5, f="propusk"))
    assert t.entry_px == pytest.approx(99.0) and t.entry_bar == 0
    assert t.exit_day == 1 and t.exit_bar == 1 and t.exit_px == pytest.approx(101.0)


def test_rod_stop_pervym():
    t = tr0(go_rod([D(F()), D([(100, 105, 97, 100)]), D(F())], 104.0, b=1.0, stop_touch=True))              # S = 98, L = 104 в одном баре
    assert t.reason == "stop" and t.exit_px == pytest.approx(98.0)


def test_rod_signal_vyhod_vyklyuchen():                                       # мутация 5 (движок): сигнал погас — позиция держится до уровня
    days, sig = [D(F()) for _ in range(5)], [1, 1, 0, 0, 0]
    assert tr0(go_rod(days, 200.0, sig=sig, tejk="net")).reason == "signal"   # контроль: без оси выход по сигналу есть
    assert tr0(go_rod(days, 200.0, sig=sig)).reason is None


# ===== Л-21: уровень rodnoy_uroven(sid, sp, dn, side) =====
def _dannye(n=60, nt=2, seed=3):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 1, (nt, n)), axis=1)
    return {"close": c, "high": c + r.uniform(0.1, 1, (nt, n)), "low": c - r.uniform(0.1, 1, (nt, n)),
            "volume": r.uniform(1e3, 2e3, (nt, n)), "dates": np.datetime64("2018-01-01") + np.arange(n) * np.timedelta64(1, "D")}


def _poison(dn, d):
    out = {k: v.copy() for k, v in dn.items()}
    for k in ("close", "high", "low", "volume"):
        out[k][:, d + 1:] *= 7.0
    return out


def _cut(dn, d):
    return {k: v[..., :d + 1] for k, v in dn.items()}


SIG = {"S1": {"ma_window": 5, "z_entry": 2.0, "z_exit": 0.5}, "S2": {"bb_window": 5, "bb_std": 2.0},
       "S5": {"pivot_period": "weekly", "pivot_type": "classic", "filter_set": 3}, "S6": {"vwap_window": 5, "dev_mult": 1.5, "filter_set": 2, "dir_bias": "OFF"}}
S3P, S4P = {"dc_window": 5}, {"atr_period": 5, "multiplier": 2.0}


def _rod(sid, dn, side=1):
    from fastbt.tejk_signal import rodnoy_uroven
    return rodnoy_uroven(sid, SIG[sid], dn, sd=side)


@pytest.mark.parametrize("side", [1, -1])
def test_rodnoy_uroven_S1_znachenie(side):
    from fastbt.indicators import calc_sma, calc_std
    dn = _dannye()
    L = _rod("S1", dn, side)
    for i in range(2):
        c = dn["close"][i]
        np.testing.assert_allclose(L[i, 4:], (calc_sma(c, 5) - side * 0.5 * calc_std(c, 5))[4:])   # лонг: SMA − z_exit·STD, шорт: SMA + z_exit·STD
    assert np.isnan(L[:, :4]).all()


def test_rodnoy_uroven_S2_S5_ravny_M():                                       # S2: SMA; S5: P — те же ряды, что у тейка (side не влияет)
    from fastbt.tejk_signal import uroven_M
    dn = _dannye()
    for sid in ("S2", "S5"):
        for side in (1, -1):
            np.testing.assert_array_equal(_rod(sid, dn, side), uroven_M(sid, SIG[sid], dn))


@pytest.mark.parametrize("side", [1, -1])
def test_rodnoy_uroven_S6_znachenie(side):
    from fastbt.signals.s56_kern import _vwap
    dn = _dannye()
    L = _rod("S6", dn, side)
    for i in range(2):
        vw, dev = _vwap(*(np.asarray(dn[k][i], float) for k in ("close", "high", "low", "volume")), 5)
        ok = ~np.isnan(vw)
        np.testing.assert_allclose(L[i][ok], (vw - side * 0.5 * dev)[ok])        # лонг: VWAP − 0,5·dev
        assert (np.isnan(L[i]) == np.isnan(vw)).all()


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
@pytest.mark.parametrize("side", [1, -1])
def test_rodnoy_uroven_ne_smotrit_vpered(sid, side):                          # мутация 1: данные d+1
    dn, d = _dannye(), 30
    a, b = _rod(sid, dn, side), _rod(sid, _poison(dn, d), side)
    np.testing.assert_array_equal(a[:, :d + 1], b[:, :d + 1])
    np.testing.assert_array_equal(a[:, d], _rod(sid, _cut(dn, d), side)[:, d])


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_rodnoy_uroven_kazhdyy_vecher(sid):                                   # мутация 4: уровень не обновлён — каждый вечер новый
    dn = _dannye()
    L = _rod(sid, dn)
    for d in (20, 21, 22, 40):
        np.testing.assert_array_equal(L[:, d], _rod(sid, _cut(dn, d))[:, d])
    assert len(np.unique(L[0][~np.isnan(L[0])])) > 5


def test_rodnoy_uroven_tolko_S1_S2_S5_S6():
    from fastbt.tejk_signal import rodnoy_uroven
    with pytest.raises(ValueError):
        rodnoy_uroven("S3", S3P, _dannye())


# ===== Л-21: схлопывание (мутация 5) =====
def _canon(sid="S1", **kw):
    from fastbt.ledger_v2 import canon_v2
    return canon_v2(sid, {"k_s": "net", "R": "net", "k_x": "net", "k_tr": "net", "bezubytok": "net", "vhod": 0.5, "t_vyhod": "close", **kw})


def test_canon_rodnoy_skhlopyvaet_t_vyhod():
    for tv in ("1500", "1700", "open2"):
        q = _canon(tejk_signal=ROD, t_vyhod=tv, R=3.0, k_x=1.5, tau=5)
        assert q["t_vyhod"] == "open2" and q["R"] == "net" and q["k_x"] == "net" and q["tau"] == 5     # τ остаётся (толкование [Ж-091 п. 1.1])
    for t in ("net", "k_srednemu", "pol_puti"):                               # остальные значения t_vyhod не трогают
        assert _canon(tejk_signal=t, t_vyhod="1500")["t_vyhod"] == "1500"


def test_klyuch_rodnoy_t_vyhod_odin():
    from fastbt.ledger_v2 import klyuch, klyuch_sha
    base = {"k_s": 2.0, "k_x": "net", "vhod": 0.5, "tejk_signal": ROD}
    s = lambda tv: klyuch_sha(klyuch("S1", {**base, "t_vyhod": tv}, "train4", 0.0004, None), "k", "v")   # noqa: E731
    assert s("1500") == s("open2") == s("close")                              # неразличимые точки — один ключ (в N один раз)
    base["tejk_signal"] = "net"
    assert s("1500") != s("close")


# ===== Л-22: stop_zayavka, исполнение. Сигнал на всех днях, лонг; день 0 — плоский 100, вход на d+1 =====
def test_stop_vhod_kasanie():                                                 # мутация 7: касание high == L — вход по L; проход на тик не нужен
    t = tr0(go_stop([D(F()), D([(100, 104.0, 99, 100)]), D(F())], 104.0, 80.0))
    assert t.entry_day == 1 and t.entry_bar == 0 and t.entry_px == pytest.approx(104.0)
    assert not go_stop([D(F()), D([(100, 103.99, 99, 100)]), D(F())], 104.0, 80.0).trades[:1]


def test_stop_vhod_gep_long():                                                # мутация 2: open 106 выше buy-stop 104 — вход по open
    t = tr0(go_stop([D(F()), D([(106, 107, 105, 106)]), D(F())], 104.0, 80.0))
    assert t.entry_px == pytest.approx(106.0)


def test_stop_vhod_d_plus_1_i_obnovlenie():                                   # мутации 1 и 4
    days = [D(F()), D([(100, 105, 100, 100)]), D([(100, 105, 100, 100)]), D(F())]
    t = tr0(go_stop(days, [110, 104, 110, 110], 80.0))
    assert t.entry_day == 2 and t.entry_px == pytest.approx(104.0)           # день 1: L0 = 110 не достигнут; день 2: L1 = 104. L[d+1] дал бы вход в день 1; устаревший L0 — без входа
    t = tr0(go_stop(days, [104, 110, 110, 110], 80.0))
    assert t.entry_day == 1 and t.entry_px == pytest.approx(104.0)


def test_stop_vyhod_ne_v_bare_vkhoda():                                       # мутация 3: вход в баре 0 (high 105), low 90 < выхода 95 — выход не в баре входа
    days = [D(F()), D([(100, 105, 90, 100), (100, 101, 94, 95)]), D(F())]
    t = tr0(go_stop(days, 104.0, 95.0))
    assert t.entry_bar == 0 and t.entry_px == pytest.approx(104.0)
    assert t.exit_day == 1 and t.exit_bar == 1 and t.exit_px == pytest.approx(95.0)


def test_stop_vyhod_kasanie():                                                # мутация 7
    d1 = D([(100, 105, 100, 100)])
    t = tr0(go_stop([D(F()), d1, D([(100, 101, 95.0, 100)]), D(F())], 104.0, 95.0))
    assert t.exit_day == 2 and t.exit_px == pytest.approx(95.0)
    assert tr0(go_stop([D(F()), d1, D([(100, 101, 95.01, 100)]), D(F())], 104.0, 95.0)).reason is None


def test_stop_vyhod_gep_i_obnovlenie():                                       # мутации 1, 2, 4: выход на день 2 — по vyhod[1] = 95, open 90 — цена 90
    days = [D(F()), D([(100, 105, 100, 104)]), D([(90, 91, 89, 90)]), D(F())]
    t = tr0(go_stop(days, 104.0, [80, 95, 70, 70]))
    assert t.exit_day == 2 and t.exit_px == pytest.approx(90.0)


def test_stop_short():                                                        # зеркало: sell-stop 96 (open 100 → цена 96), buy-stop выхода 110, гэп 115
    days = [D(F()), D([(100, 100, 95, 96)]), D([(115, 116, 114, 115)]), D(F())]
    t = tr0(go_stop(days, 96.0, [120, 110, 110, 110], side=-1))
    assert t.side == -1 and t.entry_px == pytest.approx(96.0) and t.exit_day == 2 and t.exit_px == pytest.approx(115.0)
    t = tr0(go_stop([D(F()), D([(94, 95, 93, 94)]), D(F())], 96.0, 130.0, side=-1))      # гэп вниз через sell-stop: вход по open
    assert t.entry_px == pytest.approx(94.0)


def test_stop_k_e_ne_primenyaetsya():                                         # мутация 6 (движок): лимит k_e с stop_zayavka не действует
    days = [D(F()), D([(100, 105, 98, 100), (100, 101, 94, 95)]), D(F())]
    a = go_stop(days, 104.0, 95.0)
    try:
        b = go_stop(days, 104.0, 95.0, k=0.5, f="propusk")
    except ValueError:
        return
    assert a.trades == b.trades and all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("gross", "cost", "pos", "free"))


# ===== Л-22: уровни stop_zayavki.urovni =====
N22, W22 = 260, 200                                                           # WARMUP = 200: до него заявок нет (П2)


def _td(n=N22):
    from types import SimpleNamespace
    r = np.random.default_rng(5)
    c = 100 + np.cumsum(r.normal(0, 1, n))
    v = r.uniform(1e3, 2e3, n)
    td = SimpleNamespace(close=c, high=c + r.uniform(0.1, 1, n), low=c - r.uniform(0.1, 1, n), volume=v)
    return td, dict(adx14=np.full(n, 30.0), vol_sma20=v * 0.5, vf_pass=np.ones(n, bool))     # фильтры сигнала пройдены везде


def _cut22(td, ind, d):
    from types import SimpleNamespace
    return (SimpleNamespace(**{k: getattr(td, k)[:d + 1] for k in ("close", "high", "low", "volume")}), {k: v[:d + 1] for k, v in ind.items()})


def _poison22(td, ind, d):
    from types import SimpleNamespace
    q = {k: getattr(td, k).copy() for k in ("close", "high", "low", "volume")}
    for a in q.values():
        a[d + 1:] *= 7.0
    return SimpleNamespace(**q), ind


def _urovni(sid, td, ind):
    from fastbt.stop_zayavki import urovni
    return urovni(sid, S3P if sid == "S3" else S4P, td, ind)


def test_urovni_S3_znachenie():
    td, ind = _td()
    u = _urovni("S3", td, ind)
    for d in (W22, 230, 259):
        assert u["up"][d] == pytest.approx(td.high[d - 4:d + 1].max())            # окно включает d
        assert u["dn"][d] == pytest.approx(td.low[d - 4:d + 1].min())
        assert u["xl"][d] == pytest.approx(u["dn"][d]) and u["xs"][d] == pytest.approx(u["up"][d])    # выход — обратная граница
    assert np.isnan(u["up"][:W22]).all() and np.isnan(u["dn"][:W22]).all()


def test_urovni_S4_liniya_ST():
    from fastbt.signals.s34 import supertrend
    td, ind = _td()
    u = _urovni("S4", td, ind)
    st = supertrend(td.high, td.low, td.close, 5, 2.0)
    over, under = (st > td.close)[W22:], (st < td.close)[W22:]
    np.testing.assert_allclose(u["up"][W22:][over], st[W22:][over])               # ST выше close — buy-stop
    np.testing.assert_allclose(u["dn"][W22:][under], st[W22:][under])             # ниже — sell-stop
    assert np.isnan(u["up"][W22:][~over]).all() and np.isnan(u["dn"][W22:][~under]).all()
    np.testing.assert_allclose(u["xl"][W22:], st[W22:])
    np.testing.assert_allclose(u["xs"][W22:], st[W22:])


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_urovni_ne_smotryat_vpered(sid):                                      # мутация 1
    td, ind = _td()
    d = 230
    a, b = _urovni(sid, td, ind), _urovni(sid, *_poison22(td, ind, d))
    c = _urovni(sid, *_cut22(td, ind, d))
    for k in a:
        np.testing.assert_array_equal(a[k][:d + 1], b[k][:d + 1])
        np.testing.assert_array_equal(a[k][d], c[k][d])


def test_urovni_kazhdyy_vecher():                                             # мутация 4
    td, ind = _td()
    u = _urovni("S3", td, ind)
    for d in (210, 211, 240):
        assert u["up"][d] == _urovni("S3", *_cut22(td, ind, d))["up"][d]
    assert len(np.unique(u["up"][W22:])) > 3


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_urovni_tolko_S3_S4(sid):
    from fastbt.stop_zayavki import urovni
    with pytest.raises(ValueError):
        urovni(sid, SIG[sid], *_td())


# ===== Л-22: схлопывание (мутация 6) и путь podtverzhdenie (мутация 11) =====
def _canon3(sid="S3", **kw):
    from fastbt.ledger_v2 import canon_v2
    return canon_v2(sid, {"k_s": 2.0, "R": 2.0, "k_x": "net", "vhod": 0.5, "ed_lim": "atr14", "zhizn": 3, "lestnica": "dva", "gap": "propusk", **kw})


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_canon_stop_skhlopyvaet_vhod(sid):
    q = _canon3(sid, vhod_tip=STOP)
    assert (q["vhod"], q["ed_lim"], q["zhizn"], q["lestnica"], q["gap"]) == ("rynok", "sigma", "den", "net", "vhod")


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_canon_stop_tolko_S3_S4(sid):                                         # мутация 9: vhod_tip у S1/S2/S5/S6
    from fastbt.ledger_v2 import canon_v2
    with pytest.raises(ValueError):
        canon_v2(sid, {"vhod": 0.5, "vhod_tip": STOP})


def test_klyuch_stop_vhod_odin():
    from fastbt.ledger_v2 import klyuch, klyuch_sha
    base = {"k_s": 2.0, "k_x": "net", "R": 2.0, "vhod_tip": STOP}
    s = lambda v: klyuch_sha(klyuch("S3", {**base, "vhod": v}, "train4", 0.0004, None), "k", "v")   # noqa: E731
    assert s(0.25) == s(0.75) == s("rynok")
    base["vhod_tip"] = "podtverzhdenie"
    assert s(0.25) != s(0.75)


def test_canon_podtverzhdenie_ne_trogaet():                                   # мутация 11
    q = _canon3(vhod_tip="podtverzhdenie")
    assert (q["vhod"], q["ed_lim"], q["zhizn"], q["lestnica"], q["gap"]) == (0.5, "atr14", 3, "dva", "propusk")
    assert {k: v for k, v in q.items() if k != "vhod_tip"} == {k: v for k, v in _canon3().items()}


def test_podtverzhdenie_pobaytno():                                           # мутация 11: Params(vhod_tip=podtverzhdenie) ≡ Params() побайтно
    days = [D(F()), D([(100, 105, 98.9, 100), (100, 101.5, 100, 100)]), D([(100, 110, 100, 100)]), D(F())]
    kw = dict(side=1, tick=TK, check="bary", comm=0.0004, k=0.5, f="propusk", a=1.0, b=1.0)
    a = run("T", days, np.ones(4), np.full(4, 0.02), Params(**kw))
    b = run("T", days, np.ones(4), np.full(4, 0.02), Params(vhod_tip="podtverzhdenie", **kw))
    assert a.trades == b.trades
    for f in ("gross", "cost", "pos", "free"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


# ===== куб (мутации 9, 10) =====
def _find(o, key):
    if isinstance(o, dict):
        return [o[key]] if key in o else [x for v in o.values() for x in _find(v, key)]
    return [x for v in o for x in _find(v, key)] if isinstance(o, list) else []


def _kub(sid):
    return yaml.safe_load((KUB / f"kub_{sid}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_kub_vhod_tip_tolko_S3_S4(sid):                                       # мутация 9
    assert "vhod_tip" not in (KUB / f"kub_{sid}.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_kub_vhod_tip_v_osi(sid):
    f = _find(_kub(sid), "vhod_tip")
    assert len(f) == 1 and list(f[0]["setka"]) == TIPY and f[0]["yakor"] == "podtverzhdenie"


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_kub_rodnoy_tolko_S1_S2_S5_S6(sid):                                   # мутация 10
    txt = (KUB / f"kub_{sid}.yaml").read_text(encoding="utf-8")
    assert "rodnoy_uroven" not in txt and "tejk_signal" not in txt


# ===== данные <= 2019 (train4, одна точка на ось) =====
@pytest.fixture(scope="module")
def ev_s1():
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    return EvaluatorV2("S1", etap1.build_inputs("S1", period="train4"), etap1.base_point("S1"), 0.0004)


@pytest.fixture(scope="module")
def ev_s3():
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    return EvaluatorV2("S3", etap1.build_inputs("S3", period="train4"), etap1.base_point("S3"), 0.0004)


def _x(ev, **pt):
    r = ev.evaluate_v2(pt, "train4")
    assert r[0]["status"] == "ok"
    return r[1]


def test_dannye_rodnoy_menyaet_ryad(ev_s1):
    kw = dict(vhod=0.5, proverka="bars5m")
    assert not np.array_equal(_x(ev_s1, **kw), _x(ev_s1, tejk_signal=ROD, **kw))


def test_dannye_stop_menyaet_ryad(ev_s3):
    assert not np.array_equal(_x(ev_s3, vhod_tip="podtverzhdenie"), _x(ev_s3, vhod_tip=STOP))


def test_dannye_podtverzhdenie_pobaytno(ev_s3):                               # мутация 11 на ленте
    assert np.array_equal(_x(ev_s3), _x(ev_s3, vhod_tip="podtverzhdenie"))
