"""Т-677: добор тестов к V6-10 (замечания Н1–Н3 проверяющего #677). Известные ответы, код не меняется.
Н1: вход лимитом по 99 при close дня сигнала 100 — нижняя граница 0,001·close_d = 0,1, расстояние VWAP от close_d.
Н2: VWAP/dev берутся строкой d_sig, а не d_sig+1, на ряду с разными значениями и запасом строк (без IndexError).
Н3: тейк и стоп в одном баре для шорта; округление тейка по tick."""
from fastbt import exec_e_piv
from fastbt.exec_e import Params
from fastbt.exec_e_piv import enter_levels
from test_exec_e_piv import FLAT, NAN, bar, ex, go, lv


def lim(lo, hi=100.0, c=99.2):
    return bar(lo, hi, c=c)       # бар входа: лимит 99 коснулся (low < 99), стоп 98,9 не задет


# --- Н1: вход не по close дня сигнала ---
def test_n1_nizhnyaya_granitsa_ot_close_dnya_signala():
    nan = lv(4, P=NAN, S1=NAN, S2=NAN, R1=NAN, R2=NAN)
    r = go([FLAT, lim(98.95), bar(98.9005), bar(98.85)], piv_sl="next", levels=nan, k=1.0, f="propusk")
    assert r.trades[0].entry_px == 99.0
    assert ex(r) == (3, 98.9, "stop")      # граница 0,001·100 = 0,1 → 98,9; от цены входа было бы 98,901 (выход днём раньше)


def test_n1_vwap_rasstoyanie_ot_close_dnya_signala():
    lev = lv(4, vwap=104.0, dev=1.0)       # полоса 99; от close_d = 100 расстояние 1 → стоп 98; от входа 99 было бы 0 → граница
    r = go([FLAT, lim(98.95), bar(98.5), bar(97.9)], piv_sl="vwap", vwap_m=5.0, levels=lev, k=1.0, f="propusk")
    assert r.trades[0].entry_px == 99.0 and ex(r) == (3, 98.0, "stop")


def test_n1_enter_levels_close_otdelno_ot_ceny_vhoda():
    p = Params(piv_sl="vwap", vwap_m=5.0)
    S, T = enter_levels(p, lv(1, vwap=104.0, dev=1.0), 1, 99.0, 0, 100.0, None)
    assert (S, T) == (98.0, None)


# --- Н2: строка d_sig, не d_sig+1 ---
def test_n2_vwap_dev_strokoj_d_sig():
    lev = lv(4, vwap=[104.0, 90.0, 90.0, 90.0], dev=[1.0, 5.0, 5.0, 5.0])      # строка 1: полоса 65 → стоп 65 (не 99)
    assert ex(go([FLAT, bar(98.9), FLAT, FLAT], piv_sl="vwap", vwap_m=5.0, levels=lev)) == (1, 99.0, "stop")


def test_n2_ryad_razlichaet_stroki():
    lev = lv(4, vwap=[104.0, 90.0, 90.0, 90.0], dev=[1.0, 5.0, 5.0, 5.0])
    p = Params(piv_sl="vwap", vwap_m=5.0)
    assert exec_e_piv.stop_dist(p, lev, 1, 0, 100.0) == 1.0
    assert exec_e_piv.stop_dist(p, lev, 1, 1, 100.0) == 35.0      # мутант d+1 дал бы 35 без исключения (строк с запасом)


# --- Н3: шорт, один бар, tick ---
def test_n3_short_stop_ranshe_tejka_v_odnom_bare():
    r = go([FLAT, bar(95.9, 103.1)], side=-1, piv_sl="next", piv_tp="R")     # стоп 103, тейк 96
    assert ex(r) == (1, 103.0, "stop")


def test_n3_round_tejka_po_tick():
    for sd, tp, want, tk in ((1, "R", 104.3, 0.7), (-1, "R", 95.9, 0.7), (1, "half", 102.4, 0.8), (-1, "half", 97.6, 0.8)):
        _, T = enter_levels(Params(piv_tp=tp), lv(1), sd, 100.0, 0, 100.0, tk)
        assert abs(T - want) < 1e-9, (sd, tp, T)     # лонг — вверх к сетке, шорт — вниз


def test_n3_tejk_shorta_po_tick_v_run():
    r = go([FLAT, bar(95.0, 100.0)], side=-1, piv_tp="R", tick=0.7)
    assert ex(r)[0] == 1 and ex(r)[2] == "take" and abs(r.trades[0].exit_px - 95.9) < 1e-9
