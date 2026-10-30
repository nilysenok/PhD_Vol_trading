"""Л1.3 (Ж-146), PR #843: мутации исполнителя fastbt/golaya_v7_isp.py и контроли с руками посчитанным ответом.

Мутант — копия исходника с одной подменой строки (в память, файлы не трогаются). Каждый обязан упасть на контролях
этого файла (`kontroli`); оригинал — пройти. Мутации: mutations.md № 5, 6 (комиссия выхода на конце окна), № 20 (знак),
№ 55 (вход в день сигнала), Д4 (дыра), Д7 (плата за шорт, дни года).
"""
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import golaya_v7_isp as G0

KOM, ZAEM = 0.0004, 0.05
DATES = pd.bdate_range("2016-01-04", periods=8).to_numpy()   # Пн 4.01 … Ср 13.01; Пт 8.01 → Пн 11.01 = 3 дня
O = 100.0 + np.arange(8)
C = O + 0.5
PT = O + 0.2


def sv(i, s=1, n=8):
    x = np.zeros(n, int)
    x[i] = s
    return x


def run(G, sig, o=O, c=C, rate=0.0, **kw):
    n = len(sig)
    r = np.full(n, rate) if np.isscalar(rate) else rate
    return G.ispolnit(np.asarray(sig, int), np.asarray(o, float), np.asarray(c, float), dates=DATES[:n], rate=r, **kw)


def flag(k, n=8):
    x = np.zeros(n, bool)
    x[k] = True
    return x


def k_komissiya_obeih_storon_shorta(G):
    r = run(G, sv(1, -1), tau=2)                               # вход open2=102, выход close3=103,5
    assert r["kom"][2] == pytest.approx(KOM) and r["kom"][3] == pytest.approx(KOM * 103.5 / 102)
    assert r["kom"].sum() == pytest.approx(KOM * (1 + 103.5 / 102))
    assert r["pnl"].sum() == pytest.approx(-1.5 / 102)


def k_konec_okna_dve_komissii(G):
    r = run(G, sv(5), tau=5)                                   # вход open6=106, окно кончилось: close7=107,5
    assert r["kom"].sum() == pytest.approx(KOM * (1 + 107.5 / 106)) and r["kom"][7] > 0
    r = run(G, sv(2), tau=None)                                # без τ: до конца окна
    assert r["kom"][7] == pytest.approx(KOM * 107.5 / 103)


def k_zaem_i_stavka_za_vyhodnye(G):
    r = run(G, sv(3, -1), tau=2, rate=0.10)                    # вход Пт 8.01 (open4=104), выход Пн close5=105,5
    assert r["zaem"][4] == pytest.approx(ZAEM * 3 / 365) and r["zaem"].sum() == pytest.approx(ZAEM * 3 / 365)
    assert r["stavka"][4] == pytest.approx(0.10 * 3 / 365)
    ожид = -1.5 / 104 - KOM * (1 + 105.5 / 104) - ZAEM * 3 / 365 - 0.10 * 3 / 365
    assert r["excess"].sum() == pytest.approx(ожид)
    assert run(G, sv(3, 1), tau=2, rate=0.10)["zaem"].sum() == 0


def k_znak(G):
    a, b = run(G, sv(1, 1), tau=2), run(G, sv(1, -1), tau=2)
    assert a["pnl"].sum() == pytest.approx(1.5 / 102) and b["pnl"].sum() == pytest.approx(-1.5 / 102)
    assert (b["poz"][2:4] == -1).all()


def k_vhod_open_zavtra(G):
    r = run(G, sv(1), tau=2)                                   # open2=102 → close3=103,5; не close1/open1
    assert r["pnl"].sum() == pytest.approx(1.5 / 102) and r["poz"][1] == 0


def k_dividendy(G):
    D = np.zeros(8)
    D[2], D[4] = 3.0, 2.0                                      # D на день входа не платится (гэп до open входа)
    assert run(G, sv(1), tau=3, D=D)["pnl"].sum() == pytest.approx((2.5 + 2.0) / 102)
    assert run(G, sv(1, -1), tau=3, D=D)["pnl"].sum() == pytest.approx(-(2.5 + 2.0) / 102)
    r = run(G, sv(1), exit_l=flag(3), D=D)                     # выход open4=104 после гэпа дня 4: D[4] учтён
    assert r["pnl"].sum() == pytest.approx((2.0 + 2.0) / 102)


def k_flag_shorta(G):
    r = run(G, sv(1, -1), exit_s=flag(3), exit_l=flag(2))      # exit_l шорта не трогает; exit_s: выход open4=104
    assert r["pnl"].sum() == pytest.approx(-2 / 102) and np.flatnonzero(r["poz"]).tolist() == [2, 3, 4]


def k_T_dyra_potom_sleduyushchiy_signal(G):
    pt = PT.copy()
    pt[1] = np.nan                                             # нет бара T — сделки нет; вход не по open/close
    r = run(G, np.array([0, 1, 1, 0, 0, 0, 0, 0]), vhod="T", p_T=pt, tau=1)
    assert r["propusk"] == 1 and len(r["sdelki"]) == 1 and r["poz"][1] == 0
    assert r["pnl"].sum() == pytest.approx(1.3 / 102.2)        # вход p_T2=102,2, выход close3=103,5


def k_T_posledniy_den_tau1(G):
    r = run(G, sv(7), vhod="T", p_T=PT, tau=1)                 # окно кончилось в день сигнала: close7=107,5
    assert r["pnl"].sum() == pytest.approx(0.3 / 107.2)
    assert r["kom"].sum() == pytest.approx(KOM * (1 + 107.5 / 107.2)) and r["poz"][7] == 1


def k_T_tau0_i_zaem(G):
    r = run(G, sv(2), vhod="T", p_T=PT, tau=0)                 # вход T и выход close в один день
    assert r["pnl"].sum() == pytest.approx(0.3 / 102.2) and r["zaem"].sum() == 0 and len(r["sdelki"]) == 1
    r = run(G, sv(3, -1), vhod="T", p_T=PT, tau=1)             # ночь Чт→Пт = 1 день
    assert r["zaem"][3] == pytest.approx(ZAEM / 365) and r["pnl"].sum() == pytest.approx(-1.3 / 103.2)


def k_T_bez_nakrytiya(G):
    r = run(G, np.array([0, 1, 1, 0, 0, 0, 0, 0]), vhod="T", p_T=PT, tau=1)   # сигнал 2 внутри сделки (выход close2)
    assert len(r["sdelki"]) == 1 and r["pnl"].sum() == pytest.approx(1.3 / 101.2)


def k_dyra_v_puti_otkaz(G):
    c = C.copy()
    c[3] = np.nan
    with pytest.raises(ValueError):
        run(G, sv(1), c=c, tau=3)


def k_tau_tochno(G):
    r = run(G, sv(0), tau=3)                                   # вход open1=101, выход close3=103,5 (не 2, не 4)
    assert r["pnl"].sum() == pytest.approx(2.5 / 101)


KONTROLI = [v for k, v in sorted(globals().items()) if k.startswith("k_")]


def kontroli(G):
    """Число упавших контролей."""
    n = 0
    for f in KONTROLI:
        try:
            f(G)
        except (AssertionError, ValueError, IndexError, TypeError, pytest.fail.Exception):
            n += 1
    return n


def mutant(staroe, novoe):
    src = Path(G0.__file__).read_text(encoding="utf-8")
    assert src.count(staroe) == 1, staroe
    m = types.ModuleType("mut")
    exec(compile(src.replace(staroe, novoe), "mut", "exec"), m.__dict__)  # noqa: S102
    return m


MUTANTY = {
    "kom_vhoda_snyata": ('out["kom"][e] += kom\n', 'out["kom"][e] += 0.0\n'),
    "zaem_znak_plyus_minus": ('out["zaem"][k] += zaem', 'out["zaem"][k] -= zaem'),
    "stavka_znak": ('out["stavka"][k] += rate', 'out["stavka"][k] -= rate'),
    "konec_okna_bez_komissii": ('out["kom"][xd] += kom * px / pe', 'out["kom"][xd] += kom * px / pe if xd < len(c) - 1 else 0.0'),
    "kom_vyhoda_ne_masshtab": ('out["kom"][xd] += kom * px / pe', 'out["kom"][xd] += kom'),
    "vhod_po_close_d": ('pe = float((o if vhod == "d1" else p_T)[e if vhod == "d1" else i])',
                        'pe = float((c if vhod == "d1" else p_T)[i])'),
    "gap_bez_vyhodnyh": ('gap = np.append(np.diff(np.asarray(dates, "datetime64[D]")).astype(float), 1.0)[:T]',
                         'gap = np.ones(T)'),
    "dividend_ignoriruetsya": ('(D[k] if k > e else 0.0)', '0.0'),
    "dividend_v_den_vhoda": ('(D[k] if k > e else 0.0)', 'D[k]'),
    "dividend_znak": ('(D[k] if k > e else 0.0)', '(-D[k] if k > e else 0.0)'),
    "znak_poteryan_short_kak_long": ('out["pnl"][k] += s * (', 'out["pnl"][k] += abs(s) * ('),
    "tau_plyus_odin": ('luchshiy = (i + tau, 1, None)', 'luchshiy = (min(i + tau + 1, T - 1), 1, None)'),
    "T_dyra_vhod_po_open": ('out["propusk"] += 1\n            i += 1\n            continue', 'pe = float(o[e])'),
    "dyra_v_puti_molcha": ('if not np.isfinite(v) or v <= 0:', 'if v <= 0:'),
    "T_nakrytie_sdelok": ('i = xd + 1 if (vhod == "T" and rang == 1) else xd', 'i = max(xd, i + 1)'),
    "flag_shorta_ot_exit_l": ('exit_l if s > 0 else exit_s', 'exit_l'),
    "flag_vyhod_v_den_flaga": ('xd = k + 1 if vhod == "d1" else k', 'xd = k'),
}


def test_original_prohodit_kontroli():
    assert kontroli(G0) == 0
    for f in KONTROLI:
        f(G0)


@pytest.mark.parametrize("imya", sorted(MUTANTY))
def test_mutatsiya_lovitsya(imya):
    assert kontroli(mutant(*MUTANTY[imya])) > 0, imya
