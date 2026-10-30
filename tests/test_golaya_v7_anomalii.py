"""Расследование аномалий прогона Р1 голых идей S7–S12 (#866): воспроизведение на синтетике, без починки.

xfail(strict=True) — дефект воспроизведён: тест обязан начать проходить после починки (тогда снять xfail).
Обычные тесты — проверенная и отброшенная гипотеза (поведение верное, закрепляется).
А — корпсобытие (консолидация IRAO ×100, data/corporate_actions.csv) не применяется к ценам драйвера голых идей.
Б — mSR (Корниш–Фишер) вне области применимости: один день-выброс даёт |mSR| в десятки (к2 S12 train4: −51,8).
В — плацебо сечения S7–S9 переставляет тикеры каждый день: оборот растёт в разы, плацебо платит комиссию, которой нет у центра.
В' — к2 сечения (np.roll) заносит строки прогрева без скора (NaN) в окно: доля времени в позиции у к2 ниже, чем у центра.
Г — вторая проверка сдвига S8 через приращения скора: максимум на последнем известном дне d−21, подложенная утечка видна на d+1.
Д — к2 S10–S12 сдвигает сигнал относительно неподвижных цен (привязка к времени рвётся), лаг 0 = центр.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7 as gv  # noqa: E402
from fastbt import data as fd, kross_seksiya as ks, kross_v7 as kv, metriki_e as me, progon_kontrol as pk  # noqa: E402
from test_golaya_v7 import TK, bary  # noqa: E402

FAKTOR = 100.0


def _s_konsolidaciej(tmp_path, monkeypatch):
    """Тикер TK[0]: до даты события цены в FAKTOR раз меньше (как IRAO до 20.01.2015), событие — в corporate_actions (tmp)."""
    raw, im = bary()
    t, b = TK[0], raw[TK[0]].copy()
    days = b["timestamp"].dt.normalize().unique()
    data = pd.Timestamp(days[300])
    m = b["timestamp"] < data
    for c in ("open", "high", "low", "close"):
        b.loc[m, c] = b.loc[m, c] / FAKTOR
    raw[t] = b
    p = tmp_path / "corporate_actions.csv"
    pd.DataFrame([dict(ticker=t, date=data.date(), factor=FAKTOR, kind="консолидация (синтетика)", status="applied", source="тест")]
                 ).to_csv(p, index=False)
    monkeypatch.setattr(fd, "CORP", p)
    assert len(fd.corp_actions(t)) == 1
    return raw, im, t, data


def test_a_konsolidaciya_primenyaetsya_k_cenam_drajvera(tmp_path, monkeypatch):
    raw, im, t, data = _s_konsolidaciej(tmp_path, monkeypatch)
    D = gv.iz_barov(raw, im, "train4", TK)
    i = int(np.flatnonzero(D["daty"] == data)[0])
    for k in ("o", "cr", "c"):                               # исполнение (open 10:00, close 18:50) и сигнальный ряд
        x = D[k][t].to_numpy()
        assert abs(np.log(x[i] / x[i - 1])) < 0.2, f"{k}: скачок ×{x[i] / x[i - 1]:.0f} в день консолидации"


def test_a_dlinnaya_sdelka_cherez_konsolidaciyu_ne_vidit_skachka(tmp_path, monkeypatch):
    """После починки А: лонг через день события не получает ≈ +FAKTOR−1 на единицу цены входа."""
    raw, im, t, data = _s_konsolidaciej(tmp_path, monkeypatch)
    D = gv.iz_barov(raw, im, "train4", TK)
    i, j = int(np.flatnonzero(D["daty"] == data)[0]), TK.index(t)
    sig = np.zeros((len(D["daty"]), len(TK)), np.int8)
    sig[i - 5:i + 2, j] = 1
    o = gv.schet(D, sig, vhod="d1", tau=None, sost=True)
    (_, _, e, x, s, br, *_), = o["tr"]
    assert D["daty"][e] < data <= D["daty"][x] and s == 1
    assert abs(br) < 0.5


def test_a_dvizhok_primenyaet_konsolidaciyu_tak_zhe_kak_v_csv():
    """Движок (data.load_ticker(corp=True)) берёт событие IRAO из того же файла: factor 100 на 20.01.2015, статус applied."""
    ev = fd.corp_actions("IRAO")
    assert list(zip(ev["date"].dt.date.astype(str), ev["factor"])) == [("2015-01-20", 100.0)]


def _zcf(x):
    s, k, z = me.asimmetriya(x), me.ekscess(x), me.Z95
    return z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36


def _ryad_s_vybrosom(v=0.046):
    x = np.random.default_rng(1).normal(-0.0003, 0.003, 1000)
    x[500] = v
    return x


def test_b_vybros_daet_ogromnyj_msr_pri_umerennom_sharpe():
    """Воспроизведение Б (проходит сейчас): Шарп ≈ −2, а mSR ≈ −25, потому что квантиль Корниша–Фишера ушёл выше нуля."""
    x = _ryad_s_vybrosom()
    sr = x.mean() / np.std(x, ddof=1) * np.sqrt(252)
    assert -2.5 < sr < -1.5 and _zcf(x) > 0
    assert me.msr(x) < -20


@pytest.mark.xfail(strict=True, reason="Б: msr() возвращает число при квантиле КФ ≥ 0 (вне области КФ); к2 S12 train4 lag 660: "
                                       "один день 09.04.2018, асимметрия 2,5, эксцесс 42, mSR −51,8 при Шарпе −0,68")
def test_b_msr_vne_oblasti_kf_ne_chislo():
    x = _ryad_s_vybrosom()
    assert _zcf(x) > 0
    assert np.isnan(me.msr(x))


def _smen(S, dni=None):
    """Число смен стороны (тикер-дней); dni — маска строк t ≥ 1, где смена засчитывается."""
    d = np.diff(S.astype(int), axis=0) != 0
    return int(d.sum() if dni is None else d[dni[1:]].sum())


def _ne_rebal(daty):
    """Строки, в которые недельное решение меняться не может: день — не последний торговый день своей недели (kv.rebal)."""
    iso = pd.DatetimeIndex(daty).isocalendar()
    wk = iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy()
    return np.r_[wk[1:] == wk[:-1], False]


@pytest.fixture(scope="module")
def D():
    raw, im = bary()
    return gv.iz_barov(raw, im, "train4", TK)


@pytest.mark.parametrize("sid", ["S8", "S9"])
def test_v_placebo_nedelnogo_secheniya_ne_menyaetsya_vnutri_nedeli(D, sid):
    nr = _ne_rebal(D["daty"])
    c = _smen(gv.signaly(D, sid), nr)
    p = [_smen(gv.signaly(D, sid, ("placebo", gv.SEED + s)), nr) for s in range(3)]
    assert max(p) <= c, (c, p)


@pytest.mark.parametrize("sid", ["S8", "S9"])
def test_v_placebo_nedelnogo_secheniya_oborot_kak_u_centra(D, sid):
    c = _smen(gv.signaly(D, sid))
    p = [_smen(gv.signaly(D, sid, ("placebo", gv.SEED + s))) for s in range(3)]
    assert c > 0 and max(p) <= 2 * c, (c, p)


def test_v_placebo_s7_oborot_kak_u_centra(D):
    """В, S7 после починки: одна перестановка на весь ряд — оборот плацебо ≈ центру (не в разы выше)."""
    c = _smen(gv.signaly(D, "S7"))
    p = [_smen(gv.signaly(D, "S7", ("placebo", gv.SEED + s))) for s in range(3)]
    assert c > 0 and max(p) <= 1.5 * c and min(p) >= 0.5 * c, (c, p)


def test_v_placebo_ne_trogaet_ceny_i_neytralnost(D):
    """Отброшенные гипотезы В: плацебо не переставляет доходности (цены в schet — D['o'], D['cr'], не зависят от ctl)
    и сохраняет обе стороны книги в каждой строке, где у центра они есть."""
    sk = kv.rebal(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], "S8"), D["daty"], "nedelya")
    pl = kv.placebo_sechenie(sk, gv.SEED)
    fin = np.isfinite(sk)
    assert np.array_equal(np.sort(np.where(fin, sk, -9), axis=1), np.sort(np.where(np.isfinite(pl), pl, -9), axis=1))
    S = gv.signaly(D, "S8", ("placebo", gv.SEED))
    ryad = np.abs(S).sum(axis=1) > 0
    assert ((S[ryad] > 0).any(axis=1) & (S[ryad] < 0).any(axis=1)).all()


@pytest.mark.xfail(strict=True, reason="В': к2 сечения — np.roll всей панели, строки прогрева без скора (NaN) попадают в окно; "
                                       "train4 S8: доля времени в позиции 0,87–0,92 против 0,997 у центра")
def test_v_k2_secheniya_ne_zanosit_progrev_v_okno(D):
    sk = kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], "S8")
    w = D["win"]
    for lag in pk.k2_lags(gv.N_POVT)[:3]:
        assert np.isfinite(kv.k2(sk, lag)[w]).any(axis=1).mean() >= np.isfinite(sk[w]).any(axis=1).mean() - 0.01, lag


def test_g_vtoraya_proverka_sdviga_s8_po_prirashcheniyam(D):
    """Г: Δскор S8 на d связан с r[d−21] (входит в окно) и r[d−252] (выходит), не с r[d+1]; подложенная утечка видна на d+1."""
    c = D["c"]
    r = np.log(c).diff()

    def prof(x, s):
        a, b = x.diff().to_numpy().ravel(), r.shift(-s).to_numpy().ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[m], b[m])[0, 1]

    sk, leak = ks.skor_s8(c), 2 * ks.pct(np.log(c.shift(-1) / c.shift(252))) - 1
    p = {s: prof(sk, s) for s in (-21, -1, 0, 1)}
    assert p[-21] > 0.1 and max(abs(p[s]) for s in (-1, 0, 1)) < 0.05, p
    assert prof(leak, 1) > 0.1


@pytest.mark.parametrize("sid", ["S10", "S11", "S12"])
def test_d_k2_impulsa_sdvigaet_signal_otnositelno_cen(D, sid):
    """Д (гипотеза «сдвигаются обе» отброшена): к2 — циклический сдвиг сигнала тикера на k дней, цены те же; лаг 0 — центр."""
    S0 = gv.signaly(D, sid)
    assert np.array_equal(gv.signaly(D, sid, ("k2", 0)), S0)
    n = len(S0)
    for lag in pk.k2_lags(gv.N_POVT)[:3]:
        S = gv.signaly(D, sid, ("k2", lag))
        k = pk.MIN_LAG + (lag - pk.MIN_LAG) % max(n - 2 * pk.MIN_LAG + 1, 1)
        ozhid = np.roll(S0, k, axis=0)
        ozhid[-1] = 0
        assert np.array_equal(S, ozhid)
        if S0.any():
            assert not np.array_equal(S, S0)
