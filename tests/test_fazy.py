"""Фазы рынка и сектора (Ж-099 пп. 2–3, Ж-101): синтетика с известным ответом + мутации."""
import numpy as np
import pandas as pd
import pytest

from fastbt import fazy

N, K = 600, 400            # длина ряда; день шока
PERVAYA = 274              # первая определённая позиция d: 22 доходности + 252 σ + лаг 1


def ryad(drift, shock, n=N, k=K, idx0="2015-01-05"):
    """Доходности drift ± amp (amp убывает → σ-ранг ≈ 0 → спокойно); разовый шок в день k."""
    t = np.arange(n)
    amp = 0.010 - 0.005 * t / n
    r = drift + np.where(t % 2 == 0, 1, -1) * amp
    r[0] = 0.0
    r[k] += shock
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range(idx0, periods=n))


def proverka(f, trend, n=N, k=K):
    """Известный ответ: тренд постоянен; стресс ровно на d = k+1…k+22; до PERVAYA — NaN."""
    assert f.faza.iloc[:PERVAYA].isna().all(), "до прогрева фаза должна быть NaN"
    for i in range(PERVAYA, n):
        v = "стресс" if k + 1 <= i <= k + 22 else "спокойно"
        assert f.faza.iloc[i] == f"{trend}_{v}", (i, f.faza.iloc[i])


def test_rost_i_padenie_i_stress_na_d_plus_1():
    for drift, shock, trend in ((0.002, 0.05, "рост"), (-0.002, -0.05, "падение")):
        s = ryad(drift, shock)
        proverka(fazy.faza_rynka(s.index, {"imoex": s}), trend)


def test_bokovik_na_konstante():
    s = pd.Series(100.0, index=pd.bdate_range("2015-01-05", periods=N))
    f = fazy.faza_rynka(s.index, {"imoex": s})
    assert (f.trend.iloc[200:] == "боковик").all()    # close == MA200 → ни рост, ни падение
    assert (f.vol.iloc[PERVAYA:] == "спокойно").all()  # σ=0: ранг строго меньших = 0


def test_daty_portfelya_podmnozhestvo_i_vne_kalendarya():
    s = ryad(0.002, 0.05)
    full = fazy.faza_rynka(s.index, {"imoex": s})
    sub = s.index[::7]
    assert fazy.faza_rynka(sub, {"imoex": s}).faza.equals(full.faza.loc[sub])
    vne = fazy.faza_rynka([pd.Timestamp("2015-01-03")], {"imoex": s})
    assert vne.faza.isna().all()


def test_mutaciya_faza_po_dannym_d_lovitsya(monkeypatch):
    s = ryad(0.002, 0.05)
    monkeypatch.setattr(fazy, "_LAG", 0)
    with pytest.raises(AssertionError):
        proverka(fazy.faza_rynka(s.index, {"imoex": s}), "рост")


def test_mutaciya_rang_po_vsemu_periodu_lovitsya(monkeypatch):
    s = ryad(0.002, 0.05)

    def rang_vse(sigma, n=252):
        v = sigma.dropna()
        r = pd.Series([(v < x).mean() for x in v], index=v.index)
        return r.reindex(sigma.index)

    monkeypatch.setattr(fazy, "_rang", rang_vse)
    with pytest.raises(AssertionError):
        proverka(fazy.faza_rynka(s.index, {"imoex": s}), "рост")


def test_mutaciya_ma_po_budushchim_cenam_lovitsya(monkeypatch):
    s = ryad(0.002, 0.05)
    monkeypatch.setattr(fazy, "_ma", lambda x, n: x.rolling(n).mean().shift(-(n - 1)))
    with pytest.raises(AssertionError):
        proverka(fazy.faza_rynka(s.index, {"imoex": s}), "рост")


def dannye_sektora():
    rynok = ryad(0.001, 0.0)                       # рынок без шока
    a = ryad(0.003, 0.05)                          # бумага A: шок в день K
    b = a * 2                                      # B: те же доходности
    c = ryad(0.003, 0.0)
    akcii = pd.DataFrame({"A": a, "B": b, "C": c, "LSRG": c})
    karta = {"A": "Сек", "B": "Сек", "C": "Один", "LSRG": "Строительство"}
    return {"imoex": rynok, "akcii": akcii, "karta": karta}


def test_sektor_svoya_faza_i_odinakovaya_dlya_oboih_tikerov():
    d = dannye_sektora()
    ix = d["imoex"].index
    fs = fazy.faza_sektora("Сек", ix, d)
    assert fs.faza.iloc[PERVAYA:K + 1].eq("рост_спокойно").all()
    assert fs.faza.iloc[K + 1:K + 23].eq("рост_стресс").all()
    assert fazy.faza_tikera("A", ix, d).faza.equals(fs.faza)
    assert fazy.faza_tikera("B", ix, d).faza.equals(fs.faza)


def _loo_faza_tikera(t, daty, d):
    """Мутация leave-one-out: тикер исключён из своего сектора."""
    karta = {k: v for k, v in d["karta"].items() if k != t}
    return fazy.faza_sektora(d["karta"][t], daty, {**d, "karta": karta})


def test_mutaciya_leave_one_out_lovitsya():
    d = dannye_sektora()
    ix = d["imoex"].index
    for t in ("A", "B"):                       # у честной версии фазы равны — у LOO нет
        assert fazy.faza_tikera(t, ix, d).faza.equals(fazy.faza_sektora("Сек", ix, d).faza)
    with pytest.raises(AssertionError):
        assert _loo_faza_tikera("A", ix, d).faza.equals(_loo_faza_tikera("B", ix, d).faza)
        assert _loo_faza_tikera("A", ix, d).faza.equals(fazy.faza_sektora("Сек", ix, d).faza)


def test_odin_tiker_i_lsrg_berut_fazu_rynka():
    d = dannye_sektora()
    ix = d["imoex"].index
    m = fazy.faza_rynka(ix, d).faza
    for sek in ("Один", "Строительство"):
        assert fazy.faza_sektora(sek, ix, d).faza.equals(m)
        assert fazy.otn_faza(sek, ix, d).isna().all()
    with pytest.raises(ValueError):
        fazy.faza_sektora("Нет такого", ix, d)


def test_mutaciya_sektor_po_budushchey_razmetke_lovitsya(monkeypatch):
    d = dannye_sektora()
    ix = d["imoex"].index
    monkeypatch.setattr(fazy, "_indeks_sektora",
                        lambda ret: (1 + ret.shift(-1).mean(axis=1).fillna(0)).cumprod())
    fs = fazy.faza_sektora("Сек", ix, d)
    with pytest.raises(AssertionError):
        assert fs.faza.iloc[PERVAYA:K + 1].eq("рост_спокойно").all()   # шок «приехал» на день раньше


def test_otn_faza_silnee_potom_slabee_s_lagom():
    r = pd.bdate_range("2015-01-05", periods=300)
    m = pd.Series(100 * np.exp(0.001 * np.arange(300)), index=r)
    lr = 0.003 * np.arange(300.0)
    lr[200:] -= 0.3                                        # обвал сектора в день 200
    a = pd.Series(100 * np.exp(lr), index=r)
    d = {"imoex": m, "akcii": pd.DataFrame({"A": a, "B": a}), "karta": {"A": "Сек", "B": "Сек"}}
    o = fazy.otn_faza("Сек", r, d)
    assert o.iloc[:61].isna().all() and o.iloc[61:201].eq("сильнее").all()
    assert o.iloc[201:261].eq("слабее").all()              # d=200 ещё «сильнее», обвал виден с d=201
    assert o.iloc[261:].eq("сильнее").all()                # обвал вышел из окна 60 дней
