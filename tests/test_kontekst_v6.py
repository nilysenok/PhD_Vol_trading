"""Контекст сигнала v6 (К9–К16, К20–К24): синтетика с известным ответом, «выкл.» = v5, причинность, мутация."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_rel_sektor as rel
from fastbt import e1_sektor as sek
from fastbt import filtr_e1 as f
from fastbt import kontekst_v6 as kv

N = 300
DATES = pd.bdate_range("2020-01-01", periods=N)
D64 = DATES.to_numpy()
PIT = {"T": [(kv.DAWN, "IDX")]}


def ser(x):
    return pd.Series(np.asarray(x, float), index=DATES)


def dn(**kw):
    return kv.Dannye(t="T", dates=D64, **kw)


def test_vykl_vse_net_vozvrashchaet_true_bez_ryadov():
    lo, sr = kv.long_short({}, dn())
    assert lo.all() and sr.all() and len(lo) == N
    pt = dict(sektor_trend="net", shirina="net", relf_napr="net", post_shok="net", szhatie="net",
              rv_perevorot="net", obyom_anom="net", rezhim="net")
    lo, sr = kv.long_short(pt, dn())
    assert lo.all() and sr.all()


def test_sektor_trend_izvestnyj_otvet():
    idx = ser(100 + np.arange(N))                                    # индекс растёт, выше своей MA
    d = dn(sektor=sek.Ctx({"IDX": idx}, {"T": ser(np.ones(N))}, {"T": "IDX"}))
    lo, sr = kv.long_short({"sektor_trend": "po_trendu", "ma_n": 50}, d)
    assert not lo[:49].any() and lo[49:].all() and not sr.any()      # MA ещё нет → запрет
    lo2, sr2 = kv.long_short({"sektor_trend": "protiv", "ma_n": 50}, d)
    assert not lo2.any() and sr2[49:].all() and not sr2[:49].any()


def test_relsila_izvestnyj_otvet_i_bench():
    r = np.zeros(N)
    r[100] = 0.10
    close = ser(100 * np.exp(np.cumsum(r)))
    sg = ser(np.full(N, 0.01))
    flat = ser(np.full(N, 50.0))
    up = ser(50 * np.exp(np.cumsum(np.where(np.arange(N) == 100, 0.30, 0.0))))   # сектор прыгнул сильнее
    c = {"T": close}
    d = dn(rel_sektor=rel.Ctx({"IDX": up}, c, {"T": sg}, PIT), rel_rynok=kv.rynok_ctx(flat, c, {"T": sg}))
    pt = {"relf_napr": "po_sile", "relf_okno": 1, "relf_k": 1}
    lo, sr = kv.long_short({**pt, "relf_bench": "rynok"}, d)
    # бенч = рынок: +10 % к плоскому, σ̂ 1 % → rel = +10 → лонг
    assert lo[100] and not sr[100] and not lo[:100].any()
    lo, sr = kv.long_short({**pt, "relf_bench": "sektor"}, d)   # к сектору: −20 %/1 % → rel < −k → шорт
    assert sr[100] and not lo[100]
    lo, sr = kv.long_short({**pt, "relf_napr": "protiv", "relf_bench": "rynok"}, d)
    assert sr[100] and not lo[100]
    with pytest.raises(ValueError):
        kv.long_short({**pt, "relf_bench": "x"}, d)


def _shok_dannye(z_day=-0.05, Zv=0.0):
    r = np.zeros(N)
    r[100] = z_day
    Z = np.zeros(N)
    Z[100] = Zv
    return dn(close=100 * np.exp(np.cumsum(r)), sig22=np.full(N, 0.01), Z=Z)


def test_post_shok_okno_5_dnej_i_parametry():
    d = _shok_dannye()
    lo, sr = kv.long_short({"post_shok": "3:1"}, d)
    assert lo[100:105].all() and not lo[105] and not lo[:101 - 1].any() and not sr[100:105].any()
    sd = np.ones(N)
    assert np.array_equal(lo, f.post_shok(f.z_series(f.ret(d.close), d.sig22), d.Z, sd))
    assert not kv.long_short({"post_shok": "6:1"}, d)[0].any()       # шок 5σ < порога 6
    d2 = _shok_dannye(Zv=0.8)
    assert kv.long_short({"post_shok": "3:1"}, d2)[0][100]
    assert not kv.long_short({"post_shok": "2:0,5"}, d2)[0].any()    # рынок не спокоен (0,8 > 0,5)
    assert kv.parse_shok("2:0,5") == (2.0, 0.5)
    for bad in ("3", "0:1", "a:b", "3:-1"):
        with pytest.raises(ValueError):
            kv.parse_shok(bad)


def test_szhatie_obyom_perevorot_izvestnyj_otvet():
    cl = np.full(N, 100.0)
    lo, sr = kv.long_short({"szhatie": "da"}, dn(close=cl, high=cl + 1, low=cl - 1))
    assert np.array_equal(lo, sr) and not lo[:138].any() and lo[138:].all()   # 20 + 120 − 2 дней истории
    vol = np.full(N, 100.0)
    vol[50] = 1000.0
    lo, sr = kv.long_short({"obyom_anom": "da"}, dn(close=cl, volume=vol, sig22=np.full(N, 0.01)))
    assert list(np.flatnonzero(lo)) == [50] and np.array_equal(lo, sr)
    s5 = np.ones(N)
    s5[30] = 2.0                                   # U = 4 > 3 в день 30, на 31-й спад
    lo, _ = kv.long_short({"rv_perevorot": "da"}, dn(sig5=s5, sig66=np.ones(N)))
    assert list(np.flatnonzero(lo)) == [31]


def test_rezhim_i_oshibki():
    flag = np.arange(N) % 2 == 0
    lo, sr = kv.long_short({"rezhim": "imoex_ma200"}, dn(rezhim={"ma200": flag}))
    assert np.array_equal(lo, flag) and np.array_equal(sr, flag)
    with pytest.raises(ValueError):
        kv.long_short({"rezhim": "rank08"}, dn(rezhim={"ma200": flag}))    # нужного флага нет
    with pytest.raises(ValueError):
        kv.long_short({"rezhim": "zzz"}, dn())
    with pytest.raises(ValueError):
        kv.long_short({"szhatie": "yes"}, dn())
    with pytest.raises(ValueError):
        kv.long_short({"obyom_anom": "da"}, dn(close=np.ones(N)))          # включена, рядов нет


def _build(a):
    c = {"T": ser(a["close"])}
    sg = {"T": ser(a["sig22"])}
    idx = ser(a["idx"])
    return dn(close=a["close"], high=a["high"], low=a["low"], volume=a["volume"], sig5=a["sig5"],
              sig22=a["sig22"], sig66=a["sig66"], Z=a["Z"], rezhim={"ma200": a["fl"]},
              sektor=sek.Ctx({"IDX": idx}, c, {"T": "IDX"}),
              rel_sektor=rel.Ctx({"IDX": idx}, c, sg, PIT))


def _arrays(seed=3):
    g = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(g.normal(0, 0.02, N)))
    return dict(close=close, high=close * 1.01, low=close * 0.99, volume=g.uniform(50, 150, N),
                sig5=g.uniform(0.01, 0.03, N), sig22=g.uniform(0.01, 0.03, N), sig66=g.uniform(0.01, 0.03, N),
                Z=g.normal(0, 1, N), idx=100 * np.exp(np.cumsum(g.normal(0, 0.01, N))), fl=g.random(N) > 0.5)


PT_ALL = dict(sektor_trend="po_trendu", shirina="protiv", ma_n=20, porog_shiriny=0.5, relf_napr="po_sile",
              relf_okno=3, relf_k=0.5, post_shok="1:2", szhatie="da", rv_perevorot="da", obyom_anom="da",
              rezhim="imoex_ma200")
def _causal(pt=None, ks=range(150, 250, 3)) -> bool:
    """Флаги до дня K не меняются, если испортить все ряды после K (для набора K)."""
    return all(_causal_k(pt, k) for k in ks)


def _causal_k(pt, K) -> bool:
    a = _arrays()
    base = kv.long_short(pt or PT_ALL, _build(a))
    g = np.random.default_rng(9)
    b = {k: (v.copy() if v.dtype != bool else ~v) for k, v in a.items()}
    for k, v in b.items():
        if k != "fl":
            v[K + 1:] = v[K + 1:] * g.uniform(0.3, 3, N - K - 1)
        else:
            v[:K + 1] = a[k][:K + 1]
    pert = kv.long_short(pt or PT_ALL, _build(b))
    return all(np.array_equal(x[:K + 1], y[:K + 1]) for x, y in zip(base, pert))


def test_prichinnost_vse_osi():
    assert _causal()
    for one in ({"post_shok": "1:2"}, {"szhatie": "da"}, {"rv_perevorot": "da"}, {"obyom_anom": "da"},
                {"relf_napr": "po_sile", "relf_okno": 3}, {"sektor_trend": "po_trendu", "ma_n": 20}):
        assert _causal(one), one


def test_mutatsiya_okno_s_budushchim_lovitsya(monkeypatch):
    """Подложено: окно шока видит день d+1 — проверка причинности обязана упасть."""
    orig = f.win_any
    monkeypatch.setattr(kv.f, "win_any", lambda x, w: orig(np.r_[np.asarray(x, bool)[1:], False], w))
    assert not _causal({"post_shok": "1:2"})
