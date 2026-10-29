"""Независимые тесты V6-6 (kontekst_v6: К9–К16, К20–К24). Тестировщик, не автор.
Эталоны — циклы по формулам контракта, записанным здесь заново; случайные ряды со своим seed."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_rel_sektor as rel
from fastbt import e1_sektor as sek
from fastbt import kontekst_v6 as kv

N = 320
DATES = pd.bdate_range("2019-01-01", periods=N)
D64 = DATES.to_numpy()


def ser(x):
    return pd.Series(np.asarray(x, float), index=DATES)


def mk(seed=0):
    g = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(g.normal(0, 0.02, N)))
    close2 = 50 * np.exp(np.cumsum(g.normal(0, 0.02, N)))
    return dict(
        close=close, close2=close2, high=close * (1 + g.uniform(0, 0.01, N)), low=close * (1 - g.uniform(0, 0.01, N)),
        volume=g.uniform(50, 150, N), sig5=g.uniform(0.01, 0.03, N), sig22=g.uniform(0.01, 0.03, N),
        sig66=g.uniform(0.01, 0.03, N), Z=g.normal(0, 1, N),
        idx=100 * np.exp(np.cumsum(g.normal(0, 0.01, N))), mkt=100 * np.exp(np.cumsum(g.normal(0, 0.01, N))),
        fl=g.random(N) > 0.5)


def build(a):
    c = {"T": ser(a["close"]), "U": ser(a["close2"])}
    sg = {"T": ser(a["sig22"]), "U": ser(a["sig22"])}
    pit = {"T": [(kv.DAWN, "IDX")], "U": [(kv.DAWN, "IDX")]}
    return kv.Dannye(
        t="T", dates=D64, close=a["close"], high=a["high"], low=a["low"], volume=a["volume"], sig5=a["sig5"],
        sig22=a["sig22"], sig66=a["sig66"], Z=a["Z"], rezhim={"ma200": a["fl"], "rank08": ~a["fl"]},
        sektor=sek.Ctx({"IDX": ser(a["idx"])}, c, {"T": "IDX", "U": "IDX"}),
        rel_sektor=rel.Ctx({"IDX": ser(a["idx"])}, c, sg, pit),
        rel_rynok=kv.rynok_ctx(ser(a["mkt"]), c, sg))


# ---------- эталоны ----------
def ref_sma(x, n):
    o = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        o[i] = np.mean(x[i - n + 1:i + 1])
    return o


def ref_trend(idx, n):
    ma = ref_sma(idx, n)
    up = np.array([(not np.isnan(m)) and c > m for c, m in zip(idx, ma)])
    dn = np.array([(not np.isnan(m)) and c < m for c, m in zip(idx, ma)])
    return up, dn


def ref_shok(close, sig22, Z, z0, m0):
    n = len(close)
    z = np.full(n, np.nan)
    for d in range(2, n):
        z[d] = np.log(close[d] / close[d - 1]) / sig22[d - 1]
    dn = np.zeros(n, bool)
    up = np.zeros(n, bool)
    for d in range(n):
        for k in range(max(0, d - 4), d + 1):
            if abs(Z[k]) <= m0 and not np.isnan(z[k]):
                dn[d] |= -z[k] >= z0
                up[d] |= z[k] >= z0
    return dn, up


def ref_szhatie(high, low, close):
    n = len(close)
    rg = np.full(n, np.nan)
    for d in range(19, n):
        rg[d] = (high[d - 19:d + 1].max() - low[d - 19:d + 1].min()) / close[d]
    out = np.zeros(n, bool)
    for d in range(19 + 119, n):
        out[d] = rg[d] <= rg[d - 119:d + 1].min() + 1e-12
    return out


def ref_obyom(vol, close, sig22):
    n = len(vol)
    out = np.zeros(n, bool)
    for d in range(2, n):
        if d < 20:
            continue
        m = vol[d - 20:d].mean()
        if m > 0:
            z = np.log(close[d] / close[d - 1]) / sig22[d - 1]
            out[d] = vol[d] / m > 3 and abs(z) < 0.5
    return out


def ref_perevorot(s5, s66):
    n = len(s5)
    u = (s5 / s66) ** 2
    out = np.zeros(n, bool)
    for d in range(11, n):
        was = any(u[k] > 3 for k in range(d - 10, d))
        out[d] = was and s5[d] < s5[d - 1]
    return out


# ---------- «выкл.» ----------
def test_vykl_ne_chitaet_ryady_i_pustoj_kalendar():
    junk = kv.Dannye(t="T", dates=D64, close=np.full(N, np.nan), Z=np.full(N, np.nan))
    pt = dict(sektor_trend="net", shirina="net", relf_napr="net", post_shok="net", szhatie="net",
              rv_perevorot="net", obyom_anom="net", rezhim="net")
    lo, sr = kv.long_short(pt, junk)
    assert lo.dtype == bool and lo.all() and sr.all() and len(lo) == N
    lo, sr = kv.long_short(pt, kv.Dannye(t="T", dates=D64[:0]))
    assert len(lo) == 0 and len(sr) == 0
    assert kv.long_short(None or {}, kv.Dannye(t="T", dates=D64[:0]))[0].shape == (0,)


def test_vkljuchennaya_os_ne_vypuskaet_dannye_v_nol_dlinu_tolko_obshchie():
    # одна ось включена, остальные «выкл.»: результат равен результату этой одной оси
    a = mk(1)
    d = build(a)
    one = kv.long_short({"obyom_anom": "da"}, d)
    withnet = kv.long_short({"obyom_anom": "da", "szhatie": "net", "post_shok": "net", "rezhim": "net"}, d)
    assert np.array_equal(one[0], withnet[0]) and np.array_equal(one[1], withnet[1])


# ---------- К9–К12 ----------
@pytest.mark.parametrize("n", [20, 50])
def test_sektor_trend_protiv_ciklicheskogo_etalona(n):
    a = mk(2)
    d = build(a)
    up, dn = ref_trend(a["idx"], n)
    lo, sr = kv.long_short({"sektor_trend": "po_trendu", "ma_n": n}, d)
    assert lo.tolist() == up.tolist() and sr.tolist() == dn.tolist()
    lo, sr = kv.long_short({"sektor_trend": "protiv", "ma_n": n}, d)
    assert lo.tolist() == dn.tolist() and sr.tolist() == up.tolist()


def test_sektor_padayushchiy_indeks_razreshaet_shorty_i_ma_na_den_d():
    idx = 200 - np.arange(N, dtype=float)
    a = mk(3)
    a["idx"] = idx
    lo, sr = kv.long_short({"sektor_trend": "po_trendu", "ma_n": 30}, build(a))
    assert not lo.any() and sr[29:].all() and not sr[:29].any()
    # индекс ниже MA на день d: падение последнего дня переворачивает флаг в d, но не в d-1
    idx2 = np.full(N, 100.0)
    idx2[200] = 90.0
    a["idx"] = idx2
    lo, sr = kv.long_short({"sektor_trend": "po_trendu", "ma_n": 30}, build(a))
    assert sr[200] and not sr[199] and not lo[200]


def test_shirina_protiv_etalona_dvuh_tikerov():
    a = mk(4)
    n, p = 20, 0.5
    cs = [a["close"], a["close2"]]
    B = np.full(N, np.nan)
    for d in range(n - 1, N):
        votes = [c[d] > c[d - n + 1:d + 1].mean() for c in cs]
        B[d] = np.mean(votes)
    d_ = build(a)
    for mode in ("po_trendu", "protiv"):
        lo, sr = kv.long_short({"shirina": mode, "ma_n": n, "porog_shiriny": p}, d_)
        hi, low = np.nan_to_num(B, nan=-1) >= p, (~np.isnan(B)) & (B <= 1 - p)
        el, es = (hi, low) if mode == "po_trendu" else (low, hi)
        assert lo.tolist() == el.tolist() and sr.tolist() == es.tolist()
    assert not lo[:n - 1].any() and not sr[:n - 1].any()      # нет истории — запрет


# ---------- К13–К16 ----------
def ref_rel(close, idx, sig, w):
    n = len(close)
    out = np.full(n, np.nan)
    r = np.r_[np.nan, np.diff(np.log(close))]
    ri = np.r_[np.nan, np.diff(np.log(idx))]
    for d in range(w, n):
        s = (r[d - w + 1:d + 1] - ri[d - w + 1:d + 1]).sum()
        out[d] = s / (sig[d - 1] * np.sqrt(w))
    return out


@pytest.mark.parametrize("bench,key", [("sektor", "idx"), ("rynok", "mkt")])
@pytest.mark.parametrize("w", [1, 5])
def test_relsila_protiv_etalona(bench, key, w):
    a = mk(5)
    d = build(a)
    rl = ref_rel(a["close"], a[key], a["sig22"], w)
    k = 0.7
    pt = {"relf_okno": w, "relf_k": k, "relf_bench": bench}
    with np.errstate(invalid="ignore"):
        up, dn = rl >= k, rl <= -k
    lo, sr = kv.long_short({**pt, "relf_napr": "po_sile"}, d)
    assert lo.tolist() == up.tolist() and sr.tolist() == dn.tolist()
    lo, sr = kv.long_short({**pt, "relf_napr": "protiv"}, d)
    assert lo.tolist() == dn.tolist() and sr.tolist() == up.tolist()


def test_relsila_sigma_vchera_a_ne_segodnya():
    a = mk(6)
    d = build(a)
    pt = {"relf_napr": "po_sile", "relf_okno": 3, "relf_k": 0.5, "relf_bench": "sektor"}
    base = kv.long_short(pt, d)
    a2 = dict(a)
    a2["sig22"] = a["sig22"].copy()
    a2["sig22"][250] *= 50.0                       # σ̂ дня 250 влияет с 251
    r2 = kv.long_short(pt, build(a2))
    assert base[0][:251].tolist() == r2[0][:251].tolist() and base[1][:251].tolist() == r2[1][:251].tolist()


# ---------- К20 ----------
@pytest.mark.parametrize("spec,z0,m0", [("1:2", 1.0, 2.0), ("2:0,5", 2.0, 0.5), ("3:1", 3.0, 1.0)])
def test_post_shok_protiv_etalona(spec, z0, m0):
    a = mk(7)
    dn, up = ref_shok(a["close"], a["sig22"], a["Z"], z0, m0)
    lo, sr = kv.long_short({"post_shok": spec}, build(a))
    assert lo.tolist() == dn.tolist() and sr.tolist() == up.tolist()
    assert lo.any() or sr.any() or spec == "3:1"


def test_post_shok_granica_Z_vklyuchitelno():
    c = np.full(N, 100.0)
    c[100:] *= np.exp(-0.05)                       # шок -5σ в день 100 (σ̂=0,01)
    Z = np.zeros(N)
    Z[100] = 1.0
    d = kv.Dannye(t="T", dates=D64, close=c, sig22=np.full(N, 0.01), Z=Z)
    lo, _ = kv.long_short({"post_shok": "3:1"}, d)
    assert lo[100:105].all() and not lo[105] and not lo[99]
    Z[100] = 1.0 + 1e-9
    lo, _ = kv.long_short({"post_shok": "3:1"}, d)
    assert not lo.any()                           # |Z| > m0 -> нет


# ---------- К21–К23 ----------
def test_szhatie_protiv_etalona():
    a = mk(8)
    exp = ref_szhatie(a["high"], a["low"], a["close"])
    assert exp.any() and not exp.all()
    lo, sr = kv.long_short({"szhatie": "da"}, build(a))
    assert lo.tolist() == exp.tolist() == sr.tolist()


def test_obyom_protiv_etalona_i_granica_trojki():
    a = mk(9)
    a["volume"] = a["volume"].copy()
    for k in (60, 120, 200):
        a["volume"][k] = 5000.0
    a["close"] = a["close"].copy()
    exp = ref_obyom(a["volume"], a["close"], a["sig22"])
    lo, sr = kv.long_short({"obyom_anom": "da"}, build(a))
    assert lo.tolist() == exp.tolist() == sr.tolist()
    # ровно 3x от среднего — не аномалия (строго >)
    vol = np.full(N, 100.0)
    vol[80] = 300.0
    cl = np.full(N, 100.0)
    d = kv.Dannye(t="T", dates=D64, close=cl, volume=vol, sig22=np.full(N, 0.01))
    assert not kv.long_short({"obyom_anom": "da"}, d)[0].any()
    vol[80] = 300.01
    assert kv.long_short({"obyom_anom": "da"}, d)[0].tolist() == (np.arange(N) == 80).tolist()


def test_perevorot_protiv_etalona():
    a = mk(10)
    s5 = a["sig5"].copy()
    s5[[40, 150]] = 0.2
    a["sig5"] = s5
    exp = ref_perevorot(s5, a["sig66"])
    assert exp.any()
    lo, sr = kv.long_short({"rv_perevorot": "da"}, build(a))
    assert lo.tolist() == exp.tolist() == sr.tolist()


# ---------- К24 ----------
def test_rezhim_ravny_flagu_i_vybor_klyucha():
    a = mk(11)
    d = build(a)
    lo, sr = kv.long_short({"rezhim": "imoex_ma200"}, d)
    assert lo.tolist() == a["fl"].tolist() == sr.tolist()
    lo, sr = kv.long_short({"rezhim": "rank08"}, d)
    assert lo.tolist() == (~a["fl"]).tolist()


# ---------- И-композиция ----------
def test_kompoziciya_est_peresechenie_otdelnyh_osey():
    a = mk(12)
    d = build(a)
    parts = [{"sektor_trend": "po_trendu", "ma_n": 20}, {"post_shok": "1:2"}, {"szhatie": "da"},
             {"obyom_anom": "da"}, {"rezhim": "imoex_ma200"}, {"relf_napr": "po_sile", "relf_okno": 3, "relf_k": 0.3}]
    lo = np.ones(N, bool)
    sr = np.ones(N, bool)
    pt = {}
    for p in parts:
        l, s = kv.long_short(p, d)
        lo &= l
        sr &= s
        pt.update(p)
    L, S = kv.long_short(pt, d)
    assert L.tolist() == lo.tolist() and S.tolist() == sr.tolist()


# ---------- причинность ----------
ALL = dict(sektor_trend="po_trendu", shirina="protiv", ma_n=20, porog_shiriny=0.5, relf_napr="po_sile", relf_okno=3,
           relf_k=0.4, relf_bench="rynok", post_shok="1:2", szhatie="net", rv_perevorot="da", obyom_anom="da",
           rezhim="imoex_ma200")


@pytest.mark.parametrize("pt", [ALL, {**ALL, "relf_bench": "sektor"}, {"szhatie": "da"}, {"post_shok": "1:2"},
                                {"rv_perevorot": "da"}, {"obyom_anom": "da"}, {"shirina": "protiv", "ma_n": 20},
                                {"rezhim": "imoex_ma200"}, {"rezhim": "rank08"},
                                {"relf_napr": "po_sile", "relf_okno": 3, "relf_k": 0.4, "relf_bench": "rynok"}])
@pytest.mark.parametrize("K", [130, 200, 260])
def test_prichinnost_porcha_posle_K(pt, K):
    a = mk(13)
    base = kv.long_short(pt, build(a))
    b = {k: v.copy() for k, v in a.items() if k != "fl"}
    b["fl"] = a["fl"].copy()
    g = np.random.default_rng(99)
    for k, v in b.items():
        if k == "fl":
            v[K + 1:] = ~v[K + 1:]
        else:
            v[K + 1:] = v[K + 1:] * g.uniform(0.2, 5, N - K - 1)
    pert = kv.long_short(pt, build(b))
    assert base[0][:K + 1].tolist() == pert[0][:K + 1].tolist()
    assert base[1][:K + 1].tolist() == pert[1][:K + 1].tolist()


# ---------- границы ----------
def test_ryad_koroche_okna_zapret_a_ne_oshibka():
    n = 15
    dd = D64[:n]
    g = np.random.default_rng(0)
    c = 100 + g.normal(0, 1, n).cumsum()
    d = kv.Dannye(t="T", dates=dd, close=c, high=c + 1, low=c - 1, volume=np.full(n, 10.0), sig5=np.full(n, 0.02),
                  sig22=np.full(n, 0.02), sig66=np.full(n, 0.02))
    for pt in ({"szhatie": "da"}, {"obyom_anom": "da"}, {"rv_perevorot": "da"}):
        lo, sr = kv.long_short(pt, d)
        assert len(lo) == n and not lo.any() and not sr.any()


def test_nan_v_ryadah_zapret():
    a = mk(14)
    a["close"] = a["close"].copy()
    a["close"][150] = np.nan
    assert np.isnan(np.log(a["close"][150] / a["close"][149]))
    lo2, _ = kv.long_short({"obyom_anom": "da"}, build(a))
    assert not lo2[150] and not lo2[151]          # |z| = NaN -> не аномалия
